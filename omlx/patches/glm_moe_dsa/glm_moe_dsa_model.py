# Copyright © 2025 Apple Inc.

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import CacheList, KVCache
from .deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from .deepseek_v32 import Model as DSV32Model
from .sparse_mla import (
    block_indices_to_block_mask,
    q8_vup_flat,
    sparse_mla_block_table_attention,
    sparse_mla_qblock_attention,
    sparse_mla_attention,
    topk_indices_to_block_masks,
)

_SGLANG_GLM5_INDEX_TOPK_PATTERN = (
    "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSS"
    "FSFFFSFSSSFSFFSFFSSS"
)
_BLOCK_STATS_SEEN = set()


def _parse_topk_state(topk_state):
    topk_indices = topk_state
    block_indices = None
    prefix_rows = 0
    exact_block_table = None
    if isinstance(topk_state, tuple):
        if len(topk_state) == 4:
            topk_indices, block_indices, prefix_rows, exact_block_table = topk_state
        elif len(topk_state) == 3:
            topk_indices, block_indices, prefix_rows = topk_state
        else:
            topk_indices, block_indices = topk_state
    return topk_indices, block_indices, prefix_rows, exact_block_table


def _maybe_log_block_index_stats(
    block_indices: mx.array,
    *,
    layer_idx: int,
    query_length: int,
    key_length: int,
    k_block_size: int,
) -> None:
    if os.environ.get("MLX_LM_GLM_DSA_BLOCK_STATS", "0") != "1":
        return
    if block_indices.ndim != 4 or block_indices.shape[1] != 1:
        return

    q_blocks = block_indices.shape[2]
    budget = block_indices.shape[3]
    k_blocks = (key_length + k_block_size - 1) // k_block_size
    key = (layer_idx, query_length, key_length, q_blocks, budget)
    if os.environ.get("MLX_LM_GLM_DSA_BLOCK_STATS_ONCE", "1") == "1":
        if key in _BLOCK_STATS_SEEN:
            return
        _BLOCK_STATS_SEEN.add(key)

    try:
        rows = block_indices[0, 0].tolist()
    except Exception as exc:
        print(f"GLM_DSA_BLOCK_STATS layer={layer_idx} unavailable: {exc}")
        return

    row_counts = []
    union = set()
    total_valid = 0
    for row in rows:
        valid = [int(block) for block in row if int(block) < k_blocks]
        row_counts.append(len(set(valid)))
        union.update(valid)
        total_valid += len(valid)

    avg_row = sum(row_counts) / max(len(row_counts), 1)
    coverage = len(union) / max(k_blocks, 1)
    duplicate_ratio = 1.0 - (len(union) / max(total_valid, 1))
    print(
        "GLM_DSA_BLOCK_STATS "
        f"layer={layer_idx} L={query_length} K={key_length} "
        f"q_blocks={q_blocks} k_blocks={k_blocks} budget={budget} "
        f"union={len(union)} coverage={coverage:.4f} "
        f"avg_row_unique={avg_row:.2f} duplicate_ratio={duplicate_ratio:.4f}"
    )


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict
    attention_bias: bool
    rope_scaling: Dict = None
    rope_theta: Optional[float] = None
    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[Any] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2
    quantization: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]

        config_indexer_types = self.indexer_types
        prefill_mode = os.environ.get("MLX_LM_GLM_DSA_PREFILL_MODE", "").lower()
        indexcache_mode = os.environ.get("MLX_LM_GLM_DSA_INDEXCACHE", "").lower()
        env_pattern = os.environ.get("MLX_LM_GLM_DSA_INDEX_TOPK_PATTERN")
        indexcache_disabled = indexcache_mode in {
            "0",
            "false",
            "off",
            "none",
            "disable",
            "disabled",
        }
        indexcache_requested = (
            prefill_mode in {"sglang", "indexcache"}
            or indexcache_mode in {"1", "true", "sglang", "glm5", "glm-5"}
        )
        pattern_is_sglang = (
            env_pattern is not None
            and env_pattern.strip().lower()
            in {"sglang", "glm5", "glm-5", "recommended"}
        )
        indexcache_recall_risk_allowed = os.environ.get(
            "MLX_LM_GLM_DSA_ALLOW_INDEXCACHE_RECALL_RISK", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if (
            not indexcache_disabled
            and (indexcache_requested or pattern_is_sglang)
            and not indexcache_recall_risk_allowed
        ):
            raise ValueError(
                "SGLang-style GLM DSA IndexCache is disabled by default because "
                "it failed a 32K multi-needle recall test on GLM-5.2-oQ4. "
                "Set MLX_LM_GLM_DSA_ALLOW_INDEXCACHE_RECALL_RISK=1 to enable "
                "this speed/recall-risk mode explicitly."
            )
        if not env_pattern and not indexcache_disabled and indexcache_requested:
            env_pattern = "sglang"
        if env_pattern:
            env_pattern = env_pattern.strip()
            if env_pattern.lower() in {"sglang", "glm5", "glm-5", "recommended"}:
                env_pattern = _SGLANG_GLM5_INDEX_TOPK_PATTERN
            self.index_topk_pattern = env_pattern
            self.indexer_types = None
        if "MLX_LM_GLM_DSA_INDEX_TOPK_FREQ" in os.environ:
            self.index_topk_freq = int(os.environ["MLX_LM_GLM_DSA_INDEX_TOPK_FREQ"])
            self.index_topk_pattern = None
            if config_indexer_types is not None and os.environ.get(
                "MLX_LM_GLM_DSA_ALLOW_NEW_INDEXERS", "0"
            ) != "1":
                freq = max(self.index_topk_freq, 1)
                full_count = 0
                indexer_types = []
                for layer_type in config_indexer_types:
                    if layer_type == "full":
                        indexer_types.append(
                            "full" if full_count % freq == 0 else "shared"
                        )
                        full_count += 1
                    else:
                        indexer_types.append("shared")
                self.indexer_types = indexer_types
            else:
                self.indexer_types = None
        if "MLX_LM_GLM_DSA_INDEX_SKIP_TOPK_OFFSET" in os.environ:
            self.index_skip_topk_offset = int(
                os.environ["MLX_LM_GLM_DSA_INDEX_SKIP_TOPK_OFFSET"]
            )
            if self.index_topk_pattern is None:
                self.indexer_types = None

        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                if isinstance(pattern, str):
                    if len(pattern) != self.num_hidden_layers:
                        raise ValueError(
                            "index_topk_pattern length must match "
                            f"num_hidden_layers ({len(pattern)} != "
                            f"{self.num_hidden_layers})."
                        )
                    pattern_types = [
                        {"F": "full", "S": "shared"}[c] for c in pattern
                    ]
                else:
                    pattern_types = list(pattern)
                if config_indexer_types is not None and os.environ.get(
                    "MLX_LM_GLM_DSA_ALLOW_NEW_INDEXERS", "0"
                ) != "1":
                    self.indexer_types = [
                        "full" if base == "full" and selected == "full" else "shared"
                        for base, selected in zip(config_indexer_types, pattern_types)
                    ]
                else:
                    self.indexer_types = pattern_types
            else:
                freq = max(self.index_topk_freq, 1)
                offset = self.index_skip_topk_offset
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        if self.indexer is not None:
            topk_state = self.indexer(x, qr, mask, cache=cache[1])
        else:
            topk_state = prev_topk_indices

        if L == 1:
            topk_indices, _, _, _ = _parse_topk_state(topk_state)
            if topk_indices is not None:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)

            # Ensure the indexer cache is evaluated even if the topk_indices are unused
            # to keep the graph from getting too large.
            if self.indexer is not None and cache is not None and cache[0] is not None:
                cache[0].keys = mx.depends(
                    cache[0].keys, (cache[1].keys, cache[1].values)
                )

            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )
            q_nope = self.embed_q(q_nope)
            output = scaled_dot_product_attention(
                q_nope,
                kv_latent,
                kv_latent,
                cache=cache,
                scale=self.scale,
                mask=pe_scores,
            )
            output = self.unembed_out(output)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output), topk_state

        topk_indices, block_indices, topk_prefix_rows, exact_block_table = (
            _parse_topk_state(topk_state)
        )

        # Ensure the indexer cache is evaluated even if the topk_indices are unused
        # to keep the graph from getting too large
        if self.indexer is not None and cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        if exact_block_table is not None and L > 1:
            block_k = int(
                os.environ.get("MLX_LM_GLM_DSA_BLOCK_TABLE_MLA_K_BLOCK", "8")
            )
            q_latent = self.embed_q(q_nope)
            block_table = (
                exact_block_table
                if exact_block_table.dtype == mx.uint32
                else exact_block_table.astype(mx.uint32)
            )
            output = sparse_mla_block_table_attention(
                q_latent,
                q_pe,
                kv_latent,
                k_pe,
                block_table,
                self.scale,
                k_block_size=block_k,
            )
            if output is not None:
                output_flat = q8_vup_flat(
                    output, self.unembed_out, key_length=kv_latent.shape[2]
                )
                if output_flat is None:
                    output = self.unembed_out(output)
                    output_flat = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                output = output_flat
                return self.o_proj(output), topk_state

        if block_indices is not None and L > 1:
            use_block_sdpa = os.environ.get(
                "MLX_LM_GLM_DSA_BLOCK_SPARSE_SDPA",
                os.environ.get("MLX_LM_GLM_DSA_BLOCK_UNION_SDPA", "1"),
            ) == "1" and L > 8
            if use_block_sdpa:
                use_block_index_sdpa = (
                    os.environ.get("MLX_LM_GLM_DSA_BLOCK_INDEX_SDPA", "1") == "1"
                )
                if use_block_index_sdpa:
                    block_indices_u32 = (
                        block_indices
                        if block_indices.dtype == mx.uint32
                        else block_indices.astype(mx.uint32)
                    )
                    stats_k_block_size = int(
                        os.environ.get(
                            "MLX_LM_GLM_DSA_BLOCK_SDPA_K_BLOCK",
                            os.environ.get(
                                "MLX_LM_GLM_DSA_BLOCK_BUDGET_K_BLOCK", "16"
                            ),
                        )
                    )
                    _maybe_log_block_index_stats(
                        block_indices_u32,
                        layer_idx=self.layer_idx,
                        query_length=L,
                        key_length=kv_latent.shape[2],
                        k_block_size=stats_k_block_size,
                    )
                    k = self.embed_q(kv_latent, transpose=False)
                    v = self.unembed_out(kv_latent)
                    split_qk_max_k = int(
                        os.environ.get("MLX_LM_GLM_DSA_SPLIT_QK_MAX_K", "65536")
                    )
                    if (
                        os.environ.get("MLX_LM_GLM_DSA_SPLIT_QK_SDPA", "0") == "1"
                        and kv_latent.shape[2] <= split_qk_max_k
                        and hasattr(mx.fast, "glm_dsa_attention")
                    ):
                        output = mx.fast.glm_dsa_attention(
                            q_nope,
                            q_pe,
                            k,
                            k_pe,
                            v,
                            block_indices_u32,
                            self.scale,
                        )
                    else:
                        k_pe_heads = mx.broadcast_to(
                            k_pe, k.shape[:-1] + k_pe.shape[-1:]
                        )
                        q = mx.concatenate([q_nope, q_pe], axis=-1)
                        k = mx.concatenate([k, k_pe_heads], axis=-1)
                        output = mx.fast.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            scale=self.scale,
                            mask="causal",
                            block_indices=block_indices_u32,
                        )
                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.o_proj(output), topk_state
                q_block_size = int(
                    os.environ.get(
                        "MLX_LM_GLM_DSA_BLOCK_SDPA_Q_BLOCK",
                        os.environ.get("MLX_LM_GLM_DSA_BLOCK_BUDGET_Q_BLOCK", "32"),
                    )
                )
                k_block_size = int(
                    os.environ.get(
                        "MLX_LM_GLM_DSA_BLOCK_SDPA_K_BLOCK",
                        os.environ.get("MLX_LM_GLM_DSA_BLOCK_BUDGET_K_BLOCK", "16"),
                    )
                )
                k = self.embed_q(kv_latent, transpose=False)
                v = self.unembed_out(kv_latent)
                k_pe_heads = mx.broadcast_to(k_pe, k.shape[:-1] + k_pe.shape[-1:])
                q = mx.concatenate([q_nope, q_pe], axis=-1)
                k = mx.concatenate([k, k_pe_heads], axis=-1)
                block_mask = block_indices_to_block_mask(
                    block_indices,
                    L=L,
                    K=kv_latent.shape[2],
                    q_block_size=q_block_size,
                    k_block_size=k_block_size,
                )
                if block_mask is not None:
                    output = mx.fast.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        scale=self.scale,
                        mask="causal",
                        block_mask=block_mask,
                    )
                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.o_proj(output), topk_state

        prefill_mode = os.environ.get("MLX_LM_GLM_DSA_PREFILL_MODE", "").lower()
        default_exact_prefill = (
            prefill_mode == ""
            and os.environ.get("MLX_LM_GLM_DSA_EXACT_PREFILL_DEFAULT", "1") != "0"
        )
        exact_prefill = prefill_mode in {"exact", "strict"} or default_exact_prefill
        # Exact block-token SDPA is still slightly faster around 10K, while
        # direct sparse MLA wins from 12K+ on the retained GLM-5.2 path.
        direct_sparse_mla_min_k = int(
            os.environ.get("MLX_LM_GLM_DSA_DIRECT_SPARSE_MLA_MIN_K", "11264")
        )
        direct_sparse_mla_default = (
            "1"
            if exact_prefill and kv_latent.shape[2] >= direct_sparse_mla_min_k
            else "0"
        )
        direct_sparse_mla_requested = (
            topk_indices is not None
            and L > 1
            and os.environ.get(
                "MLX_LM_GLM_DSA_DIRECT_SPARSE_MLA", direct_sparse_mla_default
            )
            == "1"
        )
        if direct_sparse_mla_requested:
            fast_topk_indices = (
                os.environ.get("MLX_LM_GLM_DSA_FAST_TOPK", "1") == "1"
                and hasattr(mx.fast, "dsa_topk_indices")
            )
            causal_prefix_indices = (
                fast_topk_indices
                and os.environ.get("MLX_LM_GLM_DSA_TOPK_CAUSAL_PREFIX_FASTPATH", "1")
                == "1"
            )
            q_latent = self.embed_q(q_nope)
            if (
                os.environ.get("MLX_LM_GLM_DSA_QBLOCK_UNION_MLA", "0") == "1"
                and hasattr(mx.fast, "dsa_topk_qblock_union")
                and hasattr(mx.fast, "glm_dsa_sparse_mla_qblock_attention")
            ):
                qblock_size = int(
                    os.environ.get("MLX_LM_GLM_DSA_QBLOCK_UNION_MLA_Q_BLOCK", "4")
                )
                qblock_capacity = int(
                    os.environ.get("MLX_LM_GLM_DSA_QBLOCK_UNION_MLA_CAPACITY", "4096")
                )
                output = sparse_mla_qblock_attention(
                    q_latent,
                    q_pe,
                    kv_latent,
                    k_pe,
                    topk_indices,
                    self.scale,
                    topk_valid_prefix=fast_topk_indices,
                    causal_prefix_indices=causal_prefix_indices,
                    causal_prefix_rows=topk_prefix_rows,
                    q_block_size=qblock_size,
                    capacity=qblock_capacity,
                )
                if output is not None:
                    output_flat = q8_vup_flat(
                        output, self.unembed_out, key_length=kv_latent.shape[2]
                    )
                    if output_flat is None:
                        output = self.unembed_out(output)
                        output_flat = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    output = output_flat
                    return self.o_proj(output), topk_state
            block_table_mode = os.environ.get(
                "MLX_LM_GLM_DSA_BLOCK_TABLE_SPARSE_MLA", "0"
            ).lower()
            use_block_table_mla = (
                topk_prefix_rows == 0
                and block_table_mode in {"1", "auto", "force"}
                and hasattr(mx.fast, "dsa_topk_to_block_table")
            )
            if use_block_table_mla:
                block_k = int(
                    os.environ.get("MLX_LM_GLM_DSA_BLOCK_TABLE_MLA_K_BLOCK", "8")
                )
                if block_table_mode != "force":
                    k_blocks = max(1, (kv_latent.shape[2] + block_k - 1) // block_k)
                    topk_size = max(1, topk_indices.shape[-1])
                    expected_blocks = k_blocks * (
                        1.0 - math.exp(topk_size * math.log1p(-1.0 / k_blocks))
                    )
                    expected_expansion = expected_blocks * block_k / topk_size
                    max_expansion = float(
                        os.environ.get(
                            "MLX_LM_GLM_DSA_BLOCK_TABLE_MLA_MAX_EXPANSION",
                            "1.5",
                        )
                    )
                    use_block_table_mla = expected_expansion <= max_expansion

            if use_block_table_mla:
                packed_block_table = (
                    block_k <= 16
                    and os.environ.get(
                        "MLX_LM_GLM_DSA_BLOCK_TABLE_MLA_PACKED", "1"
                    )
                    == "1"
                )
                block_table = mx.fast.dsa_topk_to_block_table(
                    topk_indices,
                    kv_latent.shape[2],
                    k_block_size=block_k,
                    causal=True,
                    packed=packed_block_table,
                )
                output = sparse_mla_block_table_attention(
                    q_latent,
                    q_pe,
                    kv_latent,
                    k_pe,
                    block_table,
                    self.scale,
                    k_block_size=block_k,
                )
                if output is not None:
                    output_flat = q8_vup_flat(
                        output, self.unembed_out, key_length=kv_latent.shape[2]
                    )
                    if output_flat is None:
                        output = self.unembed_out(output)
                        output_flat = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    output = output_flat
                    return self.o_proj(output), topk_state
            output = sparse_mla_attention(
                q_latent,
                q_pe,
                kv_latent,
                k_pe,
                topk_indices,
                self.scale,
                topk_valid_prefix=fast_topk_indices,
                causal_prefix_indices=causal_prefix_indices,
                causal_prefix_rows=topk_prefix_rows,
            )
            if output is not None:
                output_flat = q8_vup_flat(
                    output, self.unembed_out, key_length=kv_latent.shape[2]
                )
                if output_flat is None:
                    output = self.unembed_out(output)
                    output_flat = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                output = output_flat
                return self.o_proj(output), topk_state

        if (
            topk_indices is not None
            and L > 8
            and os.environ.get(
                "MLX_LM_GLM_DSA_EXACT_BLOCK_TOKEN_SDPA",
                "1" if exact_prefill else "0",
            )
            == "1"
        ):
            k = self.embed_q(kv_latent, transpose=False)
            k_pe_heads = mx.broadcast_to(k_pe, k.shape[:-1] + k_pe.shape[-1:])
            q = mx.concatenate([q_nope, q_pe], axis=-1)
            k = mx.concatenate([k, k_pe_heads], axis=-1)
            v = self.unembed_out(kv_latent)
            q_block_size = int(
                os.environ.get(
                    "MLX_LM_GLM_DSA_EXACT_BLOCK_SDPA_Q_BLOCK",
                    os.environ.get("MLX_LM_GLM_DSA_BLOCK_SDPA_Q_BLOCK", "32"),
                )
            )
            k_block_size = int(
                os.environ.get(
                    "MLX_LM_GLM_DSA_EXACT_BLOCK_SDPA_K_BLOCK",
                    os.environ.get(
                        "MLX_LM_GLM_DSA_BLOCK_SDPA_K_BLOCK",
                        os.environ.get("MLX_LM_GLM_DSA_BLOCK_BUDGET_K_BLOCK", "8"),
                    ),
                )
            )
            block_masks = topk_indices_to_block_masks(
                topk_indices,
                L=L,
                K=kv_latent.shape[2],
                q_block_size=q_block_size,
                k_block_size=k_block_size,
                causal_prefix_indices=(
                    os.environ.get("MLX_LM_GLM_DSA_FAST_TOPK", "1") == "1"
                    and os.environ.get(
                        "MLX_LM_GLM_DSA_TOPK_CAUSAL_PREFIX_FASTPATH", "1"
                    )
                    == "1"
                ),
                causal_prefix_rows=topk_prefix_rows,
            )
            if block_masks is not None:
                block_mask, block_token_mask = block_masks
                output = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=self.scale,
                    mask="causal",
                    block_mask=block_mask,
                    block_token_mask=block_token_mask,
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output), topk_state

        if direct_sparse_mla_requested:
            shape = list(topk_indices.shape)
            shape[-1] = kv_latent.shape[2]
            sparse_mask = mx.zeros(shape, dtype=mx.bool_)
            sparse_mask = mx.put_along_axis(
                sparse_mask, topk_indices, mx.array(True), axis=-1
            )
            if mask is not None:
                sparse_mask = sparse_mask & mask
            mask = sparse_mask

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), topk_state


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        r, topk_indices = self.self_attn(
            self.input_layernorm(x), mask, cache, prev_topk_indices
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk_indices


class GlmMoeDsaModel(DeepseekV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.layers = [
            GlmMoeDsaDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk_indices
            )

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)

    def sanitize(self, weights):
        weights = super().sanitize(weights)
        skip_prefixes = [
            f"model.layers.{i}.self_attn.indexer."
            for i, layer in enumerate(self.model.layers)
            if getattr(layer.self_attn, "skip_topk", False)
        ]
        if skip_prefixes:
            weights = {
                k: v
                for k, v in weights.items()
                if not any(k.startswith(prefix) for prefix in skip_prefixes)
            }
        return weights

    def make_cache(self):
        # Shared layers run no indexer, so they get no indexer KVCache.
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
