# SPDX-License-Identifier: Apache-2.0
"""Fused verify-length attention over TurboQuant 4-bit KV (compressed domain).

mlx-vlm's TurboQuantKVCache already fuses q_len=1 decode (compressed-domain
Metal kernels) and q_len>1 prefill for the Prod key codec. The gap this
module fills: q_len in [2, 32] with the default integer-bit MSE codec —
the speculative-verify regime (ngram spec, MTP, DFlash blocks) — which
otherwise falls back to dequantize + fp16 SDPA, materializing the full
fp16 KV and paying the O(T·D²) inverse-rotation matmul every verify step.

Public API:
  fused_verify_attention(cache, queries, keys_state, values_state, scale, mask)
      -> Optional[mx.array]  (None = unsupported config, caller falls back)
  reference_verify_attention(...)  — pure-MLX numerics reference
  set_enabled(flag) / is_enabled() — kill switch wired from model settings
      (turboquant_fused_kernel)
"""

from .fast import fused_verify_attention, is_enabled, set_enabled
from .reference import reference_verify_attention

__all__ = [
    "fused_verify_attention",
    "reference_verify_attention",
    "set_enabled",
    "is_enabled",
]
