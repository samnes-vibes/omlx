# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 ``glm_moe_dsa`` monkey-patch for mlx-lm.

Vendors the oMLX GLM-5.2 optimized mlx-lm model code without modifying the
pinned mlx-lm package. The public module name stays
``mlx_lm.models.glm_moe_dsa`` so mlx-lm's normal model loader can find it, but
the optimized helper modules remain private to ``omlx.patches.glm_moe_dsa`` and
do not replace ``mlx_lm.models.deepseek_v32`` or the shared MoE layers used by
other model families.
"""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

PATCH_SOURCE = "mlxlm-glm optimized-final mlx-lm snapshot"
CUSTOM_MLX_REF = "https://github.com/jundot/mlx@v0.31.2-omlx"

_APPLIED = False


def _missing_custom_mlx_symbols() -> list[str]:
    """Return expected custom MLX symbols missing from the installed runtime."""
    try:
        import mlx.core as mx
    except Exception:
        return []

    required = (
        "dsa_indexer_scores",
        "dsa_topk_indices",
        "glm_dsa_sparse_mla_attention",
        "glm_dsa_q8_vup_flat",
        "glm_moe_swiglu_down",
        "glm_moe_weighted_sum",
    )
    return [name for name in required if not hasattr(mx.fast, name)]


def _register_module() -> None:
    qualname = "mlx_lm.models.glm_moe_dsa"
    existing = sys.modules.get(qualname)
    if getattr(existing, "_OMLX_GLM_DSA_OPTIMIZED", False):
        module = existing
    else:
        module = importlib.import_module(f"{__name__}.glm_moe_dsa_model")
        module._OMLX_GLM_DSA_OPTIMIZED = True

    sys.modules[qualname] = module

    import mlx_lm.models as models_pkg

    models_pkg.glm_moe_dsa = module
    logger.info("Registered %s from oMLX optimized vendored module", qualname)


def apply_glm_moe_dsa_patch() -> bool:
    """Apply the GLM MoE DSA patch. Idempotent.

    Must run before ``mlx_lm.load()`` imports ``mlx_lm.models.glm_moe_dsa``.

    Returns True when oMLX registered its optimized vendored module, False when
    the patch was already applied or mlx-lm is unavailable.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.debug("mlx_lm not importable - glm_moe_dsa patch skipped")
        return False

    _register_module()
    from .generate_patch import apply_glm_moe_dsa_generate_patch

    apply_glm_moe_dsa_generate_patch()
    _APPLIED = True
    missing = _missing_custom_mlx_symbols()
    if missing:
        logger.warning(
            "GLM MoE DSA optimized patch applied, but custom MLX symbols are "
            "missing: %s. Install %s for the accelerated path.",
            ", ".join(missing),
            CUSTOM_MLX_REF,
        )
    logger.info("GLM MoE DSA optimized mlx-lm patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "apply_glm_moe_dsa_patch",
    "is_applied",
    "PATCH_SOURCE",
    "CUSTOM_MLX_REF",
]
