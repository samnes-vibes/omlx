# SPDX-License-Identifier: Apache-2.0
"""Compat shim for mlx-lm's broken ``AutoTokenizer.register()`` call.

The pinned mlx-lm commit's ``tokenizer_utils.py`` runs, at import time::

    AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)

passing a bare string as ``config_class`` instead of a ``PreTrainedConfig``
subclass. transformers' ``_LazyAutoMapping.register()`` unconditionally reads
``key.__module__`` on that argument, so the string key crashes with
``AttributeError: 'str' object has no attribute '__module__'`` on every
transformers release tested (4.57.x through 5.13.x) -- and upstream mlx-lm's
main branch still has this line as of 2026-07, so pinning a different
transformers version doesn't route around it.

``REGISTERED_TOKENIZER_CLASSES["NewlineTokenizer"] = NewlineTokenizer`` (the
part that actually matters for ``AutoTokenizer.from_pretrained`` to resolve
the class by name) already runs *before* the crash, so this shim only needs
to stop the subsequent ``TOKENIZER_MAPPING.register()`` call from raising on
non-class keys -- it restores the ``hasattr`` guard that the real class-based
registration path already relies on.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_installed = False


def install() -> None:
    """Patch ``_LazyAutoMapping.register`` to tolerate non-class keys. Idempotent."""
    global _installed
    with _LOCK:
        if _installed:
            return

        from transformers.models.auto.auto_factory import _LazyAutoMapping

        _original_register = _LazyAutoMapping.register

        def _patched_register(self, key, value, exist_ok=False):
            if not hasattr(key, "__module__"):
                self._extra_content[key] = value
                return
            return _original_register(self, key, value, exist_ok=exist_ok)

        _LazyAutoMapping.register = _patched_register
        _installed = True
