"""Import the Wairagkar et al. brain_to_voice code from third_party/ without editing it.

Their ``lib/`` and ``models/`` are namespace packages (no ``__init__.py``) imported as top-level
``lib`` and ``models``. Generic names like these can collide with other packages, so we register
both names to point at the fetched copy before importing anything from them.

Their code targets Keras 2 (TF 2.10). On TF >= 2.16 we get Keras 2 from ``tf_keras``:
``TF_USE_LEGACY_KERAS=1`` makes ``tf.keras`` resolve to it, and ``lib/inference_utils.py`` does a
bare ``from keras.models import ...``, so ``keras`` is aliased to ``tf_keras`` as well.

Usage::

    from wvx import b2v
    data_utils = b2v.load("lib.data_utils")         # no TensorFlow import
    brain2voice = b2v.load("models.brain2voice")    # imports TensorFlow (Keras 2)
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import ModuleType

from wvx.config import REPO_ROOT

B2V_ROOT = REPO_ROOT / "third_party" / "brain_to_voice"

# Modules that import TensorFlow/Keras (directly or via their imports).
_NEEDS_KERAS = {"lib.model_utils", "lib.inference_utils", "models.brain2voice"}


def _register_namespaces() -> None:
    if not B2V_ROOT.is_dir():
        raise ImportError(
            f"{B2V_ROOT} is missing; run `python scripts/fetch_third_party.py` first"
        )
    for name in ("lib", "models"):
        want = [str(B2V_ROOT / name)]
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            mod.__path__ = want
            sys.modules[name] = mod
        elif list(getattr(mod, "__path__", [])) != want:
            raise ImportError(f"a different {name!r} is already imported: {mod!r}")


def use_keras2() -> None:
    """Make ``tf.keras`` and ``import keras`` resolve to Keras 2 (``tf_keras``)."""
    if os.environ.setdefault("TF_USE_LEGACY_KERAS", "1") != "1":
        raise ImportError("TF_USE_LEGACY_KERAS must be 1 for the brain_to_voice code")
    import tf_keras
    import tf_keras.models

    keras = sys.modules.get("keras")
    if keras is not None and keras is not tf_keras:
        raise ImportError("Keras 3 was imported before wvx.b2v; import wvx.b2v first")
    sys.modules["keras"] = tf_keras
    sys.modules["keras.models"] = tf_keras.models

    import tensorflow as tf

    if not keras2_active(tf):
        raise ImportError(
            "tf.keras is not Keras 2: TensorFlow was imported before TF_USE_LEGACY_KERAS=1 "
            "was set; import wvx.b2v before tensorflow"
        )


def keras2_active(tf: ModuleType) -> bool:
    """True if ``tf.keras`` is backed by Keras 2 (``tf_keras``)."""
    return tf.keras.layers.Dense.__module__.startswith("tf_keras")


def load(module: str) -> ModuleType:
    """Import one of their modules, e.g. ``"lib.data_utils"`` or ``"models.brain2voice"``."""
    if not module.startswith(("lib.", "models.")):
        raise ValueError(f"not a brain_to_voice module: {module!r}")
    _register_namespaces()
    if module in _NEEDS_KERAS:
        use_keras2()
    return importlib.import_module(module)
