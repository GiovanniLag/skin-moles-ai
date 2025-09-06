"""Registry of available model weight configurations.

Each entry maps a human-readable weight name to a specification
containing the checkpoint file path and the expected input image size. The
``get_spec`` helper returns a dictionary with these fields along with a
boolean flag indicating whether the checkpoint file exists on disk. The
registry is deliberately simple so that additional models can be added
without modifying other parts of the code. The default active weight is
the first key in the registry.
"""

import os
from typing import Dict, Optional, Any

# Start with a minimal registry but prefer to discover checkpoints under
# the project's outputs/ directory. Entries store an absolute ckpt path.
_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _discover_checkpoints() -> None:
    """Populate _REGISTRY from known output locations if not already set.

    Priority order:
    1. outputs/byol/ckpts/*.ckpt (pretraining weights)
    2. outputs/dermanet/training/**/ (look for last.ckpt or best.ckpt)
    3. streamlit_app/weights/*.ckpt (fallback local bundle)
    """
    if _REGISTRY:
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # 1) BYOL checkpoints
    byol_dir = os.path.join(repo_root, "outputs", "byol", "ckpts")
    if os.path.isdir(byol_dir):
        for fn in sorted(os.listdir(byol_dir)):
            if fn.endswith(".ckpt"):
                path = os.path.join(byol_dir, fn)
                _REGISTRY[f"BYOL pretrain: {fn}"] = {"ckpt": path, "img_size": 224}
    # 2) Dermanet training checkpoints
    der_dir = os.path.join(repo_root, "outputs", "dermanet", "training")
    if os.path.isdir(der_dir):
        for root, _dirs, files in os.walk(der_dir):
            for candidate in ("last.ckpt", "best.ckpt"):
                if candidate in files:
                    path = os.path.join(root, candidate)
                    name = f"DermNet trained: {os.path.basename(root)} ({candidate})"
                    _REGISTRY[name] = {"ckpt": path, "img_size": 384}
    # 3) Fallback: bundled weights inside the streamlit app
    local_weights = os.path.join(os.path.dirname(__file__), "..", "weights")
    if os.path.isdir(local_weights):
        for fn in sorted(os.listdir(local_weights)):
            if fn.endswith(".ckpt"):
                path = os.path.join(local_weights, fn)
                _REGISTRY[f"Local: {fn}"] = {"ckpt": path, "img_size": 224}


# Ensure discovery runs once at import time
_discover_checkpoints()

# Fallback active weight is the first discovered entry or an explicit None
_ACTIVE: str = list(_REGISTRY.keys())[0] if _REGISTRY else ""


def list_weights() -> list[str]:
    """Return the list of available weight names in the registry.

    The registry is (lazily) discovered from disk; return an empty list
    if no checkpoints were found so the UI can handle that case.
    """
    return list(_REGISTRY.keys())


def get_spec(name: Optional[str] = None) -> Dict[str, Any]:
    """Return the spec for the given weight name (absolute ckpt path).

    Adds an ``exists`` boolean and leaves the absolute ``ckpt`` path in
    the returned dictionary. If the registry is empty the returned spec
    will be an empty dict.
    """
    key = name or _ACTIVE
    spec = dict(_REGISTRY.get(key, {}))
    ckpt_path = spec.get("ckpt")
    if ckpt_path:
        # Ensure absolute path and existence flag
        spec["ckpt"] = os.path.abspath(ckpt_path)
        spec["exists"] = os.path.exists(spec["ckpt"])
    else:
        spec["exists"] = False
    return spec


def set_active_weights(name: str) -> None:
    """Set the active weight configuration by name.

    Ignore unknown names.
    """
    global _ACTIVE
    if name in _REGISTRY:
        _ACTIVE = name


def get_active_weights() -> str:
    """Return the currently active weight name (or empty string).

    The UI should handle the empty case when no weights are present.
    """
    return _ACTIVE
