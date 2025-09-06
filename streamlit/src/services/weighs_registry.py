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
_ACTIVE: str = ""  # Name of the currently active weights

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
