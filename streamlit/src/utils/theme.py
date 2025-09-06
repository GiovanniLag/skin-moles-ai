

def load_theme_config():
    """Load theme configuration from .streamlit/config.toml if it exists."""
    import os
    import toml

    config_path = os.path.join(os.path.dirname(__file__), '..', '..', '.streamlit', 'config.toml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = toml.load(f)
            return config.get('theme', {})
    return {}


def transparency(color: str, alpha: float) -> str:
    """Return a color string with the given transparency applied.

    The input color can be a hex string (e.g. ``#RRGGBB``) or an
    ``rgb(r, g, b)`` string. The output is an ``rgba(r, g, b, a)``
    string where ``a`` is the given alpha value (0.0 to 1.0).
    """
    import re

    hex_match = re.match(r'^#([0-9a-fA-F]{6})$', color)
    rgb_match = re.match(r'^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$', color)

    if hex_match:
        hex_value = hex_match.group(1)
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    elif rgb_match:
        r = int(rgb_match.group(1))
        g = int(rgb_match.group(2))
        b = int(rgb_match.group(3))
        return f'rgba({r}, {g}, {b}, {alpha})'
    else:
        raise ValueError(f"Invalid color format: {color}")