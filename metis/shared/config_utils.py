"""Small config-handling helpers shared across pipeline and benchmark layers."""

from typing import Any


def none_safe(value: Any) -> Any:
    """Treat the YAML string ``"None"`` (case-insensitive) as Python ``None``.

    Many configs declare ``target: None`` but PyYAML loads it as the string
    ``"None"``. Use this helper consistently in loader and benchmark code
    paths instead of duplicating the same one-liner.
    """
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return value
