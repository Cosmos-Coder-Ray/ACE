"""Dict-based module registry for ACE components.

Usage:
    from ace.utils.registry import register, get

    @register("my_module")
    class MyModule(nn.Module): ...

    cls = get("my_module")
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Global registries keyed by category
_REGISTRIES: dict[str, dict[str, Any]] = {}


class RegistryError(Exception):
    """Raised when a registry operation fails."""


def register(name: str, category: str = "default") -> Callable[[T], T]:
    """Register a class or function under *name* in *category*.

    Can be used as a decorator::

        @register("mamba_block", category="layers")
        class MambaBlock(nn.Module): ...

    Args:
        name: Unique key within the category.
        category: Logical grouping (e.g. ``"layers"``, ``"heads"``).

    Returns:
        The original class/function, unmodified.

    Raises:
        RegistryError: If *name* is already registered in *category*.
    """

    def _decorator(cls: T) -> T:
        if category not in _REGISTRIES:
            _REGISTRIES[category] = {}
        if name in _REGISTRIES[category]:
            raise RegistryError(
                f"'{name}' is already registered in category '{category}'. "
                f"Existing: {_REGISTRIES[category][name]}"
            )
        _REGISTRIES[category][name] = cls
        return cls

    return _decorator


def get(name: str, category: str = "default") -> Any:
    """Retrieve a registered class/function by *name* from *category*.

    Args:
        name: Key used during registration.
        category: Logical grouping to search.

    Returns:
        The registered object.

    Raises:
        RegistryError: If *name* is not found in *category*.
    """
    if category not in _REGISTRIES or name not in _REGISTRIES[category]:
        available = list(_REGISTRIES.get(category, {}).keys())
        raise RegistryError(
            f"'{name}' not found in category '{category}'. " f"Available: {available}"
        )
    return _REGISTRIES[category][name]


def list_registered(category: str = "default") -> list[str]:
    """Return all registered names in *category*.

    Args:
        category: Logical grouping to list.

    Returns:
        Sorted list of registered names.
    """
    return sorted(_REGISTRIES.get(category, {}).keys())


def clear(category: str | None = None) -> None:
    """Clear registrations.  If *category* is ``None``, clear everything.

    Args:
        category: Optional category to clear. ``None`` clears all.
    """
    if category is None:
        _REGISTRIES.clear()
    else:
        _REGISTRIES.pop(category, None)


# ──────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    @register("test_module", category="test")
    class _TestModule:
        """Dummy module for smoke test."""

    assert get("test_module", category="test") is _TestModule
    assert list_registered("test") == ["test_module"]
    clear("test")
    assert list_registered("test") == []
    print("[PASS] registry smoke test passed")
