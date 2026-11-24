"""Persisted visual decoder K/V and scene graph geometry for single-image Qwen3-VL."""

from .schema import ImageMemory

__all__ = ["BuildConfig", "ImageMemory", "build_image_memory", "load_memory", "save_memory"]


def __getattr__(name):
    # Loading schemas or rendering the CLI help does not initialize a model.
    if name in {"BuildConfig", "build_image_memory"}:
        from . import builder

        return getattr(builder, name)
    if name in {"load_memory", "save_memory"}:
        from . import store

        return getattr(store, name)
    raise AttributeError(name)
