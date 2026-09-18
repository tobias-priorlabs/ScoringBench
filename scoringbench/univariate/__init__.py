"""ScoringBench univariate modules, loaded only when requested."""
import importlib

from ..version import __version__

__all__ = ["config", "datasets", "wrappers", "models", "metrics", "cv", "runner", "results", "utils", "__version__"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = value
    return value
