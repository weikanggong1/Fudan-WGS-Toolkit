"""Fudan WGS Toolkit: GPU association analysis."""
__version__ = "0.9.0"
from .run import run_WGS_all

__all__ = ["prepare_WGS_data", "run_WGS_all"]


def __getattr__(name):
    if name == "prepare_WGS_data":
        from .prepare import prepare_WGS_data
        return prepare_WGS_data
    raise AttributeError(name)
