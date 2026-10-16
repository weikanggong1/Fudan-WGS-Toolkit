"""Fudan WGS Toolkit: GPU association analysis."""
__version__ = "0.10.0"
from .run import run_WGS_all_binanry

__all__ = ["prepare_WGS_data", "generate_simulated_data", "run_WGS_all_binanry", "run_WGS_continuous"]


def __getattr__(name):
    if name == "generate_simulated_data":
        from .simulate import generate_simulated_data
        return generate_simulated_data
    if name == "run_WGS_continuous":
        from .continuous_run import run_WGS_continuous
        return run_WGS_continuous
    if name == "prepare_WGS_data":
        from .prepare import prepare_WGS_data
        return prepare_WGS_data
    raise AttributeError(name)
