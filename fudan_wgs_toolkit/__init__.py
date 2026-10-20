"""Fudan WGS Toolkit: preprocessing and GPU association analysis."""
__version__ = "0.11.0"

__all__ = ["single_subj_preprocessing", "prepare_WGS_data", "generate_simulated_data", "run_WGS_all_binanry", "run_WGS_continuous"]


def __getattr__(name):
    if name == "single_subj_preprocessing":
        from .single_preprocessing import single_subj_preprocessing
        return single_subj_preprocessing
    if name == "run_WGS_all_binanry":
        from .run import run_WGS_all_binanry
        return run_WGS_all_binanry
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
