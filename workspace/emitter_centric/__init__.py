"""Emitter-Centric Analysis workspace sub-mode (isolated research sandbox)."""

__all__ = ["run_straight_cv_job"]


def __getattr__(name: str):
    if name == "run_straight_cv_job":
        from workspace.emitter_centric.synthesis import run_straight_cv_job

        return run_straight_cv_job
    raise AttributeError(name)
