"""Model-reduction algorithms."""

_EXPORTS = {
    "DMDReduction": ("autoreduce.reductions.projection.dmd", "DMDReduction"),
    "Reduce": ("autoreduce.reductions.timescale", "Reduce"),
    "ReduceUtils": ("autoreduce.reductions.timescale", "ReduceUtils"),
    "apply_conservation_laws": (
        "autoreduce.reductions.conservation",
        "apply_conservation_laws",
    ),
    "create_system": ("autoreduce.reductions.timescale", "create_system"),
    "find_conserved_sets": (
        "autoreduce.reductions.conservation",
        "find_conserved_sets",
    ),
    "fit_dmd": ("autoreduce.reductions.projection.dmd", "fit_dmd"),
    "fit_dmdc": ("autoreduce.reductions.projection.dmd", "fit_dmdc"),
    "get_conservation_laws": (
        "autoreduce.reductions.conservation",
        "get_conservation_laws",
    ),
    "setup_conservation_laws": (
        "autoreduce.reductions.conservation",
        "setup_conservation_laws",
    ),
    "solve_approximations": (
        "autoreduce.reductions.abundance",
        "solve_approximations",
    ),
    "solve_conservation_laws": (
        "autoreduce.reductions.conservation",
        "solve_conservation_laws",
    ),
    "solve_timescale_separation": (
        "autoreduce.reductions.timescale",
        "solve_timescale_separation",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    """Lazily load reduction exports and optional integrations."""
    if name not in _EXPORTS:
        raise AttributeError(
            f"module 'autoreduce.reductions' has no attribute {name!r}"
        )
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
