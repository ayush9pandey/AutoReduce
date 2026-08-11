"""Projection-based reduced models using PyDMD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from pydmd import DMD, DMDc
except ImportError as exc:
    raise ImportError(
        "PyDMD is required for autoreduce.reductions.projection.dmd. "
        "Install it with `pip install autoreduce[dmd]`."
    ) from exc


@dataclass(frozen=True)
class DMDReduction:
    """Fitted PyDMD model and the data used to construct it."""

    model: object
    snapshots: np.ndarray
    control_inputs: Optional[np.ndarray] = None

    @property
    def reduced_dimension(self) -> int:
        """Return the number of DMD modes in the fitted model."""
        return int(self.model.modes.shape[1])


def fit_dmd(snapshots, *, svd_rank: int, **dmd_options) -> DMDReduction:
    """Fit a dynamic mode decomposition model to state snapshots.

    Parameters
    ----------
    snapshots
        State snapshot matrix accepted by `pydmd.DMD.fit`.
    svd_rank
        Rank passed directly to PyDMD.
    dmd_options
        Additional keyword arguments passed directly to `pydmd.DMD`.

    Returns
    -------
    DMDReduction
        Fitted PyDMD model and snapshot data.
    """
    snapshot_array = np.asarray(snapshots)
    model = DMD(svd_rank=svd_rank, **dmd_options)
    model.fit(snapshot_array)
    return DMDReduction(model=model, snapshots=snapshot_array)


def fit_dmdc(
    snapshots,
    control_inputs,
    *,
    svd_rank: int,
    **dmdc_options,
) -> DMDReduction:
    """Fit a dynamic mode decomposition with control model.

    Parameters
    ----------
    snapshots
        State snapshot matrix accepted by `pydmd.DMDc.fit`.
    control_inputs
        Input snapshot matrix accepted by `pydmd.DMDc.fit`.
    svd_rank
        Rank passed directly to PyDMD.
    dmdc_options
        Additional keyword arguments passed directly to `pydmd.DMDc`.

    Returns
    -------
    DMDReduction
        Fitted PyDMDc model, state snapshots, and control inputs.
    """
    snapshot_array = np.asarray(snapshots)
    control_array = np.asarray(control_inputs)
    model = DMDc(svd_rank=svd_rank, **dmdc_options)
    model.fit(snapshot_array, control_array)
    return DMDReduction(
        model=model,
        snapshots=snapshot_array,
        control_inputs=control_array,
    )
