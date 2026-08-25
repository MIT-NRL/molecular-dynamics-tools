"""Plot normalized scattering results without mixing curve baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .calculation import ProbeScatteringResult


def _axis(ax):
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "plotting requires matplotlib; install molecular-dynamics-tools[plot]"
        ) from error
    return plt.subplots()[1]


def _table(result: ProbeScatteringResult | pd.DataFrame, attribute: str) -> pd.DataFrame:
    if isinstance(result, ProbeScatteringResult):
        return getattr(result, attribute)
    if isinstance(result, pd.DataFrame):
        return result
    raise TypeError("result must be a ProbeScatteringResult or pandas DataFrame")


def _styles(
    total_kwargs: Mapping[str, Any] | None,
    partial_kwargs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total = {"color": "black", "linewidth": 2.0, "label": "Total"}
    partial = {"linestyle": "--", "linewidth": 1.0, "alpha": 0.7}
    if total_kwargs:
        total.update(total_kwargs)
    if partial_kwargs:
        partial.update(partial_kwargs)
    return total, partial


def plot_structure_factor(
    result: ProbeScatteringResult | pd.DataFrame,
    *,
    ax=None,
    show_partials: bool = True,
    total_kwargs: Mapping[str, Any] | None = None,
    partial_kwargs: Mapping[str, Any] | None = None,
):
    """Plot total and pair-resolved ``S(Q)`` on the same unit baseline.

    Stored pair columns are additive contributions to ``S(Q)-1``. For plotting,
    one is added to every pair curve so that both pair curves and the total use
    the conventional ``S(Q) -> 1`` baseline.
    """

    table = _table(result, "structure_factor")
    if "Q" not in table or "Total" not in table:
        raise ValueError("structure-factor table must contain 'Q' and 'Total'")
    axis = _axis(ax)
    total_style, partial_style = _styles(total_kwargs, partial_kwargs)
    q = table["Q"]
    if show_partials:
        for column in table.columns:
            if column in {"Q", "Total"}:
                continue
            axis.plot(q, 1.0 + table[column], label=column, **partial_style)
    axis.plot(q, table["Total"], **total_style)
    axis.axhline(1.0, color="0.75", linewidth=0.8)
    axis.set_xlabel(r"$Q$ ($\AA^{-1}$)")
    axis.set_ylabel(r"$S(Q)$")
    return axis


def plot_weighted_rdf(
    result: ProbeScatteringResult | pd.DataFrame,
    *,
    ax=None,
    show_partials: bool = True,
    total_kwargs: Mapping[str, Any] | None = None,
    partial_kwargs: Mapping[str, Any] | None = None,
):
    """Plot the dimensionless weighted ``g(r)`` and additive pair terms."""

    table = _table(result, "weighted_rdf")
    if "r" not in table or "Total" not in table:
        raise ValueError("weighted-RDF table must contain 'r' and 'Total'")
    if np.any(~np.isfinite(table.to_numpy(dtype=np.float64))):
        raise ValueError("weighted-RDF table must contain only finite values")
    axis = _axis(ax)
    total_style, partial_style = _styles(total_kwargs, partial_kwargs)
    r = table["r"]
    if show_partials:
        for column in table.columns:
            if column in {"r", "Total"}:
                continue
            axis.plot(r, table[column], label=column, **partial_style)
    axis.plot(r, table["Total"], **total_style)
    axis.axhline(1.0, color="0.75", linewidth=0.8)
    axis.set_xlabel(r"$r$ ($\AA$)")
    axis.set_ylabel(r"$g(r)$")
    return axis


__all__ = ["plot_structure_factor", "plot_weighted_rdf"]
