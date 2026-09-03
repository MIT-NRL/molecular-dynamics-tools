"""Numerical transforms used by molecular-dynamics analyses."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson
from scipy.signal import ZoomFFT

TransformBackend = Literal["zoomfft", "direct"]
IntegrationMethod = Literal["uniform", "simpson"]


def spherical_bessel_transform(
    coordinates: ArrayLike,
    values: ArrayLike,
    output_grid: ArrayLike,
    *,
    backend: TransformBackend = "zoomfft",
    integration: IntegrationMethod = "uniform",
    chunk_size: int = 512,
) -> NDArray[np.float64]:
    """Evaluate ``integral x^2 f(x) sinc(x y) dx`` for one or more curves.

    The FFT backend evaluates the same trapezoidal sum as the direct backend
    when both grids are uniform. A two-dimensional ``values`` array is treated
    as ``(coordinates, curves)`` and all curves are transformed together.
    """

    source = np.asarray(coordinates, dtype=np.float64)
    targets = np.asarray(output_grid, dtype=np.float64)
    curves = np.asarray(values, dtype=np.float64)
    one_dimensional = curves.ndim == 1
    if one_dimensional:
        curves = curves[:, np.newaxis]
    if source.ndim != 1 or targets.ndim != 1 or curves.ndim != 2:
        raise ValueError("coordinates and output_grid must be one-dimensional")
    if len(source) < 2 or len(targets) < 2:
        raise ValueError("transform grids must contain at least two points")
    if curves.shape[0] != len(source):
        raise ValueError("values must have one row per source coordinate")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(targets)):
        raise ValueError("transform grids must contain only finite values")
    if not np.all(np.isfinite(curves)):
        raise ValueError("transform values must contain only finite values")
    if np.any(np.diff(source) <= 0) or np.any(np.diff(targets) <= 0):
        raise ValueError("transform grids must be strictly increasing")
    if backend not in {"zoomfft", "direct"}:
        raise ValueError("backend must be 'zoomfft' or 'direct'")
    if integration not in {"uniform", "simpson"}:
        raise ValueError("integration must be 'uniform' or 'simpson'")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    source_steps = np.diff(source)
    target_steps = np.diff(targets)
    source_uniform = np.allclose(source_steps, source_steps[0], rtol=1e-8)
    target_uniform = np.allclose(target_steps, target_steps[0], rtol=1e-8)
    if integration == "uniform" and not source_uniform:
        raise ValueError("uniform integration requires a uniformly spaced source grid")

    use_zoomfft = backend == "zoomfft" and integration == "uniform" and target_uniform
    if use_zoomfft:
        step = float(source_steps[0])
        weights = np.full(len(source), step, dtype=np.float64)
        weights[[0, -1]] *= 0.5
        source_values = curves * source[:, np.newaxis] * weights[:, np.newaxis]
        source_values = source_values * np.exp(
            1j * targets[0] * source
        )[:, np.newaxis]
        frequency_end = (
            target_steps[0] * step * (len(targets) - 1) / (2.0 * np.pi)
        )
        zoom = ZoomFFT(
            len(source), [0.0, frequency_end], len(targets), fs=1, endpoint=True
        )
        complex_sum = np.conj(
            zoom(np.conj(source_values.T), axis=-1)
        ).T
        phase = np.exp(
            1j * target_steps[0] * np.arange(len(targets)) * source[0]
        )
        transformed = np.empty((len(targets), curves.shape[1]), dtype=np.float64)
        at_zero = np.isclose(targets, 0.0, rtol=0.0, atol=1e-14)
        transformed[at_zero] = np.sum(
            curves * source[:, np.newaxis] ** 2 * weights[:, np.newaxis], axis=0
        )
        transformed[~at_zero] = (
            np.imag(phase[~at_zero, np.newaxis] * complex_sum[~at_zero])
            / targets[~at_zero, np.newaxis]
        )
    else:
        transformed = np.empty((len(targets), curves.shape[1]), dtype=np.float64)
        integrand = curves * source[:, np.newaxis] ** 2
        for start in range(0, len(targets), chunk_size):
            stop = min(start + chunk_size, len(targets))
            kernel = np.sinc(np.outer(targets[start:stop], source) / np.pi)
            expanded = kernel[:, :, np.newaxis] * integrand[np.newaxis, :, :]
            if integration == "simpson":
                transformed[start:stop] = simpson(expanded, x=source, axis=1)
            else:
                transformed[start:stop] = np.trapezoid(expanded, x=source, axis=1)

    return transformed[:, 0] if one_dimensional else transformed


__all__ = ["IntegrationMethod", "TransformBackend", "spherical_bessel_transform"]
