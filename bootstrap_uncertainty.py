"""Configurable bootstrap uncertainty for glacier temperature reconstructions.

The defaults are deliberately placeholders. Replace them with the measurement
variances and fitted parameter uncertainties from the paper before comparing
the resulting interval with published values.
"""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class BootstrapConfig:
    """Configure resampling, measurement errors, model uncertainty, and smoothing."""

    samples: int = 100
    perturbations: int = 10
    seed: int | None = 0
    c1: float = 0.00204
    c2: float = 19.0
    measurement_std_m: Mapping[int, float] = field(
        default_factory=lambda: {1: 50.0, 2: 200.0, 3: 300.0, 4: 500.0, 5: 500.0}
    )
    gamma_log_std: float = 0.10
    tau_log_std: float = 0.10
    ar_coefficient: float = 0.7
    ar_innovation_std_m: float = 25.0
    smoothing_years: int = 20

    def __post_init__(self) -> None:
        if self.samples < 1 or self.perturbations < 1:
            raise ValueError("samples and perturbations must be positive")
        if not 0 <= self.ar_coefficient < 1:
            raise ValueError("ar_coefficient must be in [0, 1)")
        if self.smoothing_years < 1:
            raise ValueError("smoothing_years must be positive")


def _distance_weights(props: pd.DataFrame, glacier_ids: np.ndarray) -> np.ndarray:
    """Return normalized spatial weights from the glaciers' pairwise distances."""
    if len(glacier_ids) == 1:
        return np.ones(1)
    coordinates = props.loc[glacier_ids, ["lat", "lon"]].to_numpy(dtype=float)
    # Convert degrees to radians before applying the spherical-distance formula.
    lat = np.radians(coordinates[:, 0])
    lon = np.radians(coordinates[:, 1])
    # ``[:, None]`` turns each coordinate vector into a column so that
    # broadcasting creates every glacier-to-glacier pair.
    d_lat = lat[:, None] - lat[None, :]
    d_lon = (lon[:, None] - lon[None, :] + np.pi) % (2 * np.pi) - np.pi
    haversine = np.sin(d_lat / 2) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(d_lon / 2) ** 2
    )
    distance_squared = (2 * 6371.0 * np.arcsin(np.sqrt(haversine))) ** 2
    weights = distance_squared.sum(axis=1)
    return weights / weights.sum() if weights.sum() else np.full(len(glacier_ids), 1 / len(glacier_ids))


def _ar_noise(rng: np.random.Generator, length: int, coefficient: float, innovation_std: float) -> np.ndarray:
    """Generate correlated noise for years without direct observations."""
    noise = np.zeros(length, dtype=float)
    innovations = rng.normal(0.0, innovation_std, length)
    for index in range(1, length):
        noise[index] = coefficient * noise[index - 1] + innovations[index]
    return noise


def reconstruct_glacier(
    glacier: pd.Series,
    record: pd.DataFrame,
    rng: np.random.Generator,
    config: BootstrapConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Perturb one glacier record and return yearly temperature anomalies.

    The reconstruction treats measurement error, missing-year interpolation,
    and uncertain response parameters as separate sources of uncertainty.
    """

    record = record.sort_values("year")
    if record["year"].duplicated().any():
        return None

    precipitation = float(glacier["precip_m_per_a"])
    slope = float(glacier["slope"])
    reference_length = float(glacier["L1950_km"]) * 1000.0

    # Reject missing or non-physical metadata before using it in square roots.
    if not np.isfinite(precipitation) or not np.isfinite(slope) or precipitation <= 0 or slope <= 0:
        return None

    years = record["year"].to_numpy(dtype=int)
    observed = record["dL_m"].to_numpy(dtype=float).copy()
    source = record["source"].to_numpy(dtype=int)

    # Historical sources have different measurement reliability.
    std = np.array([config.measurement_std_m.get(method, 0.0) for method in source])
    observed += rng.normal(0.0, std)

    # Interpolate the perturbed observations onto a complete annual grid.
    model_years = np.arange(years.min(), years.max() + 1)
    interpolated = CubicSpline(years, observed)(model_years)
    observed_mask = np.isin(model_years, years)
    gap_mask = ~observed_mask

    # Add temporally correlated uncertainty only where observations are absent.
    if gap_mask.any():
        interpolated[gap_mask] += _ar_noise(
            rng, len(model_years), config.ar_coefficient, config.ar_innovation_std_m
        )[gap_mask]

    # Return all matching positions; here there should be at most one because
    # model_years is an annual, strictly increasing sequence.
    reference_index = np.flatnonzero(model_years == 1950)
    if not len(reference_index):
        return None

    # Convert length change to temperature using the glacier response model.
    length_change = interpolated - interpolated[reference_index[0]] +reference_length
    beta = 0.006 * np.sqrt(precipitation)
    gamma = config.c1 * slope / np.sqrt(precipitation)
    tau = config.c2 / (beta * slope * np.sqrt(1 + 20 * slope) * np.sqrt(reference_length))

    # Do a log transform of the parameters to ensure they remain positive after perturbation.
    gamma *= np.exp(rng.normal(0.0, config.gamma_log_std))
    tau *= np.exp(rng.normal(0.0, config.tau_log_std))

    dL_dT = np.gradient(length_change, model_years)
    temperature = -(length_change + tau * dL_dT) / (1 / gamma)
    baseline = (model_years >= 1961) & (model_years <= 1990)
    if not baseline.any():
        return None
    return model_years, temperature - np.mean(temperature[baseline])


def _regional_mean(
    props: pd.DataFrame,
    reconstructions: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    distance_weighted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine glacier reconstructions into a yearly regional mean."""
    # The outer key is a year; each inner dictionary would map glacier IDs to
    # values. The explicit type makes this nested mapping structure readable.
    by_year: dict[int, dict[int, float]] = {}
    # Store lists because several glaciers can contribute to the same year.
    by_year_lists: dict[int, list[tuple[int, float]]] = {}
    for glacier_id, (years, values) in reconstructions:
        for year, value in zip(years, values):
            if np.isfinite(value):
                # setdefault creates an empty list for a new year, then returns
                # the existing list for later glaciers in that same year.
                by_year_lists.setdefault(int(year), []).append((glacier_id, float(value)))

    years = np.array(sorted(by_year_lists), dtype=int)
    mean = np.full(len(years), 0.0)
    for index, year in enumerate(years):
        values = by_year_lists[year]
        ids = np.array([val[0] for val in values], dtype=int)
        observations = np.array([val[1] for val in values], dtype=float)
        mean[index] = np.dot(_distance_weights(props, ids), observations) if distance_weighted else observations.mean()
    return years, mean


def bootstrap_temperature_ensemble(
    props: pd.DataFrame,
    records: pd.DataFrame,
    config: BootstrapConfig | None = None,
    region: int | None = None,
    distance_weighted: bool = True,
) -> pd.DataFrame:
    """Generate an ensemble and return yearly mean, spread, and 2-sigma bounds.

    Bootstrap resampling captures sampling uncertainty across glaciers, while
    repeated perturbations capture measurement and reconstruction uncertainty.
    """

    config = config or BootstrapConfig()
    rng = np.random.default_rng(config.seed)
    available = props.index.to_numpy(dtype=int)
    if region is not None:
        available = props.loc[props["region"] == region].index.to_numpy(dtype=int)
    if not len(available):
        raise ValueError("no glaciers available for the requested region")

    regional_series: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(config.samples):
        sampled_ids = rng.choice(available, size=len(available), replace=True)
        for _ in range(config.perturbations):
            reconstructions = []
            for glacier_id in sampled_ids:
                record = records[records["ID"] == glacier_id]
                result = reconstruct_glacier(props.loc[glacier_id], record, rng, config)
                if result is not None:
                    reconstructions.append((int(glacier_id), result))
            if reconstructions:
                regional_series.append(_regional_mean(props, reconstructions, distance_weighted))
    if not regional_series:
        raise ValueError("all bootstrap reconstructions were invalid")

    # Align each irregular reconstruction on one common annual time axis.
    years = np.arange(
        min(series[0].min() for series in regional_series),
        max(series[0].max() for series in regional_series) + 1,
    )
    ensemble = np.full((len(regional_series), len(years)), 0.0)
    for row, (series_years, values) in enumerate(regional_series):
        # Since ``years`` is sorted, searchsorted converts calendar years to
        # column positions in the common ensemble array.
        ensemble[row, np.searchsorted(years, series_years)] = values
    mean = np.nanmean(ensemble, axis=0)
    spread = np.nanstd(ensemble, axis=0, ddof=1)

    # Smooth the uncertainty estimate, not the reconstructed mean.
    smoothed = pd.Series(spread, index=years).rolling(config.smoothing_years, center=True, min_periods=1).mean().to_numpy()
    return pd.DataFrame({
        "year": years,
        "mean": mean,
        "std": smoothed,
        "lower": mean - 2 * smoothed,
        "upper": mean + 2 * smoothed,
    })
