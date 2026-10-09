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
    """Controls the ensemble size and all perturbation assumptions."""

    samples: int = 100
    perturbations: int = 10
    seed: int | None = 0
    c1: float = 0.00204
    c2: float = 19.0
    measurement_std_m: Mapping[int, float] = field(
        default_factory=lambda: {1: 25.0, 2: 50.0, 3: 100.0, 4: 150.0, 5: 200.0}
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
    if len(glacier_ids) == 1:
        return np.ones(1)
    coordinates = props.loc[glacier_ids, ["lat", "lon"]].to_numpy(dtype=float)
    lat = np.radians(coordinates[:, 0])
    lon = np.radians(coordinates[:, 1])
    d_lat = lat[:, None] - lat[None, :]
    d_lon = (lon[:, None] - lon[None, :] + np.pi) % (2 * np.pi) - np.pi
    haversine = np.sin(d_lat / 2) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(d_lon / 2) ** 2
    )
    distance_squared = (2 * 6371.0 * np.arcsin(np.sqrt(haversine))) ** 2
    weights = distance_squared.sum(axis=1)
    return weights / weights.sum() if weights.sum() else np.full(len(glacier_ids), 1 / len(glacier_ids))


def _ar_noise(rng: np.random.Generator, length: int, coefficient: float, innovation_std: float) -> np.ndarray:
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
    """Perturb one record and return ``(years, temperature_anomaly)``."""

    record = record.sort_values("year")
    if len(record) < 3 or record["year"].duplicated().any():
        return None

    #Make sure the the type is float
    precipitation = float(glacier["precip_m_per_a"])
    slope = float(glacier["slope"])
    reference_length = float(glacier["L1950_km"]) * 1000.0

    if not np.isfinite(precipitation) or not np.isfinite(slope) or precipitation <= 0 or slope <= 0:
        return None
    #again making sure type is correct
    years = record["year"].to_numpy(dtype=int)
    observed = record["dL_m"].to_numpy(dtype=float).copy()
    source = record["source"].to_numpy(dtype=int)
    # Create an array of measurement standard deviations for each observation based on its source, some are less reliable than others(paintings etc.)
    std = np.array([config.measurement_std_m.get(method, 0.0) for method in source])
    #add noise to the length record
    observed += rng.normal(0.0, std)
    #make the time array with interpolation
    model_years = np.arange(years.min(), years.max() + 1)
    interpolated = CubicSpline(years, observed)(model_years)
    #makes a list of the years where we have observed data
    observed_mask = np.isin(model_years, years)
    gap_mask = ~observed_mask
    #fill in the gaps with autoregressive noise
    if gap_mask.any():
        interpolated[gap_mask] += _ar_noise(
            rng, len(model_years), config.ar_coefficient, config.ar_innovation_std_m
        )[gap_mask]

    reference_index = np.flatnonzero(model_years == 1950)
    if not len(reference_index):
        return None
    length_change = interpolated - interpolated[reference_index[0]]
    beta = 0.006 * np.sqrt(precipitation)
    gamma = config.c1 * slope / np.sqrt(precipitation)
    tau = config.c2 / (beta * slope * np.sqrt(1 + 20 * slope) * np.sqrt(reference_length))
    gamma *= np.exp(rng.normal(0.0, config.gamma_log_std))
    tau *= np.exp(rng.normal(0.0, config.tau_log_std))
    derivative = np.gradient(length_change, model_years)
    #vectorised temperature reconstruction
    temperature = -(length_change + tau * derivative) / (1 / gamma)
    baseline = (model_years >= 1945) & (model_years <= 1965)
    if not baseline.any():
        return None
    return model_years, temperature - np.mean(temperature[baseline])


def _regional_mean(
    props: pd.DataFrame,
    reconstructions: list[tuple[int, tuple[np.ndarray, np.ndarray]]],
    distance_weighted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    by_year: dict[int, dict[int, float]] = {}
    by_year_lists: dict[int, list[tuple[int, float]]] = {}
    for glacier_id, (years, values) in reconstructions:
        for year, value in zip(years, values):
            if np.isfinite(value):
                by_year_lists.setdefault(int(year), []).append((glacier_id, float(value)))
    years = np.array(sorted(by_year_lists), dtype=int)
    mean = np.full(len(years), np.nan)
    for index, year in enumerate(years):
        values = by_year_lists[year]
        ids = np.array([item[0] for item in values], dtype=int)
        observations = np.array([item[1] for item in values], dtype=float)
        mean[index] = np.dot(_distance_weights(props, ids), observations) if distance_weighted else observations.mean()
    return years, mean


def bootstrap_temperature_ensemble(
    props: pd.DataFrame,
    records: pd.DataFrame,
    config: BootstrapConfig | None = None,
    region: int | None = None,
    distance_weighted: bool = True,
) -> pd.DataFrame:
    """Generate the ensemble and return yearly mean, spread, and 2-sigma bounds."""

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

    years = np.arange(
        min(series[0].min() for series in regional_series),
        max(series[0].max() for series in regional_series) + 1,
    )
    ensemble = np.full((len(regional_series), len(years)), np.nan)
    for row, (series_years, values) in enumerate(regional_series):
        ensemble[row, np.searchsorted(years, series_years)] = values
    mean = np.nanmean(ensemble, axis=0)
    spread = np.nanstd(ensemble, axis=0, ddof=1)
    smoothed = pd.Series(spread, index=years).rolling(config.smoothing_years, center=True, min_periods=1).mean().to_numpy()
    return pd.DataFrame({
        "year": years,
        "mean": mean,
        "std": smoothed,
        "lower": mean - 2 * smoothed,
        "upper": mean + 2 * smoothed,
    })
