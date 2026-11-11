"""Export wave diagnostics from a cache and optional measured spectra."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from era5_cache import load_cache


def write_rows(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def empirical_distribution(values):
    finite = values[np.isfinite(values) & (values >= 0)]
    if not finite.size:
        return np.array([]), np.array([]), np.array([], dtype=int)
    heights, counts = np.unique(finite, return_counts=True)
    return heights, np.cumsum(counts) / counts.sum(), counts


def height_diagnostics(times, metrics):
    combined, wind, swell = (metrics[name] for name in ("swh", "shww", "shts"))
    paired = (
        np.isfinite(combined) & np.isfinite(wind)
        & (combined >= 0) & (wind >= 0)
    )
    partitioned = paired & np.isfinite(swell) & (swell >= 0)
    residual = combined[partitioned] ** 2 - wind[partitioned] ** 2
    residual -= swell[partitioned] ** 2
    coverage = {}
    for name in ("swh", "shww", "shts", "pp1d", "mwd", "mdww", "mdts", "wsp"):
        selected = np.isfinite(metrics[name])
        indices = np.flatnonzero(selected)
        coverage[name] = {
            "finite_hours": int(selected.sum()),
            "first": str(times[indices[0]]) if indices.size else None,
            "last": str(times[indices[-1]]) if indices.size else None,
        }
    return {
        "total_hours": int(times.size),
        "paired_height_hours": int(paired.sum()),
        "combined_below_wind_by_more_than_0_01m_hours": int(
            (paired & (combined < wind - 0.01)).sum()
        ),
        "three_partition_hours": int(partitioned.sum()),
        "energy_residual_m2_mean": (
            float(residual.mean()) if residual.size else None
        ),
        "energy_residual_m2_rmse": (
            float(np.sqrt(np.mean(residual ** 2))) if residual.size else None
        ),
        "swell_energy_exceeds_wind_hours": int(
            (swell[partitioned] ** 2 > wind[partitioned] ** 2).sum()
        ),
        "coverage": coverage,
        "interpretation": [
            "swh: combined Hs; shww: wind-wave Hs; shts: total-swell Hs.",
            "swh - shww is a height increment, not swell significant height.",
            "sqrt(max(swh^2 - shww^2, 0)) is an energy-based swell proxy.",
            "Clipped negative energy differences remain flagged in QA counts.",
            "Energy closure is a diagnostic, not proof of correctness.",
            "Monthly and variable coverage must be checked before comparison.",
            "Wave spectral peakedness wsp is not JONSWAP gamma.",
            "Bulk Hs and Tp alone do not identify spectral gamma.",
        ],
    }, paired, partitioned


def export_diagnostics(cache_file, results_dir, start_year, end_year):
    times, metrics, latitude, longitude, _ = load_cache(
        cache_file, start_year, end_year
    )
    summary, paired, partitioned = height_diagnostics(times, metrics)
    summary["cache_file"] = str(cache_file.resolve())
    summary["location"] = {"latitude": latitude, "longitude": longitude}
    summary["directional_probability"] = {
        "bin_widths": {"hs_m": 0.25, "tp_s": 0.25, "direction_deg": 30},
        "denominator": "Finite paired Hs/pp1d/wave-direction hours per source",
        "period_caveat": (
            "pp1d is the combined-sea peak period, not partition Tp"
        ),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    month_values = times.astype("datetime64[M]")
    write_rows(
        results_dir / "wave_variable_coverage.csv",
        ["month", "source", "finite_hours", "total_hours", "coverage"],
        (
            [str(month), source,
             int(np.isfinite(metrics[source][selected]).sum()),
             int(selected.sum()),
             float(np.isfinite(metrics[source][selected]).mean())]
            for month in np.unique(month_values)
            for selected in [month_values == month]
            for source in summary["coverage"]
        ),
    )
    combined, wind, swell = (metrics[name] for name in ("swh", "shww", "shts"))
    write_rows(
        results_dir / "wave_height_comparison.csv",
        ["valid_time", "swh_m", "shww_m", "shts_m", "swh_minus_shww_m",
         "energy_swell_proxy_m", "energy_residual_m2", "wind_speed_10m_ms"],
        (
            [str(times[index]), combined[index], wind[index],
             swell[index] if partitioned[index] else "",
             combined[index] - wind[index],
             np.sqrt(max(combined[index] ** 2 - wind[index] ** 2, 0)),
             (combined[index] ** 2 - wind[index] ** 2 - swell[index] ** 2)
             if partitioned[index] else "",
             np.hypot(metrics["u10"][index], metrics["v10"][index])]
            for index in np.flatnonzero(paired)
        ),
    )
    cdf_rows = []
    for source in ("swh", "shww", "shts"):
        for sample, selected in (
            ("available", np.isfinite(metrics[source])),
            ("common_three_heights", partitioned),
        ):
            heights, cdf, counts = empirical_distribution(
                metrics[source][selected]
            )
            cdf_rows.extend(
                [source, sample, height, probability, 1 - probability,
                 int(count), int(counts.sum())]
                for height, probability, count in zip(heights, cdf, counts)
            )
    write_rows(
        results_dir / "wave_height_cdf.csv",
        ["source", "sample", "hs_m", "cdf", "exceedance", "count", "hours"],
        cdf_rows,
    )
    direction_rows = []
    for height_source, direction_source in (("swh", "mwd"), ("shww", "mdww")):
        heights = metrics[height_source]
        periods = metrics["pp1d"]
        directions = metrics[direction_source]
        selected = (
            np.isfinite(heights) & np.isfinite(periods)
            & np.isfinite(directions) & (heights >= 0) & (periods >= 0)
        )
        if not selected.any():
            continue
        indices = np.column_stack([
            np.floor(heights[selected] / 0.25).astype(int),
            np.floor(periods[selected] / 0.25).astype(int),
            np.floor((directions[selected] % 360) / 30).astype(int),
        ])
        bins, counts = np.unique(indices, axis=0, return_counts=True)
        direction_rows.extend(
            [height_source, direction_source,
             (height_bin + 0.5) * 0.25, (period_bin + 0.5) * 0.25,
             (direction_bin + 0.5) * 30, int(count), int(selected.sum()),
             count / selected.sum()]
            for (height_bin, period_bin, direction_bin), count
            in zip(bins, counts)
        )
    write_rows(
        results_dir / "directional_wave_probability.csv",
        ["height_source", "direction_source", "hs_m", "tp_s", "direction_deg",
         "count", "paired_hours", "probability"], direction_rows,
    )
    (results_dir / "wave_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def fit_jonswap_gamma(frequency, density):
    frequency = np.asarray(frequency, dtype=float)
    density = np.asarray(density, dtype=float)
    order = np.argsort(frequency)
    frequency, density = frequency[order], density[order]
    if (frequency.size < 12 or not np.isfinite(frequency).all()
            or not np.isfinite(density).all() or np.any(frequency <= 0)
            or np.any(density < 0) or np.any(np.diff(frequency) <= 0)
            or density.max() <= 0):
        raise ValueError(
            "Spectrum needs >=12 unique positive Hz bins and finite PSD"
        )
    peak_index = int(np.argmax(density))
    if peak_index in (0, frequency.size - 1):
        raise ValueError(
            "Spectral peak must lie inside measured frequency range"
        )
    peak = frequency[peak_index]
    selected = (
        (frequency >= 0.5 * peak) & (frequency <= 3 * peak) & (density > 0)
    )
    if selected.sum() < 8:
        raise ValueError("Insufficient positive samples around spectral peak")
    fit_frequency = frequency[selected]
    observed = np.log(density[selected])
    best = None
    for peak_frequency in np.linspace(peak * 0.85, peak * 1.15, 101):
        sigma = np.where(fit_frequency <= peak_frequency, 0.07, 0.09)
        enhancement = np.exp(
            -0.5 * ((fit_frequency / peak_frequency - 1) / sigma) ** 2
        )
        baseline = -5 * np.log(fit_frequency)
        baseline -= 1.25 * (peak_frequency / fit_frequency) ** 4
        for gamma in np.linspace(1, 10, 181):
            shape = baseline + np.log(gamma) * enhancement
            log_amplitude = float(np.mean(observed - shape))
            error = float(np.mean((observed - shape - log_amplitude) ** 2))
            if best is None or error < best[0]:
                best = (error, gamma, peak_frequency, log_amplitude)
    error, gamma, peak_frequency, log_amplitude = best
    return {
        "gamma": float(gamma), "tp_s": float(1 / peak_frequency),
        "log_psd_rmse": float(np.sqrt(error)),
        "amplitude": float(np.exp(log_amplitude)),
        "at_gamma_bound": bool(gamma in (1, 10)),
    }


def export_spectral_fits(spectra_file, results_dir):
    with spectra_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"spectrum_id", "frequency_hz", "psd_m2_per_hz"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Spectral CSV requires " + ", ".join(sorted(required))
            )
        groups = {}
        for row in reader:
            groups.setdefault(row["spectrum_id"], []).append(
                (float(row["frequency_hz"]), float(row["psd_m2_per_hz"]))
            )
    fits = []
    curves = []
    for spectrum_id, samples in groups.items():
        frequency, density = np.asarray(samples).T
        fit = fit_jonswap_gamma(frequency, density)
        fits.append({"spectrum_id": spectrum_id, **fit})
        peak_frequency = 1 / fit["tp_s"]
        sigma = np.where(frequency <= peak_frequency, 0.07, 0.09)
        enhancement = np.exp(
            -0.5 * ((frequency / peak_frequency - 1) / sigma) ** 2
        )
        fitted = fit["amplitude"] * frequency ** -5
        fitted *= np.exp(-1.25 * (peak_frequency / frequency) ** 4)
        fitted *= fit["gamma"] ** enhancement
        order = np.argsort(frequency)
        curves.extend(
            [spectrum_id, frequency[index], density[index], fitted[index]]
            for index in order
        )
    if not fits:
        raise ValueError("No measured spectra found")
    write_rows(
        results_dir / "jonswap_fits.csv", list(fits[0]),
        [list(fit.values()) for fit in fits],
    )
    write_rows(
        results_dir / "jonswap_spectral_curves.csv",
        ["spectrum_id", "frequency_hz", "observed_m2_per_hz",
         "fitted_m2_per_hz"], curves,
    )
    (results_dir / "jonswap_summary.json").write_text(json.dumps({
        "spectra": len(fits),
        "arithmetic_mean_gamma": float(np.mean([
            fit["gamma"] for fit in fits
        ])),
        "median_gamma": float(np.median([fit["gamma"] for fit in fits])),
        "method": "Log-PSD least squares, free amplitude, gamma 1:0.05:10",
        "caveat": (
            "Inspect residuals, boundary fits and multimodal spectra "
            "before averaging."
        ),
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--spectra-file", type=Path)
    args = parser.parse_args()
    export_diagnostics(
        args.cache_file, args.results_dir, args.start_year, args.end_year
    )
    if args.spectra_file:
        export_spectral_fits(args.spectra_file, args.results_dir)


if __name__ == "__main__":
    main()
