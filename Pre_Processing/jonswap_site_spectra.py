"""Construct, not fit, combined ERA5 Hs/Tp-matched JONSWAP spectra."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from era5_cache import load_cache


BASE_DIR = Path(__file__).resolve().parent
BIN_WIDTH = 0.25
SENSITIVITY_GAMMAS = (1.0, 3.3, 5.0)
GRID_TOLERANCE = 1e-5


def iec_gamma(hs, tp):
    """Apply the RAFT IEC rule to positive heights; calm gamma is undefined."""
    height, period = np.broadcast_arrays(
        np.asarray(hs, dtype=float), np.asarray(tp, dtype=float)
    )
    if np.any(~np.isfinite(height) | (height < 0)):
        raise ValueError("Hs must be finite and nonnegative")
    if np.any(~np.isfinite(period) | (period <= 0)):
        raise ValueError(
            "Tp must be finite and positive, including calm states")
    ratio = np.full(height.shape, np.nan)
    np.divide(period, np.sqrt(height), out=ratio, where=height > 0)
    gamma = np.exp(5.75 - 1.15 * ratio)
    return np.where(ratio <= 3.6, 5.0, np.where(ratio >= 5, 1.0, gamma))


def raft_density(frequency_hz, hs, tp, gamma):
    """Return the unmodified helper output, evaluated at omega=2*pi*f."""
    root = str(BASE_DIR.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from raft.helpers import JONSWAP

    return JONSWAP(2 * np.pi * frequency_hz, hs, tp, gamma)


def construct_spectrum(hs, tp, gamma):
    """Normalize an adaptive positive Hz grid to variance Hs^2/16."""
    iec_gamma(hs, tp)
    if not np.isfinite(gamma) or not 1 <= gamma <= 10:
        raise ValueError("Gamma must be finite and in [1, 10]")
    lower, upper, points = 0.05, 32.0, 4097
    target = hs * hs / 16
    for _ in range(10):
        frequency = np.geomspace(lower, upper, points) / tp
        raw = raft_density(
            frequency,
            hs,
            tp,
            gamma) if hs else np.zeros(points)
        variance = float(np.trapezoid(raw, frequency))
        fine_frequency = np.geomspace(lower, upper, 2 * points - 1) / tp
        fine = (raft_density(fine_frequency, hs, tp, gamma)
                if hs else np.zeros(fine_frequency.size))
        fine_variance = float(np.trapezoid(fine, fine_frequency))
        relative_error = abs(variance - fine_variance) / \
            variance if hs else 0.0
        coefficient = (1 - 0.287 * np.log(gamma)) * \
            0.3125 * hs**2 / (2 * np.pi)
        tail_bound = coefficient * gamma / (4 * upper**4)
        low_bound = coefficient * gamma / 5 * np.exp(-1.25 / lower**4)
        omitted_bound = (tail_bound + low_bound) / variance if hs else 0.0
        peak_step = float(np.max(np.diff(frequency * tp)[
            (frequency[:-1] * tp >= 0.8) & (frequency[:-1] * tp <= 1.2)
        ]))
        if (omitted_bound <= GRID_TOLERANCE
            and relative_error <= GRID_TOLERANCE
                and peak_step <= 0.002):
            break
        if omitted_bound > GRID_TOLERANCE:
            upper *= 2
        points = 2 * points - 1
    else:
        raise RuntimeError("Frequency grid did not converge")
    factor = target / variance if hs else 1.0
    density = raw * factor
    normalized_variance = float(np.trapezoid(density, frequency))
    checks = {
        "target_variance_m2": target,
        "raw_helper_integral_over_hz_m2": variance,
        "normalization_factor": factor,
        "normalized_variance_m2": normalized_variance,
        "reconstructed_hs_m": 4 * np.sqrt(normalized_variance),
        "omitted_raw_energy_relative_bound": omitted_bound,
        "grid_refinement_relative_error": relative_error,
        "peak_step_over_fp": peak_step,
        "frequency_min_hz": float(frequency[0]),
        "frequency_max_hz": float(frequency[-1]),
        "frequency_points": points,
    }
    return frequency, density, checks


def summarize_hours(hs, tp):
    height, period = np.asarray(hs, dtype=float), np.asarray(tp, dtype=float)
    if height.ndim != 1 or period.shape != height.shape:
        raise ValueError(
            "Hs and Tp must be matching one-dimensional hourly arrays")
    valid_height = np.isfinite(height) & (height >= 0)
    valid_period = np.isfinite(period) & (period > 0)
    valid = valid_height & valid_period
    counts = {
        "total_hours": int(height.size),
        "finite_nonnegative_swh_hours": int(valid_height.sum()),
        "finite_positive_pp1d_hours": int(valid_period.sum()),
        "valid_pair_hours": int(valid.sum()),
        "excluded_invalid_pair_hours": int((~valid).sum()),
        "invalid_height_hours": int((~valid_height).sum()),
        "invalid_period_hours": int((~valid_period).sum()),
        "both_invalid_hours": int((~valid_height & ~valid_period).sum()),
        "zero_hs_valid_pair_hours": int((valid & (height == 0)).sum()),
    }
    if not valid.any():
        raise ValueError("No valid combined swh/pp1d hourly pairs")
    return height[valid], period[valid], counts


def select_states(height, period):
    bins = np.column_stack(
        (np.floor(
            height /
            BIN_WIDTH),
            np.floor(
            period /
            BIN_WIDTH))).astype(int)
    occupied, counts = np.unique(bins, axis=0, return_counts=True)
    centers = (occupied + 0.5) * BIN_WIDTH
    selectors = [("most_probable", None)] + [
        (f"hs_p{int(quantile * 100)}", float(np.quantile(height, quantile)))
        for quantile in (0.5, 0.9, 0.99)
    ]
    states = []
    for name, target in selectors:
        distance = np.zeros(counts.size) if target is None else abs(
            centers[:, 0] - target)
        index = np.lexsort(
            (occupied[:, 1], occupied[:, 0], -counts, distance))[0]
        hs, tp = centers[index]
        states.append(
            {"state_id": name, "quantile_target_hs_m": target,
             "hs_bin_lower_m": float(occupied[index, 0] * BIN_WIDTH),
             "hs_bin_upper_m": float((occupied[index, 0] + 1) * BIN_WIDTH),
             "tp_bin_lower_s": float(occupied[index, 1] * BIN_WIDTH),
             "tp_bin_upper_s": float((occupied[index, 1] + 1) * BIN_WIDTH),
             "hs_m": float(hs),
             "tp_s": float(tp),
             "count": int(counts[index]),
             "probability": float(counts[index] / height.size),
             "valid_pair_hours": int(height.size),
             "iec_gamma": float(iec_gamma(hs, tp)), })
    return states, int(counts.size)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_constructed(hs, tp, results_dir, source_metadata):
    height, period, counts = summarize_hours(hs, tp)
    states, occupied_count = select_states(height, period)
    gammas = iec_gamma(height, period)
    positive = gammas[np.isfinite(gammas)]
    edges = np.linspace(1, max(5.0, float(np.exp(5.75 - 1.15 * 3.6))), 42)
    histogram, edges = np.histogram(positive, bins=edges)
    distribution = [
        {"gamma_lower": float(lower),
         "gamma_upper": float(upper),
         "count": int(count),
         "probability_of_positive_hs_hours": float(count / positive.size)
         if positive.size else 0.0, "positive_hs_hours": int(positive.size),
         "valid_pair_hours": int(height.size)} for lower, upper,
        count in zip(edges[: -1],
                     edges[1:],
                     histogram)]
    spectra, checks = [], []
    for state in states:
        for model, gamma in [("iec", state["iec_gamma"])] + [
            (f"assumed_{gamma:g}", gamma) for gamma in SENSITIVITY_GAMMAS
        ]:
            frequency, density, check = construct_spectrum(
                state["hs_m"], state["tp_s"], gamma)
            checks.append(
                {"state_id": state["state_id"],
                 "model": model, "gamma": gamma, **check})
            spectra.extend({"state_id": state["state_id"], "model": model,
                            "gamma": gamma, "frequency_hz": float(freq),
                            "psd_m2_per_hz": float(value)}
                           for freq, value in zip(frequency, density))
    metadata = {
        **source_metadata,
        "counts": counts,
        "occupied_combined_bins": occupied_count,
        "interpretation": (
            "Constructed combined swh/pp1d-matched spectra; "
            "no measured spectra or fitted site gamma."
        ),
        "bin_widths": {"hs_m": BIN_WIDTH, "tp_s": BIN_WIDTH},
        "selection": (
            "Half-open bins from zero; bin centers used. Mode: largest "
            "count, then Hs/Tp bin order. Quantiles: nearest Hs center, "
            "largest count, then Hs/Tp order. Duplicate bins across "
            "selectors retained."
        ),
        "gamma_provenance": (
            "raft/helpers.py:JONSWAP, IEC 61400-3 / FAST v7 rule; "
            "r=Tp/sqrt(Hs); r<=3.6:5; r>=5:1; "
            "otherwise exp(5.75-1.15*r)."
        ),
        "hourly_iec_gamma": {
            "hours": int(positive.size),
            "mean": float(positive.mean()) if positive.size else None,
            "median": float(np.median(positive)) if positive.size else None,
            "gamma_1_hours": int((positive == 1).sum()),
            "gamma_5_hours": int((positive == 5).sum()),
            "interior_rule_hours": int(
                ((positive != 1) & (positive != 5)).sum()
            ),
            "weighting": (
                "One weight per valid positive-Hs hour, using actual "
                "hourly Hs/Tp; not bin centers or equal bin weights."
            ),
        },
        "units": {"frequency": "Hz", "psd": "m^2/Hz", "variance": "m^2"},
        "normalization": (
            "Pass omega=2*pi*f into unmodified RAFT helper. Its documented "
            "per-Hz output includes a 1/(2*pi) factor; numerical "
            "normalization, not another PSD conversion, enforces "
            "integral S(f)df=Hs^2/16."
        ),
        "grid_tolerance": GRID_TOLERANCE, "spectrum_checks": checks,
        "caveats": [
            ("59 N, 4 W is a regional proxy about 42 km from the nominal "
             "farm, not a farm measurement."),
            ("Combined pp1d is not a wind-wave partition Tp. Unimodal "
             "JONSWAP may not represent mixed wind/swell; no average "
             "spectrum is asserted to be a realizable sea."),
            ("No wind or workability filters applied; denominator is all "
             "valid combined swh/pp1d pairs."),
            ("Zero-Hs hours have zero physical spectrum and undefined "
             "gamma; excluded only from gamma summary. Representative "
             "occupied-bin curves use bin centers even for a bin "
             "containing calm hours."),
            ("IEC rule has small boundary discontinuities; its interior "
             "branch can slightly exceed 5 near r=3.6. Histogram "
             "includes these values."),
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    write_rows(results_dir / "jonswap_constructed_states.csv", states)
    write_rows(results_dir / "jonswap_constructed_spectra.csv", spectra)
    write_rows(
        results_dir /
        "jonswap_iec_gamma_distribution.csv",
        distribution)
    (results_dir / "jonswap_constructed_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=BASE_DIR /
        "era5_processed/era5_point_2006_2025.parquet")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASE_DIR /
        "wave_results_2006_2025")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(
            str(BASE_DIR), pattern="test_jonswap_site_spectra.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    if args.start_year > args.end_year:
        parser.error("start-year must not exceed end-year")
    times, metrics, latitude, longitude, units = load_cache(
        args.cache_file, args.start_year, args.end_year)
    source_metadata = {
        "cache_file": str(args.cache_file.resolve()),
        "period": {"start_year": args.start_year, "end_year": args.end_year,
                   "first_hour": str(times[0]), "last_hour": str(times[-1])},
        "location": {"latitude": latitude, "longitude": longitude},
        "source_variables": {"height": "swh", "peak_period": "pp1d"},
        "source_units": {name: units[name] for name in ("swh", "pp1d")},
    }
    metadata = export_constructed(
        metrics["swh"], metrics["pp1d"], args.results_dir, source_metadata
    )
    print(
        json.dumps(
            {"results_dir": str(args.results_dir.resolve()),
             "counts": metadata["counts"],
             "hourly_iec_gamma": metadata["hourly_iec_gamma"]},
            indent=2))


if __name__ == "__main__":
    main()
