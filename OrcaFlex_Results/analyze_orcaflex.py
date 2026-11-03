"""Analyze the parked 600 s OrcaFlex case and compare with a regular-wave RAFT run.

Run from any directory. Use --orcaflex-only to omit the RAFT calculation.
All rotations in exported analysis tables are degrees; PSDs are one-sided per Hz.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram, welch


ROOT = Path(__file__).resolve().parents[1]
DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
MOTION_COLUMNS = ("Dynamic x", "Dynamic y", "Dynamic z", "Dynamic Rx", "Dynamic Ry", "Dynamic Rz")
ACCEL_COLUMNS = ("x acceleration", "y acceleration", "z acceleration",
                 "x angular acceleration", "y angular acceleration", "z angular acceleration")
GLOBAL_COLUMNS = ("X", "Y", "Z", "GX acceleration", "GY acceleration", "GZ acceleration")


def read_rows(path):
    """Read OrcaFlex's UTF-8 or Windows-1252 CSV exports without losing metadata."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    return list(csv.reader(text.splitlines()))


def read_time_history(path):
    rows = read_rows(path)
    headers = [index for index, row in enumerate(rows) if row and row[0].strip() == "Time"]
    if len(headers) != 1:
        raise ValueError("Expected exactly one numeric Time header in the time export.")
    header_index = headers[0]
    names = [name.strip() for name in rows[header_index]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate time-history column names.")
    required = ("Time",) + MOTION_COLUMNS + ACCEL_COLUMNS + GLOBAL_COLUMNS
    missing = set(required).difference(names)
    if missing:
        raise ValueError(f"Missing time-history channels: {sorted(missing)}")
    data = np.asarray([[float(value) for value in row] for row in rows[header_index + 1:] if row], dtype=float)
    if data.ndim != 2 or data.shape[1] != len(names) or len(data) < 4 or not np.isfinite(data).all():
        raise ValueError("Invalid or nonfinite time-history data.")
    time = data[:, names.index("Time")]
    intervals = np.diff(time)
    if np.any(intervals <= 0) or not np.allclose(intervals, np.median(intervals), rtol=1e-6, atol=1e-9):
        raise ValueError("Time must be increasing and uniformly sampled; no automatic resampling is applied.")
    return time, {name: data[:, index] for index, name in enumerate(names)}


def read_raos(path):
    sections = {}
    section = None
    for row in read_rows(path):
        label = " ".join(row).lower()
        if "position raos for point" in label:
            section = "motion"
            sections[section] = []
        elif "acceleration raos for point" in label:
            section = "acceleration"
            sections[section] = []
        elif "raos for point" in label:
            section = None
        if section is None or len(row) < 14:
            continue
        try:
            period = float(row[1])
            values = np.asarray([float(value) for value in row[2:14]])
        except ValueError:
            continue
        if period > 0 and np.isfinite(values).all():
            sections[section].append((1 / period, values[::2] * np.exp(1j * np.deg2rad(values[1::2]))))
    output = {}
    for name in ("motion", "acceleration"):
        if not sections.get(name):
            raise ValueError(f"No valid {name} RAO section found.")
        ordered = sorted(sections[name], key=lambda item: item[0])
        frequency = np.asarray([item[0] for item in ordered])
        if np.any(np.diff(frequency) <= 0):
            raise ValueError(f"Duplicate {name} RAO frequencies.")
        output[name] = (frequency, np.stack([item[1] for item in ordered]))
    return output


def interpolate_rao(section, frequency):
    frequencies, amplitudes = section
    if not frequencies[0] <= frequency <= frequencies[-1]:
        raise ValueError("Requested frequency is outside the exported RAO range.")
    return np.asarray([np.interp(frequency, frequencies, amplitudes[:, index].real)
                       + 1j * np.interp(frequency, frequencies, amplitudes[:, index].imag)
                       for index in range(6)])


def harmonic_fit(time, values, frequency):
    relative_time = time - time[0]
    phase = 2 * np.pi * frequency * relative_time
    basis = np.column_stack((np.ones(len(time)), np.cos(phase), np.sin(phase)))
    coefficients = np.linalg.lstsq(basis, values, rcond=None)[0]
    amplitude = np.hypot(coefficients[1], coefficients[2])
    residual = values - basis @ coefficients
    return amplitude, np.sqrt(np.mean(residual ** 2, axis=0))


def spectra(time, values, segment_seconds):
    sample_rate = 1 / np.median(np.diff(time))
    frequency, density = periodogram(values, fs=sample_rate, window="hann", detrend="constant", axis=0)
    segment_samples = min(len(time), max(4, round(segment_seconds * sample_rate)))
    welch_frequency, welch_density = welch(values, fs=sample_rate, window="hann", nperseg=segment_samples,
                                         noverlap=segment_samples // 2, detrend="constant", axis=0)
    return frequency, density, welch_frequency, welch_density


def statistics(time, values, frequency, quantity, frame):
    amplitudes, residuals = harmonic_fit(time, values, frequency)
    rows = []
    for index, dof in enumerate(DOFS):
        signal = values[:, index]
        mean = float(signal.mean())
        std = float(signal.std(ddof=0))
        unit = ("m" if index < 3 else "deg") + ("/s^2" if quantity == "acceleration" else "")
        rows.append(dict(dof=dof, quantity=quantity, unit=unit, frame=frame,
                         mean=mean, std=std, total_rms=float(np.sqrt(np.mean(signal ** 2))),
                         observed_min=float(signal.min()), observed_max=float(signal.max()),
                         peak_to_peak=float(np.ptp(signal)), mean_minus_3std=mean - 3 * std,
                         mean_plus_3std=mean + 3 * std, wave_amplitude=float(amplitudes[index]),
                         wave_rms=float(amplitudes[index] / np.sqrt(2)), residual_rms=float(residuals[index])))
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_spectrum(path, frequency, density, quantity):
    columns = [f"{dof}_{'m' if index < 3 else 'deg'}{'_s2' if quantity == 'acceleration' else ''}_squared_per_Hz"
               for index, dof in enumerate(DOFS)]
    np.savetxt(path, np.column_stack((frequency, density)), delimiter=",", header=",".join(["frequency_Hz"] + columns), comments="")


def plot_six(path, horizontal, values, quantity, title, frequency=None, reference=None):
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(horizontal, values[:, index], label="OrcaFlex", linewidth=0.9)
        if reference is not None:
            axis.plot(horizontal, reference[:, index], label="RAFT harmonic, arbitrary phase", linewidth=0.9)
        unit = "m" if index < 3 else "deg"
        if quantity == "acceleration":
            unit += "/s^2"
        if frequency is not None:
            axis.axvline(frequency, color="black", linestyle=":", linewidth=0.8)
            axis.set(xlabel="Frequency [Hz]", ylabel=f"({unit})^2/Hz", xlim=(0, 0.5))
            axis.set_yscale("log")
        else:
            axis.set(xlabel="Time [s]", ylabel=unit)
        axis.set_title(DOFS[index].capitalize(), fontsize=11)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(title, fontsize=12)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-csv", type=Path, default=ROOT / "OrcaFlex_Results/time_response.csv")
    parser.add_argument("--rao-csv", type=Path, default=ROOT / "OrcaFlex_Results/RAO_response.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/orcaflex_comparison")
    parser.add_argument("--design", type=Path, default=ROOT / "designs/VolturnUS-S.yaml")
    parser.add_argument("--wave-height", type=float, default=0.125, help="Regular crest-to-trough height [m]; source labels it Hs.")
    parser.add_argument("--period", type=float, default=9.875)
    parser.add_argument("--wind-speed-10m", type=float, default=3.93)
    parser.add_argument("--shear-exponent", type=float, default=0.12, help="Assumed wind shear, not measured.")
    parser.add_argument("--wave-heading", type=float, default=0, help="RAFT global propagation heading [deg]; provisional axis mapping.")
    parser.add_argument("--wind-heading", type=float, default=0)
    parser.add_argument("--welch-seconds", type=float, default=200)
    parser.add_argument("--orcaflex-only", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if args.period <= 0 or args.wave_height < 0 or args.welch_seconds <= 0 or args.wind_speed_10m < 0:
        raise ValueError("Period/segment duration must be positive; wave height and wind speed nonnegative.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    time, channels = read_time_history(args.time_csv)
    frequency = 1 / args.period
    motions = np.column_stack([channels[name] for name in MOTION_COLUMNS])
    accelerations = np.column_stack([channels[name] for name in ACCEL_COLUMNS])
    accelerations[:, 3:] = np.rad2deg(accelerations[:, 3:])
    global_accelerations = np.column_stack([channels[f"G{axis} acceleration"] for axis in "XYZ"])
    raos = read_raos(args.rao_csv)
    platform_rows = []
    for quantity in ("motion", "acceleration"):
        amplitudes = np.abs(interpolate_rao(raos[quantity], frequency)) * args.wave_height / 2
        amplitudes[3:] = np.rad2deg(amplitudes[3:])
        for index, dof in enumerate(DOFS):
            platform_rows.append(dict(dof=dof, quantity=quantity, source="OrcaFlex platform-origin RAO prediction",
                                      amplitude=float(amplitudes[index]), rms=float(amplitudes[index] / np.sqrt(2)),
                                      unit=("m" if index < 3 else "deg") + ("/s^2" if quantity == "acceleration" else "")))
    write_csv(output / "platform_rao_predictions.csv", platform_rows)
    summary = []
    for quantity, values in (("motion", motions), ("acceleration", accelerations)):
        summary.extend(statistics(time, values, frequency, quantity, "OrcaFlex dynamic/local channels; axis mapping unverified"))
        frequencies, density, welch_frequency, welch_density = spectra(time, values, args.welch_seconds)
        write_spectrum(output / f"orcaflex_{quantity}_PSD.csv", frequencies, density, quantity)
        write_spectrum(output / f"orcaflex_{quantity}_Welch_PSD.csv", welch_frequency, welch_density, quantity)
        plot_six(output / f"{quantity}_PSD.png", frequencies, density, quantity,
                 f"Turbine-point {quantity}: full-record Hann periodogram", frequency=frequency)
        plot_six(output / f"{quantity}_time_history.png", time, values, quantity,
                 f"Turbine-point {quantity}: native OrcaFlex dynamic/local axes")
    write_csv(output / "orcaflex_six_dof_statistics.csv", summary)
    write_csv(output / "orcaflex_global_position_acceleration.csv",
              [dict(time_s=float(time[index]), **{name: float(channels[name][index]) for name in GLOBAL_COLUMNS})
               for index in range(len(time))])
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest.update(sample_count=len(time), dt_s=float(np.median(np.diff(time))),
                    duration_s=float(time[-1] - time[0]), start_s=float(time[0]), end_s=float(time[-1]),
                    frequency_Hz=frequency, turbine_status="parked", turbulence=0,
                    wave_amplitude_m=args.wave_height / 2,
                    input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (args.time_csv, args.rao_csv)})
    notes = [
        "The complete exported record is post-startup; no samples were removed.",
        "Regular wave: wave_height=0.125 m is treated as crest-to-trough H, A=H/2. The source calls it Hs; confirm this convention in OrcaFlex.",
        "Motion/acceleration six-DOF time channels use native dynamic/local axes, not assumed global platform axes.",
        "Turbine attachment point and axis mapping are unverified. No quantitative hub error percentages are reported.",
        "Platform predictions come from origin RAOs, not measured platform time histories. No platform means/extrema can be recovered from these exports.",
        "RAFT parked status omits rotor aerodynamic loads. Steady parked wind loading may therefore differ from OrcaFlex.",
        "Wind is specified at 10 m; hub conversion uses an assumed power-law exponent, configurable with --shear-exponent.",
        "Head-on alignment is given, but world axes and from/to conventions are not verified; RAFT headings default provisionally to 0 degrees. OrcaFlex RAO direction is 180 degrees.",
        "PSD is one-sided density per Hz. RAFT density per rad/s must be multiplied by 2*pi, not divided.",
        "Regular-wave comparisons use line amplitude and variance; peak PSD height depends on record/window. No wave phase or coherence is inferred.",
        "Observed extrema, total RMS, fluctuating standard deviation, harmonic RMS and mean +/- 3std are distinct statistics.",
    ]
    if not args.orcaflex_only:
        raise NotImplementedError("RAFT adapter is not yet installed; use --orcaflex-only.")
    manifest["raft_status"] = "not requested"
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = "# Parked OrcaFlex / RAFT comparison\n\n"
    report += f"Record: {len(time)} samples, {time[-1] - time[0]:.3f} s, dt={np.median(np.diff(time)):.6f} s.\n\n"
    report += f"Wave: H={args.wave_height} m, T={args.period} s, f={frequency:.8f} Hz. Wind: {args.wind_speed_10m} m/s at 10 m.\n\n"
    report += "## Interpretation and limitations\n\n" + "\n".join(f"- {note}" for note in notes) + "\n"
    (output / "analysis_report.md").write_text(report, encoding="utf-8")
    print(f"Analyzed {len(time)} samples over {time[-1] - time[0]:.3f} s. Results: {output}")


if __name__ == "__main__":
    main()