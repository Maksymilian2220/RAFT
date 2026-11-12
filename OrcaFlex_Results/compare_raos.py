"""Compare RAOS-folder OrcaFlex motion RAOs with the saved RAFT RAO table.

Run from the repository root with:

    python OrcaFlex_Results/compare_raos.py
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAOS_DIR = ROOT / "OrcaFlex_Results/RAOS"
DEFAULT_RAFT_RAO = ROOT / "results/floater_raos.csv"
DEFAULT_OUTPUT = ROOT / "results/rao_comparison"
MOTION_FILES = {
    "surge": "IEA15MW_VolturnUS_Disp_RAO_surge_0deg.csv",
    "heave": "IEA15MW_VolturnUS_Disp_RAO_heave_0deg.csv",
    "pitch": "IEA15MW_VolturnUS_Disp_RAO_pitch_0deg.csv",
}
RAFT_DOFS = tuple(MOTION_FILES)
LOAD_FILES = {
    "surge": "IEA15MW_VolturnUS_Load_RAO_surge_0deg.csv",
    "heave": "IEA15MW_VolturnUS_Load_RAO_heave_0deg.csv",
    "pitch": "IEA15MW_VolturnUS_Load_RAO_pitch_0deg.csv",
}
DOF_UNITS = {"surge": "m/m", "heave": "m/m", "pitch": "rad/m"}


def read_raft_raos(path):
    """Return frequency and complex displacement RAOs indexed by compared DOF."""
    by_dof = {dof: {} for dof in RAFT_DOFS}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"frequency_hz", "dof", "displacement_rao_real", "displacement_rao_imag"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"RAFT RAO CSV must contain columns: {sorted(required)}")
        for row in reader:
            dof = row["dof"].strip().lower()
            if dof not in by_dof:
                if dof not in ("sway", "roll", "yaw"):
                    raise ValueError(f"Unexpected RAFT DOF: {dof!r}")
                continue
            frequency = float(row["frequency_hz"])
            value = complex(float(row["displacement_rao_real"]), float(row["displacement_rao_imag"]))
            if not math.isfinite(frequency) or frequency <= 0 or not np.isfinite([value.real, value.imag]).all():
                raise ValueError(f"Invalid RAFT RAO at {frequency} Hz for {dof}.")
            if frequency in by_dof[dof]:
                raise ValueError(f"Duplicate RAFT RAO at {frequency} Hz for {dof}.")
            by_dof[dof][frequency] = value

    if any(not values for values in by_dof.values()):
        missing = [dof for dof, values in by_dof.items() if not values]
        raise ValueError(f"RAFT RAO CSV is missing DOFs: {missing}")
    reference_frequencies = np.array(sorted(by_dof[RAFT_DOFS[0]]))
    for dof, values in by_dof.items():
        if not np.array_equal(reference_frequencies, np.array(sorted(values))):
            raise ValueError(f"RAFT frequency grid differs for {dof}.")
    return reference_frequencies, {
        dof: np.array([by_dof[dof][frequency] for frequency in reference_frequencies], dtype=complex)
        for dof in RAFT_DOFS
    }


def read_amplitude_rao_csv(path, dof, expected_units):
    """Read a metadata-prefixed OrcaFlex amplitude RAO CSV."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    metadata = [row[0].strip() for row in rows if row and row[0].strip()]
    frequency_unit = next(
        (value.split(":", 1)[1].strip() for value in metadata
         if value.startswith("Column A:")),
        None,
    )
    amplitude_unit = next(
        (value.split(":", 1)[1].strip() for value in metadata
         if value.startswith("Column B:")),
        None,
    )
    header_index = next(
        (index for index, row in enumerate(rows)
         if len(row) >= 2 and row[0].strip().startswith("Frequency (")
         and row[1].strip().startswith("Amplitude (")),
        None,
    )
    if header_index is None or frequency_unit is None or amplitude_unit is None:
        raise ValueError(f"Missing frequency/amplitude metadata in {path}")
    if expected_units not in amplitude_unit:
        raise ValueError(
            f"Expected {expected_units} amplitude units for {dof}, "
            f"got {amplitude_unit}"
        )
    frequency_scale = 1.0 / (2.0 * np.pi) if "rad/s" in frequency_unit else 1.0
    if "rad/s" not in frequency_unit and "Hz" not in frequency_unit:
        raise ValueError(f"Unsupported frequency units: {frequency_unit}")
    samples = []
    for row in rows[header_index + 1:]:
        if len(row) < 2 or not row[0].strip() or not row[1].strip():
            continue
        try:
            frequency = float(row[0]) * frequency_scale
            amplitude = float(row[1])
        except ValueError:
            continue
        if (frequency <= 0 or amplitude < 0
                or not np.isfinite([frequency, amplitude]).all()):
            raise ValueError(f"Invalid RAO sample for {dof} in {path}")
        samples.append((frequency, amplitude))
    samples.sort(key=lambda item: item[0])
    if len(samples) < 2:
        raise ValueError(f"Expected at least two RAO samples in {path}")
    return {
        "dof": dof,
        "frequencies_hz": np.array([item[0] for item in samples]),
        "amplitudes": np.array([item[1] for item in samples]),
        "frequency_units_in_file": frequency_unit,
        "frequency_column": rows[header_index][0].strip(),
        "amplitude_column": rows[header_index][1].strip(),
        "amplitude_units": expected_units,
        "source_file": str(Path(path)),
    }


def read_orcaflex_motion_rao(path, dof):
    """Read one OrcaFlex displacement file from the RAOS folder."""
    data = read_amplitude_rao_csv(path, dof, DOF_UNITS[dof])
    frequencies = data["frequencies_hz"]
    if len(frequencies) < 2 or np.any(np.diff(frequencies) <= 0):
        raise ValueError(f"RAO frequencies must increase strictly for {dof} in {path}")
    return data


def inspect_load_csv(path, dof):
    units = "kN.m/m" if dof == "pitch" else "kN/m"
    data = read_amplitude_rao_csv(path, dof, units)
    return {
        "dof": dof,
        "source_file": data["source_file"],
        "frequency_units_in_file": data["frequency_units_in_file"],
        "frequency_column": data["frequency_column"],
        "amplitude_column": data["amplitude_column"],
        "amplitude_units": data["amplitude_units"],
        "sample_count": len(data["frequencies_hz"]),
        "frequency_min_hz": float(data["frequencies_hz"].min()),
        "frequency_max_hz": float(data["frequencies_hz"].max()),
        "phase_available": False,
        "load_definition": "Not stated in CSV metadata",
        "reference_frame_and_point": "Not stated in CSV metadata",
    }


def interpolate_complex(frequencies, values, targets):
    """Interpolate RAFT complex values without extrapolation."""
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(values, dtype=complex)
    targets = np.asarray(targets, dtype=float)
    if frequencies.ndim != 1 or values.shape != frequencies.shape or len(frequencies) < 2:
        raise ValueError("Interpolation needs matching one-dimensional arrays with two samples.")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("Interpolation frequencies must increase strictly.")
    if np.any(targets < frequencies[0]) or np.any(targets > frequencies[-1]):
        raise ValueError("Requested frequency is outside the RAFT RAO range.")
    real = np.interp(targets, frequencies, values.real)
    imaginary = np.interp(targets, frequencies, values.imag)
    return real + 1j * imaginary


def compare_raos(orcaflex_raos, raft_frequencies, raft_values, relative_floor=1e-8):
    """Compare RAOS-folder amplitudes with RAFT magnitudes at shared frequencies."""
    rows = []
    for dof, of_data in orcaflex_raos.items():
        frequencies = of_data["frequencies_hz"]
        start = max(float(frequencies[0]), float(raft_frequencies[0]))
        stop = min(float(frequencies[-1]), float(raft_frequencies[-1]))
        selected = (frequencies >= start) & (frequencies <= stop)
        if not np.any(selected):
            raise ValueError(f"OrcaFlex and RAFT have no overlapping frequencies for {dof}.")
        for frequency, of_amplitude in zip(frequencies[selected], of_data["amplitudes"][selected]):
            raft_value = complex(
                interpolate_complex(raft_frequencies, raft_values[dof], [frequency])[0]
            )
            raft_amplitude = abs(raft_value)
            absolute_error = abs(of_amplitude - raft_amplitude)
            relative_error = (
                absolute_error / raft_amplitude * 100
                if raft_amplitude >= relative_floor else math.nan
            )
            rows.append({
                "frequency_hz": float(frequency),
                "period_s": 1.0 / float(frequency),
                "dof": dof,
                "unit": of_data["amplitude_units"],
                "raft_amplitude_interpolated": raft_amplitude,
                "orcaflex_amplitude": float(of_amplitude),
                "absolute_amplitude_error": absolute_error,
                "relative_amplitude_error_percent": relative_error,
                "phase_comparison": "unavailable; OrcaFlex RAOS file has amplitude only",
            })
    return rows


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "" if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in row.items()
            })


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_comparison(path, orcaflex_raos, raft_frequencies, raft_values):
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    for index, dof in enumerate(RAFT_DOFS):
        axis = axes.flat[index]
        orcaflex = orcaflex_raos[dof]
        axis.plot(
            raft_frequencies,
            np.abs(raft_values[dof]),
            label="RAFT magnitude",
        )
        axis.plot(
            orcaflex["frequencies_hz"],
            orcaflex["amplitudes"],
            "o",
            markersize=3,
            label="OrcaFlex RAOS amplitude",
        )
        axis.set(
            title=dof.capitalize(),
            xlabel="Frequency [Hz]",
            ylabel=f"RAO [{orcaflex['amplitude_units']}]",
        )
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend(fontsize=8)
    figure.suptitle("OrcaFlex and RAFT motion RAO magnitudes")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _report_text(rao_rows, load_inventory, motion_paths, raft_path, inputs):
    frequencies = sorted({row["frequency_hz"] for row in rao_rows})
    target_frequency = 1.0 / 9.875
    target_frequency = min(frequencies, key=lambda value: abs(value - target_frequency))
    report = [
        "# OrcaFlex and RAFT RAO comparison",
        "",
        f"Compared RAOS-folder OrcaFlex files at {len(frequencies)} distinct frequencies from {frequencies[0]:.6g} to {frequencies[-1]:.6g} Hz.",
        "",
        "## Motion RAOs",
        "",
        "The RAOS displacement files contain amplitude only, with no phase. RAFT complex values are interpolated in real/imaginary coordinates and reduced to magnitude for comparison. Phase differences are therefore unavailable. Relative errors are blank when the RAFT amplitude is below 1e-8 in the DOF's RAO units.",
        "",
        "The plot shows RAFT and OrcaFlex at their original frequency samples; the comparison table uses the interpolated RAFT values. The saved RAFT RAO CSV does not record wave amplitude, so it cannot identify frequencies where the wave component is zero or too small for a reliable ratio. A saved zero at such a frequency is not evidence of a zero physical RAO. RAFT's getRAO helper excludes wave amplitudes at or below 1e-6 m; the exporter instead divides by every nonzero wave component.",
        "",
        "Only the three per-DOF OrcaFlex displacement CSVs in `OrcaFlex_Results/RAOS/` are used as OrcaFlex motion inputs. Their filenames specify 0 degrees. The RAFT generator uses heading 0 degrees; physical axis and direction conventions remain unverified.",
        "",
        "RAFT generation settings visible in `examples/special_course_single_raos.py`: VolturnUS-S design, parked turbine, zero wind, JONSWAP, Tp=16 s, Hs=7 m, gamma=3.3, heading=0 degrees. These are source defaults, not embedded provenance in the RAO CSV.",
        "",
        f"At the compared sample nearest 9.875 s ({target_frequency:.8f} Hz):",
        "",
        "| DOF | Unit | OrcaFlex amplitude | RAFT interpolated amplitude | Absolute difference | Relative difference |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dof in RAFT_DOFS:
        row = next(item for item in rao_rows if item["dof"] == dof and item["frequency_hz"] == target_frequency)
        relative = "n/a" if math.isnan(row["relative_amplitude_error_percent"]) else f"{row['relative_amplitude_error_percent']:.2f}%"
        report.append(
            f"| {dof} | {row['unit']} | {row['orcaflex_amplitude']:.6g} | "
            f"{row['raft_amplitude_interpolated']:.6g} | "
            f"{row['absolute_amplitude_error']:.6g} | {relative} |"
        )
    report.extend([
        "",
        "## Load RAO files",
        "",
        "The three load files are plain-text CSVs with frequency and amplitude only, without phase. Their metadata does not identify a load definition, reference frame, or reference point. RAFT's current saved RAO table contains motion/velocity/acceleration RAOs, not force RAOs, so these load amplitudes are inventoried but not quantitatively compared.",
        "",
        "| DOF | Amplitude units | Samples | Frequency range [Hz] | Phase available |",
        "|---|---|---:|---:|---|",
    ])
    for item in load_inventory:
        report.append(f"| {item['dof']} | {item['amplitude_units']} | {item['sample_count']} | {item['frequency_min_hz']:.6g} to {item['frequency_max_hz']:.6g} | no |")
    report.extend([
        "",
        "## Caveats",
        "",
        "- The per-DOF OrcaFlex files do not state a reference point; the RAFT platform reference point and axis mapping have not been independently verified.",
        "- The OrcaFlex motion files specify 0-degree filenames, but physical heading alignment is not independently established.",
        "- RAFT's JONSWAP case and OrcaFlex's exported RAO conditions are not proven identical by the available files.",
        "- Interpolated values are not extrapolated beyond shared frequency coverage.",
        "",
        "## Inputs",
        "",
        "- OrcaFlex motion inputs (all in `OrcaFlex_Results/RAOS/`):",
        *[f"  - `{path.name}` (SHA-256 `{inputs[str(path)]}`)" for path in motion_paths],
        f"- RAFT motion RAOs: `{raft_path}` (SHA-256 `{inputs[str(raft_path)]}`)",
        "- OrcaFlex load CSV hashes and metadata are recorded in `load_rao_inventory.csv` and `run_manifest.json`.",
        "",
    ])
    return "\n".join(report)


def run(raos_dir=DEFAULT_RAOS_DIR, raft_path=DEFAULT_RAFT_RAO, output=DEFAULT_OUTPUT):
    raos_dir, raft_path, output = map(Path, (raos_dir, raft_path, output))
    orcaflex_raos = {
        dof: read_orcaflex_motion_rao(raos_dir / filename, dof)
        for dof, filename in MOTION_FILES.items()
    }
    raft_frequencies, raft_values = read_raft_raos(raft_path)
    rows = compare_raos(orcaflex_raos, raft_frequencies, raft_values)
    load_inventory = [
        inspect_load_csv(raos_dir / filename, dof)
        for dof, filename in LOAD_FILES.items()
    ]
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "motion_rao_comparison.csv", rows)
    _write_csv(output / "load_rao_inventory.csv", load_inventory)
    _plot_comparison(
        output / "motion_rao_comparison.png", orcaflex_raos,
        raft_frequencies, raft_values,
    )
    motion_paths = [raos_dir / filename for filename in MOTION_FILES.values()]
    input_paths = motion_paths + [raft_path] + [raos_dir / filename for filename in LOAD_FILES.values()]
    input_hashes = {str(path): _sha256(path) for path in input_paths}
    (output / "comparison_report.md").write_text(
        _report_text(rows, load_inventory, motion_paths, raft_path, input_hashes),
        encoding="utf-8",
    )
    manifest = {
        "orcaflex_inputs_scope": "Only per-DOF displacement and load files located in OrcaFlex_Results/RAOS/ are used.",
        "orcaflex_motion_dofs_compared": list(MOTION_FILES),
        "phase_comparison": "unavailable; OrcaFlex displacement files provide amplitude only",
        "frequency_sample_count": len({row["frequency_hz"] for row in rows}),
        "frequency_min_hz": min(row["frequency_hz"] for row in rows),
        "frequency_max_hz": max(row["frequency_hz"] for row in rows),
        "relative_error_floor": 1e-8,
        "input_sha256": input_hashes,
        "load_comparison_status": "inventory only; no matching RAFT force RAO or verified load definition/frame/reference point",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Compared {manifest['frequency_sample_count']} shared-frequency samples. Results: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raos-dir", type=Path, default=DEFAULT_RAOS_DIR)
    parser.add_argument("--raft-rao", type=Path, default=DEFAULT_RAFT_RAO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.raos_dir, args.raft_rao, args.output)


if __name__ == "__main__":
    main()