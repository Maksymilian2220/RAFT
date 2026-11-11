"""Run one RAFT JONSWAP case and extract one frequency component.

The selected component can be compared with a regular-wave OrcaFlex result.
The RAFT run itself still uses the complete JONSWAP spectrum; this script
only extracts the response at the requested period after the run.

Run from the repository root, for example:

    python examples/special_course_singlef.py --period 8 --height 2
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import raft
import yaml
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = REPO_ROOT / "designs" / "VolturnUS-S.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "single_frequency_response.csv"
DEFAULT_FULL_OUTPUT = REPO_ROOT / "results" / "full_frequency_response.csv"
DEFAULT_SPECTRUM_PLOT = REPO_ROOT / "results" / "jonswap_spectrum.png"

# Main JONSWAP inputs. Change these values here when running the file
# without command-line arguments.
TP = 16.0       # Peak period [s]
HS = 7.0       # Significant wave height [m]
GAMMA = 30    # JONSWAP peak enhancement factor [-]
MIN_FREQ = 0.0001  # Frequency step and lower cutoff [Hz]
MAX_FREQ = 0.15   # Upper frequency cutoff [Hz]

CASE_KEYS = [
    "wind_speed",
    "wind_heading",
    "turbulence",
    "turbine_status",
    "yaw_misalign",
    "wave_spectrum",
    "wave_period",
    "wave_height",
    "wave_heading",
    "wave_gamma",
]

DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def load_design(path: Path) -> dict:
    """Load a RAFT YAML design."""
    with path.open(encoding="utf-8") as handle:
        design = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(design, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")
    return design


def make_case(
    *,
    period: float,
    height: float,
    heading: float,
    wind_speed: float,
    turbulence: float,
    gamma: float,
) -> list:
    """Create one load case using the existing RAFT case format."""
    if period <= 0.0:
        raise ValueError("period must be positive.")
    if height < 0.0:
        raise ValueError("height must be non-negative.")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")

    return [
        wind_speed,
        heading,
        turbulence,
        "parked",
        0.0,
        "JONSWAP",
        period,
        height,
        heading,
        gamma,
    ]


def select_frequency(model: raft.Model, period: float) -> tuple[int, float]:
    """Return the nearest RAFT frequency index and its frequency in Hz."""
    frequencies_hz = np.asarray(model.results["freq_rad"]) / (2.0 * np.pi)
    target_frequency_hz = 1.0 / period
    index = int(np.argmin(np.abs(frequencies_hz - target_frequency_hz)))
    return index, float(frequencies_hz[index])


def extract_single_frequency(
    model: raft.Model,
    *,
    case_index: int,
    fowt_index: int,
    period: float,
) -> dict:
    """Extract wave and six-DOF response amplitudes at one frequency."""
    if not 0 <= fowt_index < len(model.fowtList):
        raise IndexError(f"FOWT index {fowt_index} is out of range.")

    metrics = model.results["case_metrics"][case_index][fowt_index]
    fowt = model.fowtList[fowt_index]
    frequency_index, frequency_hz = select_frequency(model, period)
    omega = 2.0 * np.pi * frequency_hz

    wave_component = complex(fowt.zeta[0, frequency_index])
    output = {
        "requested_period_s": period,
        "selected_frequency_hz": frequency_hz,
        "selected_period_s": 1.0 / frequency_hz,
        "frequency_index": frequency_index,
        "wave_amplitude_m": abs(wave_component),
        "wave_height_m": 2.0 * abs(wave_component),
        "wave_phase_deg": np.angle(wave_component, deg=True),
        "dof": [],
    }

    for dof in DOFS:
        displacement = complex(metrics[f"{dof}_RA"][0, frequency_index])
        velocity = 1j * omega * displacement
        acceleration = -(omega**2) * displacement
        output["dof"].append(
            {
                "name": dof,
                "displacement_amplitude": abs(displacement),
                "displacement_phase_deg": np.angle(displacement, deg=True),
                "velocity_amplitude": abs(velocity),
                "velocity_phase_deg": np.angle(velocity, deg=True),
                "acceleration_amplitude": abs(acceleration),
                "acceleration_phase_deg": np.angle(acceleration, deg=True),
            }
        )

    return output


def save_csv(result: dict, path: Path) -> None:
    """Save the selected wave component and six-DOF response to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dof",
        "selected_frequency_hz",
        "selected_period_s",
        "wave_amplitude_m",
        "wave_height_m",
        "displacement_amplitude",
        "displacement_phase_deg",
        "velocity_amplitude",
        "velocity_phase_deg",
        "acceleration_amplitude",
        "acceleration_phase_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dof_result in result["dof"]:
            writer.writerow(
                {
                    "dof": dof_result["name"],
                    "selected_frequency_hz": result["selected_frequency_hz"],
                    "selected_period_s": result["selected_period_s"],
                    "wave_amplitude_m": result["wave_amplitude_m"],
                    "wave_height_m": result["wave_height_m"],
                    **{
                        key: dof_result[key]
                        for key in fields
                        if key in dof_result
                    },
                }
            )


def save_full_frequency_csv(
    model: raft.Model,
    *,
    case_index: int,
    fowt_index: int,
    path: Path,
) -> None:
    """Save wave and six-DOF response results at every RAFT frequency."""
    if not 0 <= fowt_index < len(model.fowtList):
        raise IndexError(f"FOWT index {fowt_index} is out of range.")

    fowt = model.fowtList[fowt_index]
    metrics = model.results["case_metrics"][case_index][fowt_index]
    frequencies_rad = np.asarray(model.results["freq_rad"])
    frequencies_hz = frequencies_rad / (2.0 * np.pi)
    fields = [
        "frequency_index",
        "frequency_rad_s",
        "frequency_hz",
        "period_s",
        "wave_amplitude_m",
        "wave_height_m",
        "wave_phase_deg",
        "wave_psd_m2_per_rad_s",
        "dof",
        "displacement_amplitude",
        "displacement_phase_deg",
        "displacement_psd",
        "velocity_amplitude",
        "velocity_phase_deg",
        "velocity_psd",
        "acceleration_amplitude",
        "acceleration_phase_deg",
        "acceleration_psd",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frequency_index, (omega, frequency_hz) in enumerate(
            zip(frequencies_rad, frequencies_hz)
        ):
            wave = complex(fowt.zeta[0, frequency_index])
            wave_psd = 0.5 * abs(wave)**2 / fowt.dw
            frequency_rows = []
            for dof in DOFS:
                displacement = complex(
                    metrics[f"{dof}_RA"][0, frequency_index]
                )
                velocity = 1j * omega * displacement
                acceleration = -(omega**2) * displacement
                frequency_rows.append(
                    {
                        "frequency_index": frequency_index,
                        "frequency_rad_s": omega,
                        "frequency_hz": frequency_hz,
                        "period_s": 1.0 / frequency_hz,
                        "wave_amplitude_m": abs(wave),
                        "wave_height_m": 2.0 * abs(wave),
                        "wave_phase_deg": np.angle(wave, deg=True),
                        "wave_psd_m2_per_rad_s": wave_psd,
                        "dof": dof,
                        "displacement_amplitude": abs(displacement),
                        "displacement_phase_deg": np.angle(
                            displacement, deg=True
                        ),
                        "displacement_psd": 0.5 * abs(displacement)**2 / fowt.dw,
                        "velocity_amplitude": abs(velocity),
                        "velocity_phase_deg": np.angle(velocity, deg=True),
                        "velocity_psd": 0.5 * abs(velocity)**2 / fowt.dw,
                        "acceleration_amplitude": abs(acceleration),
                        "acceleration_phase_deg": np.angle(
                            acceleration, deg=True
                        ),
                        "acceleration_psd": 0.5 * abs(acceleration)**2 / fowt.dw,
                    }
                )
            writer.writerows(frequency_rows)


def save_spectrum_plot(
    model: raft.Model,
    *,
    fowt_index: int,
    period: float,
    path: Path,
) -> dict:
    """Save the exact discrete RAFT input spectrum and report Hs checks."""
    fowt = model.fowtList[fowt_index]
    frequencies_hz = np.asarray(model.results["freq_rad"]) / (2.0 * np.pi)
    spectrum = np.asarray(fowt.S[0, :])
    amplitudes = np.asarray(fowt.zeta[0, :])
    target_frequency_hz = 1.0 / period
    selected_index = int(np.argmin(
        np.abs(frequencies_hz - target_frequency_hz)
    ))
    selected_frequency_hz = float(frequencies_hz[selected_index])

    hs_from_spectrum = 4.0 * np.sqrt(
        np.sum(spectrum * fowt.dw)
    )
    hs_from_amplitudes = 4.0 * np.sqrt(
        0.5 * np.sum(np.abs(amplitudes) ** 2)
    )
    hs_requested = float(model.design["cases"]["data"][0][7])
    tolerance = max(0.01 * hs_requested, 1e-9)
    if abs(hs_from_spectrum - hs_requested) > tolerance:
        raise ValueError(
            "Discrete JONSWAP spectrum does not reproduce the requested "
            f"Hs within 1%: requested={hs_requested:.6f}, "
            f"calculated={hs_from_spectrum:.6f}."
        )
    if not np.isclose(hs_from_spectrum, hs_from_amplitudes, rtol=1e-10):
        raise ValueError(
            "Hs calculated from S and zeta are inconsistent: "
            f"{hs_from_spectrum:.6f} versus {hs_from_amplitudes:.6f}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        frequencies_hz,
        spectrum,
        label="JONSWAP spectrum used by RAFT",
    )
    axis.axvline(
        target_frequency_hz,
        color="tab:red",
        linestyle="--",
        label=f"Target 1/Tp = {target_frequency_hz:.6f} Hz",
    )
    axis.axvline(
        selected_frequency_hz,
        color="tab:green",
        linestyle=":",
        label=f"Selected grid frequency = {selected_frequency_hz:.6f} Hz",
    )
    axis.set(
        xlabel="Frequency [Hz]",
        ylabel=r"$S_\eta$ [m$^2$/(rad/s)]",
        title=(
            f"RAFT JONSWAP spectrum: Hs={hs_requested:.3f} m, "
            f"Tp={period:.3f} s, gamma="
            f"{model.design['cases']['data'][0][9]:.3g}"
        ),
        xlim=(0.0, frequencies_hz[-1]),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)

    return {
        "hs_requested": hs_requested,
        "hs_from_spectrum": hs_from_spectrum,
        "hs_from_amplitudes": hs_from_amplitudes,
        "target_frequency_hz": target_frequency_hz,
        "selected_frequency_hz": selected_frequency_hz,
    }


def run(
    *,
    design_path: Path = DEFAULT_DESIGN,
    period: float = TP,
    height: float = HS,
    heading: float = 0.0,
    wind_speed: float = 0.0,
    turbulence: float = 0.0,
    gamma: float = GAMMA,
    fowt_index: int = 0,
    display: int = 1,
) -> tuple[raft.Model, dict]:
    """Run RAFT and return the model plus the extracted result."""
    if MIN_FREQ <= 0.0:
        raise ValueError("MIN_FREQ must be positive.")
    if MAX_FREQ < MIN_FREQ:
        raise ValueError("MAX_FREQ must be greater than or equal to MIN_FREQ.")

    design = load_design(design_path)
    design.setdefault("settings", {})
    design["settings"]["min_freq"] = MIN_FREQ
    design["settings"]["max_freq"] = MAX_FREQ
    design["cases"] = {
        "keys": CASE_KEYS,
        "data": [
            make_case(
                period=period,
                height=height,
                heading=heading,
                wind_speed=wind_speed,
                turbulence=turbulence,
                gamma=gamma,
            )
        ],
    }

    model = raft.Model(design)
    model.analyzeUnloaded()
    model.analyzeCases(display=display)
    result = extract_single_frequency(
        model,
        case_index=0,
        fowt_index=fowt_index,
        period=period,
    )
    return model, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAFT and extract one JONSWAP frequency component."
    )
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--tp",
        "--period",
        dest="period",
        type=float,
        default=TP,
        help="JONSWAP peak period Tp [s].",
    )
    parser.add_argument(
        "--hs",
        "--height",
        dest="height",
        type=float,
        default=HS,
        help="JONSWAP significant wave height Hs [m].",
    )
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--wind-speed", type=float, default=0.0)
    parser.add_argument("--turbulence", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--fowt-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_OUTPUT)
    parser.add_argument(
        "--spectrum-plot",
        type=Path,
        default=DEFAULT_SPECTRUM_PLOT,
    )
    parser.add_argument("--quiet", action="store_true")
    argv = sys.argv[1:]
    if "ipykernel" in sys.modules:
        # Jupyter passes its connection-file argument through sys.argv.
        # It is not an argument for this script.
        cleaned_argv = []
        skip_next = False
        for index, argument in enumerate(argv):
            if skip_next:
                skip_next = False
                continue
            if (
                argument == "--fowt-index"
                and index + 1 < len(argv)
                and argv[index + 1].endswith(".json")
            ):
                skip_next = True
                continue
            cleaned_argv.append(argument)
        argv = cleaned_argv
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    model, result = run(
        design_path=args.design,
        period=args.period,
        height=args.height,
        heading=args.heading,
        wind_speed=args.wind_speed,
        turbulence=args.turbulence,
        gamma=args.gamma,
        fowt_index=args.fowt_index,
        display=0 if args.quiet else 1,
    )
    save_csv(result, args.output)
    save_full_frequency_csv(
        model,
        case_index=0,
        fowt_index=args.fowt_index,
        path=args.full_output,
    )
    spectrum_check = save_spectrum_plot(
        model,
        fowt_index=args.fowt_index,
        period=args.period,
        path=args.spectrum_plot,
    )

    print(
        f"Selected frequency: {result['selected_frequency_hz']:.6f} Hz "
        f"(period {result['selected_period_s']:.6f} s)"
    )
    print(
        f"Selected JONSWAP component: amplitude "
        f"{result['wave_amplitude_m']:.6e} m, "
        f"equivalent height {result['wave_height_m']:.6e} m"
    )
    for dof in result["dof"]:
        print(
            f"{dof['name']:>6}: "
            f"x={dof['displacement_amplitude']:.6e}, "
            f"v={dof['velocity_amplitude']:.6e}, "
            f"a={dof['acceleration_amplitude']:.6e}"
        )
    print(f"Saved extracted response to {args.output}")
    print(f"Saved full-frequency response to {args.full_output}")
    print(
        "Hs check: requested="
        f"{spectrum_check['hs_requested']:.6f} m, "
        "from S="
        f"{spectrum_check['hs_from_spectrum']:.6f} m, "
        "from zeta="
        f"{spectrum_check['hs_from_amplitudes']:.6f} m"
    )
    print(f"Saved spectrum plot to {args.spectrum_plot}")


if __name__ == "__main__":
    main()
