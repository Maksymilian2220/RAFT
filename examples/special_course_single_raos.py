"""Run one RAFT JONSWAP case and save RAOs and all RAFT channels.

Run from the repository root, for example:

    python examples/special_course_single_raos.py --period 8 --height 2
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
DEFAULT_RAO_OUTPUT = REPO_ROOT / "results" / "floater_raos.csv"
DEFAULT_RAO_PLOT_DIR = REPO_ROOT / "results" / "raos"
DEFAULT_CHANNEL_OUTPUT = REPO_ROOT / "results" / "all_raft_channels.csv"
DEFAULT_CHANNEL_PLOT_DIR = REPO_ROOT / "results" / "all_channels"

# Main JONSWAP inputs. Change these values here when running the file
# without command-line arguments.
TP = 16.0       # Peak period [s]
HS = 7.0       # Significant wave height [m]
GAMMA = 3.3   # JONSWAP peak enhancement factor [-]
MIN_FREQ = 0.01  # Frequency step and lower cutoff [Hz]
MAX_FREQ = 0.4   # Upper frequency cutoff [Hz]

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


def save_rao_csv(
    model: raft.Model,
    *,
    case_index: int,
    fowt_index: int,
    path: Path,
) -> None:
    """Save complex floater motion, velocity, and acceleration RAOs.

    RAFT stores the solved motion components in ``*_RA`` and the corresponding
    discrete wave amplitudes in ``fowt.zeta``. Dividing the former by the
    latter gives the response per unit wave amplitude. Rotational RAFT
    responses are converted from degrees to radians before the RAO is saved.
    """
    if not 0 <= fowt_index < len(model.fowtList):
        raise IndexError(f"FOWT index {fowt_index} is out of range.")

    fowt = model.fowtList[fowt_index]
    metrics = model.results["case_metrics"][case_index][fowt_index]
    frequencies_rad = np.asarray(model.results["freq_rad"], dtype=float)
    frequencies_hz = frequencies_rad / (2.0 * np.pi)
    wave_components = np.asarray(fowt.zeta[0, :], dtype=complex)
    fields = [
        "frequency_index",
        "frequency_rad_s",
        "frequency_hz",
        "period_s",
        "dof",
        "displacement_rao_real",
        "displacement_rao_imag",
        "displacement_rao_amplitude",
        "displacement_rao_phase_deg",
        "velocity_rao_real",
        "velocity_rao_imag",
        "velocity_rao_amplitude",
        "velocity_rao_phase_deg",
        "acceleration_rao_real",
        "acceleration_rao_imag",
        "acceleration_rao_amplitude",
        "acceleration_rao_phase_deg",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frequency_index, (omega, frequency_hz) in enumerate(
            zip(frequencies_rad, frequencies_hz)
        ):
            wave_component = wave_components[frequency_index]
            for dof in DOFS:
                displacement = complex(
                    metrics[f"{dof}_RA"][0, frequency_index]
                )
                if dof in ("roll", "pitch", "yaw"):
                    displacement *= np.pi / 180.0

                if abs(wave_component) > 0.0:
                    displacement_rao = displacement / wave_component
                else:
                    displacement_rao = 0.0j

                velocity_rao = 1j * omega * displacement_rao
                acceleration_rao = -(omega**2) * displacement_rao

                writer.writerow(
                    {
                        "frequency_index": frequency_index,
                        "frequency_rad_s": omega,
                        "frequency_hz": frequency_hz,
                        "period_s": 1.0 / frequency_hz,
                        "dof": dof,
                        "displacement_rao_real": displacement_rao.real,
                        "displacement_rao_imag": displacement_rao.imag,
                        "displacement_rao_amplitude": abs(displacement_rao),
                        "displacement_rao_phase_deg": np.angle(
                            displacement_rao, deg=True
                        ),
                        "velocity_rao_real": velocity_rao.real,
                        "velocity_rao_imag": velocity_rao.imag,
                        "velocity_rao_amplitude": abs(velocity_rao),
                        "velocity_rao_phase_deg": np.angle(
                            velocity_rao, deg=True
                        ),
                        "acceleration_rao_real": acceleration_rao.real,
                        "acceleration_rao_imag": acceleration_rao.imag,
                        "acceleration_rao_amplitude": abs(acceleration_rao),
                        "acceleration_rao_phase_deg": np.angle(
                            acceleration_rao, deg=True
                        ),
                    }
                )


def save_rao_plot(
    model: raft.Model,
    *,
    case_index: int,
    fowt_index: int,
    directory: Path,
) -> None:
    """Save one displacement-RAO magnitude plot for each response."""
    if not 0 <= fowt_index < len(model.fowtList):
        raise IndexError(f"FOWT index {fowt_index} is out of range.")

    fowt = model.fowtList[fowt_index]
    metrics = model.results["case_metrics"][case_index][fowt_index]
    frequencies_rad = np.asarray(model.results["freq_rad"], dtype=float)
    frequencies_hz = frequencies_rad / (2.0 * np.pi)
    wave_components = np.asarray(fowt.zeta[0, :], dtype=complex)

    def displacement_rao(values, rotational=False):
        response = np.asarray(values[0, :], dtype=complex)
        if rotational:
            response = response * np.pi / 180.0
        rao = np.zeros_like(response)
        nonzero_wave = np.abs(wave_components) > 0.0
        rao[nonzero_wave] = response[nonzero_wave] / wave_components[nonzero_wave]
        return np.abs(rao)

    translations = {
        "Surge": displacement_rao(metrics["surge_RA"]),
        "Sway": displacement_rao(metrics["sway_RA"]),
        "Heave": displacement_rao(metrics["heave_RA"]),
    }
    rotations = {
        "Roll": displacement_rao(metrics["roll_RA"], rotational=True),
        "Pitch": displacement_rao(metrics["pitch_RA"], rotational=True),
        "Yaw": displacement_rao(metrics["yaw_RA"], rotational=True),
    }

    rotor = fowt.rotorList[0]
    node_id = rotor.nodeList[0].id
    nacelle_response = fowt.Xi_fullDOF[
        0, node_id * 6, :
    ]
    nacelle_rao = np.zeros_like(nacelle_response, dtype=complex)
    nonzero_wave = np.abs(wave_components) > 0.0
    nacelle_rao[nonzero_wave] = (
        nacelle_response[nonzero_wave] / wave_components[nonzero_wave]
    )
    translations["Nacelle surge"] = np.abs(nacelle_rao)

    responses = {
        **translations,
        **rotations,
    }
    units = {
        **{label: "RAO [m/m]" for label in translations},
        **{label: "RAO [rad/m]" for label in rotations},
    }
    colors = {
        "Surge": "tab:blue",
        "Sway": "tab:orange",
        "Heave": "tab:green",
        "Nacelle surge": "tab:red",
        "Roll": "tab:purple",
        "Pitch": "tab:brown",
        "Yaw": "tab:gray",
    }

    directory.mkdir(parents=True, exist_ok=True)
    for label, values in responses.items():
        filename = f"{label.lower().replace(' ', '_')}_rao.png"
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.plot(
            frequencies_hz,
            values,
            color=colors[label],
            linewidth=2.0,
        )
        axis.set(
            xlabel="Frequency [Hz]",
            ylabel=units[label],
            title=f"{label} displacement RAO",
            xlim=(frequencies_hz.min(), frequencies_hz.max()),
        )
        axis.grid(True, which="major", color="0.85", linewidth=0.8)
        axis.grid(True, which="minor", color="0.93", linewidth=0.5)
        axis.minorticks_on()
        axis.spines[["top", "right"]].set_visible(False)
        figure.savefig(directory / filename, dpi=180)
        plt.close(figure)


def save_all_channels_csv(
    model: raft.Model,
    *,
    case_index: int,
    fowt_index: int,
    path: Path,
) -> None:
    """Save every numeric per-FOWT RAFT output channel.

    Scalar channels are written once with an empty frequency. Arrays whose
    final dimension is the RAFT frequency grid are expanded into one row per
    leading-index combination. This includes platform responses, PSDs,
    nacelle accelerations, tower moments, rotor channels, and mooring arrays.
    """
    if not 0 <= fowt_index < len(model.fowtList):
        raise IndexError(f"FOWT index {fowt_index} is out of range.")

    metrics = model.results["case_metrics"][case_index][fowt_index]
    frequencies_rad = np.asarray(model.results["freq_rad"], dtype=float)
    frequencies_hz = frequencies_rad / (2.0 * np.pi)
    frequency_count = len(frequencies_rad)
    fields = [
        "channel",
        "component_index",
        "frequency_index",
        "frequency_rad_s",
        "frequency_hz",
        "period_s",
        "value_real",
        "value_imag",
        "value_amplitude",
        "value_phase_deg",
    ]

    def write_value(writer, channel, component_index, value, frequency_index=None):
        complex_value = complex(value)
        if frequency_index is None:
            omega = frequency_hz_value = period = ""
        else:
            omega = frequencies_rad[frequency_index]
            frequency_hz_value = frequencies_hz[frequency_index]
            period = 1.0 / frequency_hz_value
        writer.writerow(
            {
                "channel": channel,
                "component_index": component_index,
                "frequency_index": (
                    "" if frequency_index is None else frequency_index
                ),
                "frequency_rad_s": omega,
                "frequency_hz": frequency_hz_value,
                "period_s": period,
                "value_real": complex_value.real,
                "value_imag": complex_value.imag,
                "value_amplitude": abs(complex_value),
                "value_phase_deg": np.angle(complex_value, deg=True),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for channel, raw_values in sorted(metrics.items()):
            values = np.asarray(raw_values)
            if not np.issubdtype(values.dtype, np.number):
                continue

            if values.ndim == 0:
                write_value(writer, channel, "", values.item())
                continue

            frequency_axis = -1
            if (
                values.shape[-1] != frequency_count
                and values.shape[0] == frequency_count
            ):
                frequency_axis = 0

            if (
                values.shape[frequency_axis] != frequency_count
            ):
                for component_index, value in enumerate(values.flat):
                    write_value(writer, channel, component_index, value)
                continue

            values_by_frequency = np.moveaxis(values, frequency_axis, -1)
            component_shape = values_by_frequency.shape[:-1]
            for component_index in np.ndindex(component_shape or (1,)):
                if not component_shape:
                    component_values = values_by_frequency
                    label = ""
                else:
                    component_values = values_by_frequency[component_index]
                    label = ",".join(str(index) for index in component_index)
                for frequency_index, value in enumerate(component_values):
                    write_value(
                        writer,
                        channel,
                        label,
                        value,
                        frequency_index,
                    )


def save_all_channel_plots(
    model: raft.Model,
    *,
    case_index: int,
    fowt_index: int,
    directory: Path,
) -> None:
    """Save one frequency-domain magnitude plot for every channel component."""
    if not 0 <= fowt_index < len(model.fowtList):
        raise IndexError(f"FOWT index {fowt_index} is out of range.")

    metrics = model.results["case_metrics"][case_index][fowt_index]
    frequencies_rad = np.asarray(model.results["freq_rad"], dtype=float)
    frequencies_hz = frequencies_rad / (2.0 * np.pi)
    frequency_count = len(frequencies_hz)
    directory.mkdir(parents=True, exist_ok=True)

    def safe_name(value):
        return "".join(
            character if character.isalnum() or character in "_-" else "_"
            for character in str(value)
        )

    for channel, raw_values in sorted(metrics.items()):
        values = np.asarray(raw_values)
        if not np.issubdtype(values.dtype, np.number) or values.ndim == 0:
            continue

        frequency_axis = -1
        if (
            values.shape[-1] != frequency_count
            and values.shape[0] == frequency_count
        ):
            frequency_axis = 0
        if values.shape[frequency_axis] != frequency_count:
            continue

        values_by_frequency = np.moveaxis(values, frequency_axis, -1)
        component_shape = values_by_frequency.shape[:-1]
        components = np.ndindex(component_shape or (1,))
        for component_index in components:
            if component_shape:
                component_values = values_by_frequency[component_index]
                component_label = ",".join(
                    str(index) for index in component_index
                )
            else:
                component_values = values_by_frequency
                component_label = ""

            figure, axis = plt.subplots(
                figsize=(8, 5),
                constrained_layout=True,
            )
            axis.plot(
                frequencies_hz,
                np.abs(np.asarray(component_values, dtype=complex)),
                linewidth=1.8,
                color="tab:blue",
            )
            axis.set(
                xlabel="Frequency [Hz]",
                ylabel="Magnitude",
                title=(
                    f"{channel}"
                    + (f" [{component_label}]" if component_label else "")
                ),
                xlim=(frequencies_hz.min(), frequencies_hz.max()),
            )
            axis.grid(True, which="major", color="0.85", linewidth=0.8)
            axis.grid(True, which="minor", color="0.93", linewidth=0.5)
            axis.minorticks_on()
            axis.spines[["top", "right"]].set_visible(False)
            suffix = f"_{safe_name(component_label)}" if component_label else ""
            figure.savefig(
                directory / f"{safe_name(channel)}{suffix}.png",
                dpi=180,
            )
            plt.close(figure)


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
) -> raft.Model:
    """Run RAFT and return the analyzed model."""
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
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAFT and save RAOs and all RAFT channels."
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
    parser.add_argument("--rao-output", type=Path, default=DEFAULT_RAO_OUTPUT)
    parser.add_argument("--rao-plot", type=Path, default=DEFAULT_RAO_PLOT_DIR)
    parser.add_argument(
        "--channel-output",
        type=Path,
        default=DEFAULT_CHANNEL_OUTPUT,
        help="CSV containing every numeric RAFT output channel.",
    )
    parser.add_argument(
        "--channel-plot",
        type=Path,
        default=DEFAULT_CHANNEL_PLOT_DIR,
        help="Directory for separate frequency-domain channel plots.",
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
    model = run(
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
    save_rao_csv(
        model,
        case_index=0,
        fowt_index=args.fowt_index,
        path=args.rao_output,
    )
    save_rao_plot(
        model,
        case_index=0,
        fowt_index=args.fowt_index,
        directory=args.rao_plot,
    )
    save_all_channels_csv(
        model,
        case_index=0,
        fowt_index=args.fowt_index,
        path=args.channel_output,
    )
    save_all_channel_plots(
        model,
        case_index=0,
        fowt_index=args.fowt_index,
        directory=args.channel_plot,
    )
    print(f"Saved floater RAOs to {args.rao_output}")
    print(f"Saved separate RAO plots to {args.rao_plot}")
    print(f"Saved all RAFT channels to {args.channel_output}")
    print(f"Saved separate channel plots to {args.channel_plot}")


if __name__ == "__main__":
    main()
