"""Generate a deterministic Dirac-limit (regular-wave) time series with RAFT.

RAFT is used to calculate the complex response at the frequency nearest the
JONSWAP peak. The response is then converted to a regular-wave response by
dividing by RAFT's complex wave component and multiplying by H/2.

Physics
-------
For a linear system, the wave elevation and response are represented by
complex harmonic amplitudes:

    eta(t) = Re{a exp(i omega t)}
    x_j(t) = Re{X_j exp(i omega t)}

The regular-wave amplitude is a = H/2. In the narrow-band (Dirac) limit, the
wave spectrum can be written as

    S_eta(omega) = m0 delta(omega - omega_p),
    m0 = integral(S_eta d omega) = Hs**2 / 16.

The corresponding response spectrum is

    S_xj(omega) = |H_j(omega)|**2 S_eta(omega),

where H_j is the complex RAO. This script obtains that RAO from the finite
band RAFT result, H_j = X_j / a_RAFT, and then evaluates the deterministic
response at omega_p. The resulting sinusoid has amplitude |H_j| H/2 and
RMS value |H_j| H/(2 sqrt(2)) = |H_j| Hs/4.

Time differentiation is performed in the frequency domain:

    velocity amplitude     = i omega X_j
    acceleration amplitude  = -omega**2 X_j.

This is a linear, first-order wave-frequency response. It does not include
second-order difference-frequency drift, sum-frequency loads, or nonlinear
mooring and viscous effects.

For example, from the repository root:

    python examples/diric_timeseries.py --tp 8 --hs 2 --cycles 20

The output contains the wave elevation and displacement, velocity, and
acceleration for all six rigid-body degrees of freedom. Rotational quantities
are in radians, radians/s, and radians/s^2.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import raft
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = REPO_ROOT / "designs" / "VolturnUS-S.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "dirac_time_series.csv"
DEFAULT_PLOT = REPO_ROOT / "results" / "dirac_time_series.png"
DEFAULT_SPECTRUM_OUTPUT = REPO_ROOT / "results" / "jonswap_spectrum.csv"
DEFAULT_SPECTRUM_PLOT = REPO_ROOT / "results" / "jonswap_spectrum.png"

# Main user inputs. Command-line arguments override these values.
HS_M = 6.0
TP_S = 12.0
GAMMA = 32.0
TIME_STEP_S = 0.1
NUMBER_OF_CYCLES = 10.0

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
RAFT_JONSWAP_GAMMA_LIMIT = np.exp(1.0 / 0.287)


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
    if height <= 0.0:
        raise ValueError("height must be positive.")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if gamma >= RAFT_JONSWAP_GAMMA_LIMIT:
        raise ValueError(
            f"gamma={gamma:g} is unsupported by RAFT's JONSWAP "
            f"normalization; use gamma < {RAFT_JONSWAP_GAMMA_LIMIT:.2f}."
        )
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


def extract_regular_response(
    model: raft.Model,
    *,
    period: float,
    height: float,
    fowt_index: int,
) -> dict:
    """Extract complex regular-wave response amplitudes from a RAFT case."""
    if not 0 <= fowt_index < len(model.fowtList):
        raise IndexError(f"FOWT index {fowt_index} is out of range.")

    frequency_index, frequency_hz = select_frequency(model, period)
    omega = 2.0 * np.pi * frequency_hz
    fowt = model.fowtList[fowt_index]
    metrics = model.results["case_metrics"][0][fowt_index]
    wave_component = complex(fowt.zeta[0, frequency_index])
    if abs(wave_component) == 0.0:
        raise ValueError("RAFT returned a zero wave component at the selected frequency.")

    # RAFT's JONSWAP run supplies a finite-band complex wave amplitude
    # a_RAFT and response X_j. Their ratio is the complex RAO, independent
    # of the finite-band wave amplitude. We then apply the deterministic
    # regular-wave amplitude a = H/2, as required by the Dirac-limit model.
    wave_amplitude = height / 2.0
    response_amplitudes = []
    for dof in DOFS:
        stochastic_response = complex(metrics[f"{dof}_RA"][0, frequency_index])
        amplitude = stochastic_response / wave_component * wave_amplitude
        if dof in ("roll", "pitch", "yaw"):
            # RAFT stores rotational RA outputs in degrees.
            amplitude *= np.pi / 180.0
        response_amplitudes.append(amplitude)

    return {
        "omega": omega,
        "frequency_hz": frequency_hz,
        "period_s": 1.0 / frequency_hz,
        "wave_amplitude_m": wave_amplitude,
        "response_amplitudes": np.asarray(response_amplitudes, dtype=complex),
    }


def make_time_series(
    response: dict,
    *,
    duration: float,
    time_step: float,
    phase_deg: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build real wave and six-DOF harmonic time series."""
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    if time_step <= 0.0:
        raise ValueError("time step must be positive.")

    time = np.arange(0.0, duration + 0.5 * time_step, time_step)
    phase = np.deg2rad(phase_deg)
    omega = response["omega"]
    wave_amplitude = response["wave_amplitude_m"]
    complex_response = response["response_amplitudes"]
    wave = wave_amplitude * np.cos(omega * time + phase)

    values = {"wave_elevation_m": wave}
    for dof, amplitude in zip(DOFS, complex_response):
        # Multiplication by exp(i omega t) preserves the RAO phase. The
        # factors i*omega and -omega**2 are the frequency-domain derivatives.
        harmonic = amplitude * np.exp(1j * (omega * time + phase))
        values[f"{dof}_displacement"] = np.real(harmonic)
        values[f"{dof}_velocity"] = np.real(1j * omega * harmonic)
        values[f"{dof}_acceleration"] = np.real(-(omega**2) * harmonic)
    return time, values


def save_csv(
    path: Path,
    time: np.ndarray,
    values: dict[str, np.ndarray],
) -> None:
    """Write the time series to CSV."""
    fields = ["time_s", *values]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(
            [time[index], *(values[field][index] for field in values)]
            for index in range(len(time))
        )


def save_plot(
    path: Path,
    time: np.ndarray,
    values: dict[str, np.ndarray],
) -> None:
    """Save wave elevation and six-DOF displacement plots."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)

    axes[0, 0].plot(time, values["wave_elevation_m"], color="black")
    axes[0, 0].set_ylabel("Wave [m]")
    axes[0, 0].set_title("Wave elevation")
    axes[0, 1].axis("off")

    for axis, dof in zip(axes.flat[2:], DOFS):
        axis.plot(time, values[f"{dof}_displacement"])
        axis.set_ylabel(dof)
        axis.grid(True, alpha=0.3)

    for axis in axes[-1, :]:
        axis.set_xlabel("Time [s]")
    figure.suptitle("Dirac-limit regular-wave response", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_spectrum_csv(
    path: Path,
    model: raft.Model,
    *,
    fowt_index: int,
    selected_frequency_index: int,
) -> None:
    """Write RAFT's initial discrete JONSWAP spectrum to CSV."""
    fowt = model.fowtList[fowt_index]
    omega = np.asarray(model.results["freq_rad"], dtype=float)
    frequency_hz = omega / (2.0 * np.pi)
    spectrum = np.asarray(fowt.S[0], dtype=float)
    wave_amplitude = np.sqrt(2.0 * spectrum * fowt.dw)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frequency_index",
        "frequency_rad_s",
        "frequency_hz",
        "period_s",
        "jonswap_psd_m2_per_rad_s",
        "wave_component_amplitude_m",
        "selected_frequency",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index, (omega_value, frequency_value, spectrum_value) in enumerate(
            zip(omega, frequency_hz, spectrum)
        ):
            writer.writerow(
                [
                    index,
                    omega_value,
                    frequency_value,
                    1.0 / frequency_value if frequency_value > 0.0 else "",
                    spectrum_value,
                    wave_amplitude[index],
                    index == selected_frequency_index,
                ]
            )


def save_spectrum_plot(
    path: Path,
    model: raft.Model,
    *,
    fowt_index: int,
    selected_frequency_index: int,
) -> None:
    """Plot RAFT's initial discrete JONSWAP spectrum."""
    import matplotlib.pyplot as plt

    fowt = model.fowtList[fowt_index]
    omega = np.asarray(model.results["freq_rad"], dtype=float)
    frequency_hz = omega / (2.0 * np.pi)
    spectrum = np.asarray(fowt.S[0], dtype=float)
    selected_frequency = frequency_hz[selected_frequency_index]

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(frequency_hz, spectrum, color="tab:blue", label="JONSWAP spectrum")
    axis.axvline(
        selected_frequency,
        color="tab:red",
        linestyle="--",
        label=f"Selected frequency = {selected_frequency:.4f} Hz",
    )
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(r"$S_\eta$ [m$^2$/(rad/s)]")
    axis.set_title("Initial JONSWAP wave spectrum used by RAFT")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    *,
    design_path: Path = DEFAULT_DESIGN,
    period: float = TP_S,
    height: float = HS_M,
    heading: float = 0.0,
    wind_speed: float = 0.0,
    turbulence: float = 0.0,
    gamma: float = GAMMA,
    fowt_index: int = 0,
    duration: float | None = None,
    time_step: float = TIME_STEP_S,
    phase_deg: float = 0.0,
    display: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict, raft.Model, int]:
    """Run RAFT and return the deterministic regular-wave time series."""
    design = load_design(design_path)
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
    response = extract_regular_response(
        model, period=period, height=height, fowt_index=fowt_index
    )
    selected_frequency_index, _ = select_frequency(model, period)
    if duration is None:
        duration = 10.0 * period
    time, values = make_time_series(
        response,
        duration=duration,
        time_step=time_step,
        phase_deg=phase_deg,
    )
    return time, values, response, model, selected_frequency_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Dirac-limit regular-wave RAFT time series."
    )
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--tp", "--period", dest="period", type=float, default=TP_S)
    parser.add_argument("--hs", "--height", dest="height", type=float, default=HS_M)
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--wind-speed", type=float, default=0.0)
    parser.add_argument("--turbulence", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--fowt-index", type=int, default=0)
    parser.add_argument("--cycles", type=float, default=NUMBER_OF_CYCLES)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--dt", "--time-step", dest="time_step", type=float, default=TIME_STEP_S
    )
    parser.add_argument("--phase-deg", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--spectrum-output", type=Path, default=DEFAULT_SPECTRUM_OUTPUT)
    parser.add_argument("--spectrum-plot", type=Path, default=DEFAULT_SPECTRUM_PLOT)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    argv = sys.argv[1:]
    if "ipykernel" in sys.modules:
        argv = [argument for argument in argv if not argument.endswith(".json")]
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    duration = args.duration if args.duration is not None else args.cycles * args.period
    time, values, response, model, selected_frequency_index = run(
        design_path=args.design,
        period=args.period,
        height=args.height,
        heading=args.heading,
        wind_speed=args.wind_speed,
        turbulence=args.turbulence,
        gamma=args.gamma,
        fowt_index=args.fowt_index,
        duration=duration,
        time_step=args.time_step,
        phase_deg=args.phase_deg,
        display=0 if args.quiet else 1,
    )
    save_csv(args.output, time, values)
    save_spectrum_csv(
        args.spectrum_output,
        model,
        fowt_index=args.fowt_index,
        selected_frequency_index=selected_frequency_index,
    )
    if not args.no_plot:
        save_plot(args.plot_output, time, values)
        save_spectrum_plot(
            args.spectrum_plot,
            model,
            fowt_index=args.fowt_index,
            selected_frequency_index=selected_frequency_index,
        )
    print(
        f"Selected frequency {response['frequency_hz']:.6f} Hz "
        f"(period {response['period_s']:.6f} s)"
    )
    print(f"Regular-wave height: {2.0 * response['wave_amplitude_m']:.6f} m")
    for dof, amplitude in zip(DOFS, response["response_amplitudes"]):
        print(f"{dof:>6}: amplitude={abs(amplitude):.6e}")
    print(f"Saved {len(time)} samples to {args.output}")
    print(f"Saved JONSWAP spectrum to {args.spectrum_output}")
    if not args.no_plot:
        print(f"Saved plot to {args.plot_output}")
        print(f"Saved JONSWAP spectrum plot to {args.spectrum_plot}")


if __name__ == "__main__":
    main()
