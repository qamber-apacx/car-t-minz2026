"""
Baseline 0: single-antigen qualification model

Goal:
    Before comparing 2s / dCAR / 2v, qualify a single-antigen CAR-T model
    against README / slide-derived summary anchors.

This is an evidence-constrained hybrid stochastic ABM:
    - Counts are simulated as populations, not individual 10^12 cells.
    - Tumour has A-high, A-low, A-negative phenotypes.
    - Effector CAR-T expands, peaks, contracts.
    - Tumour killing depends on antigen density and a killing threshold.
    - ABC / rejection calibration keeps only parameter sets passing validation gates.

Dependencies:
    numpy
    pandas
    matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. Run configuration
# ============================================================

# Stage-1 ABC settings. Change these if you want a larger or smaller run.
N_PARAMETER_SETS = 2000
COHORT_SIZE = 50
RANDOM_SEED = 2026

# Output folder for the 2000-sample run.
BASELINE0_OUTPUT_DIR = "baseline0_outputs_2000"

# When retained parameter sets are available, plot curves for up to this many.
MAX_EXAMPLE_RETAINED_SETS = 5


# ============================================================
# 1. Data structures
# ============================================================

@dataclass(frozen=True)
class ModelParams:
    """
    Parameter set theta.

    Important:
        These are not claimed to be patient-level fitted parameters.
        They are priors / calibrated parameters constrained by summary anchors.
    """

    # Infused CAR-T dose scale
    E0: float

    # Effector expansion fitness
    effector_doubling_time_days: float

    # Effector death / contraction
    base_death_rate_per_day: float
    aicd_max_rate_per_day: float
    aicd_mid_day: float
    aicd_width_days: float

    # Stimulation saturation
    stim_half_saturation_cells: float

    # Killing mechanism
    max_kills_per_effector_per_day: float
    kill_threshold_molecules: float
    kill_steepness: float

    # Tumour growth
    tumour_growth_rate_per_day: float

    # Stochasticity
    effector_noise_sigma: float
    kill_noise_sigma: float

    # Simulation settings
    t_end_days: float = 60.0
    dt_days: float = 0.25
    clearance_threshold_cells: float = 1e6


@dataclass(frozen=True)
class PatientParams:
    """
    Virtual patient / tumour configuration.

    Tumour phenotypes:
        A_high:    antigen A clearly above threshold
        A_low:     antigen A around / below threshold
        A_neg:     antigen A absent
    """

    T0: float
    frac_A_high: float
    frac_A_low: float
    frac_A_neg: float

    antigen_A_high: float
    antigen_A_low: float
    antigen_A_neg: float

    E0_multiplier: float


@dataclass
class SimulationResult:
    times: np.ndarray
    E: np.ndarray
    T_high: np.ndarray
    T_low: np.ndarray
    T_neg: np.ndarray
    params: ModelParams
    patient: PatientParams


# ============================================================
# 2. Utility functions
# ============================================================

def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample from a log10-uniform distribution."""
    return 10 ** rng.uniform(np.log10(low), np.log10(high))


def safe_sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def recognition_probability(
    antigen_density: float,
    threshold: float,
    steepness: float,
) -> float:
    """
    CAR-antigen recognition / killing transition.

    phi(alpha) = 1 / (1 + exp[-h(log10(alpha) - log10(theta))])

    Around theta ~ 1e3 molecules/cell, killing transitions from weak to strong.
    """

    if antigen_density <= 0:
        return 0.0

    x = np.log10(antigen_density) - np.log10(threshold)
    return safe_sigmoid(steepness * x)


def noisy_positive_value(
    rng: np.random.Generator,
    expected: float,
    sigma: float,
) -> float:
    """
    Multiplicative lognormal noise around an expected positive value.

    For huge cell counts, direct Poisson / binomial simulation can be impractical.
    This keeps stochasticity while staying computationally light.
    """

    if expected <= 0:
        return 0.0

    if sigma <= 0:
        return float(expected)

    noise = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma)
    return float(max(0.0, expected * noise))


def clip_count(x: float, lower: float = 0.0, upper: float = 1e15) -> float:
    """Keep population counts in a reasonable numerical range."""
    return float(np.clip(x, lower, upper))


# ============================================================
# 3. Prior sampling
# ============================================================

def sample_model_params(rng: np.random.Generator) -> ModelParams:
    """
    Sample theta from evidence-constrained priors.

    These are deliberately broad. ABC rejection will later remove parameter sets
    that cannot reproduce the summary-level CAR-T anchors.
    """

    return ModelParams(
        # README / slides anchor: dose around 1e8 cells
        E0=log_uniform(rng, 5e7, 2e8),

        # Positive vs null response doubling time scale:
        # around 1.6 d vs 2.1 d, but we keep a broad range.
        effector_doubling_time_days=rng.uniform(1.3, 2.8),

        # Basal death and AICD / contraction
        base_death_rate_per_day=rng.uniform(0.01, 0.06),
        aicd_max_rate_per_day=rng.uniform(0.20, 0.70),
        aicd_mid_day=rng.uniform(8.0, 15.0),
        aicd_width_days=rng.uniform(1.5, 5.0),

        # Stimulation saturation
        stim_half_saturation_cells=log_uniform(rng, 1e6, 1e9),

        # Killing mechanism
        max_kills_per_effector_per_day=log_uniform(rng, 0.05, 5.0),
        kill_threshold_molecules=log_uniform(rng, 1e3, 3e3),
        kill_steepness=rng.uniform(2.0, 8.0),

        # Tumour growth is slow relative to acute CAR-T killing
        tumour_growth_rate_per_day=rng.uniform(0.0, 0.035),

        # Stochasticity
        effector_noise_sigma=rng.uniform(0.01, 0.08),
        kill_noise_sigma=rng.uniform(0.05, 0.25),
    )


def sample_patient(rng: np.random.Generator) -> PatientParams:
    """
    Sample virtual patient / tumour configuration.

    T0 covers 1e8-1e12 cells.
    Antigen-negative fraction is deliberately variable because escape is one of
    the validation targets.
    """

    # Tumour burden distribution:
    # More patients in low/moderate range, fewer in very high range.
    if rng.random() < 0.70:
        T0 = log_uniform(rng, 1e8, 1e10)
    else:
        T0 = log_uniform(rng, 1e10, 1e12)

    # Pre-existing antigen-negative reservoir.
    # This is not an observed patient-level distribution; it is a scenario prior.
    frac_A_neg = log_uniform(rng, 1e-6, 5e-2)

    # Low-antigen population around / below threshold.
    frac_A_low = rng.uniform(0.03, 0.35)

    # Avoid impossible fractions.
    if frac_A_neg + frac_A_low > 0.80:
        frac_A_low = 0.80 - frac_A_neg

    frac_A_high = 1.0 - frac_A_low - frac_A_neg

    # Antigen densities.
    antigen_A_high = log_uniform(rng, 1e4, 1e5)
    antigen_A_low = log_uniform(rng, 2e2, 3e3)
    antigen_A_neg = 0.0

    # Product / dosing variability around E0.
    E0_multiplier = rng.lognormal(mean=-0.5 * 0.20**2, sigma=0.20)

    return PatientParams(
        T0=T0,
        frac_A_high=frac_A_high,
        frac_A_low=frac_A_low,
        frac_A_neg=frac_A_neg,
        antigen_A_high=antigen_A_high,
        antigen_A_low=antigen_A_low,
        antigen_A_neg=antigen_A_neg,
        E0_multiplier=E0_multiplier,
    )


# ============================================================
# 4. Single-patient simulation
# ============================================================

def simulate_single_patient(
    params: ModelParams,
    patient: PatientParams,
    rng: np.random.Generator,
) -> SimulationResult:
    """
    Simulate one single-antigen CAR-T treatment course.

    Mechanistic simplification:
        - Effector expansion is stimulation-dependent.
        - Contraction is represented by increasing AICD / exhaustion pressure.
        - Tumour killing depends on contact capacity and antigen recognition phi.
    """

    dt = params.dt_days
    times = np.arange(0.0, params.t_end_days + dt, dt)
    n = len(times)

    E = np.zeros(n)
    T_high = np.zeros(n)
    T_low = np.zeros(n)
    T_neg = np.zeros(n)

    E[0] = params.E0 * patient.E0_multiplier
    T_high[0] = patient.T0 * patient.frac_A_high
    T_low[0] = patient.T0 * patient.frac_A_low
    T_neg[0] = patient.T0 * patient.frac_A_neg

    phi_high = recognition_probability(
        patient.antigen_A_high,
        params.kill_threshold_molecules,
        params.kill_steepness,
    )
    phi_low = recognition_probability(
        patient.antigen_A_low,
        params.kill_threshold_molecules,
        params.kill_steepness,
    )
    phi_neg = 0.0

    growth_rate_effector = math.log(2.0) / params.effector_doubling_time_days

    for k in range(1, n):
        t = times[k - 1]

        e_prev = max(E[k - 1], 0.0)
        th = max(T_high[k - 1], 0.0)
        tl = max(T_low[k - 1], 0.0)
        tn = max(T_neg[k - 1], 0.0)

        # Tumour intrinsic growth before CAR-T killing.
        tumour_growth_factor = math.exp(params.tumour_growth_rate_per_day * dt)
        th *= tumour_growth_factor
        tl *= tumour_growth_factor
        tn *= tumour_growth_factor

        total_T = th + tl + tn

        if total_T <= 1.0 or e_prev <= 1.0:
            # No meaningful tumour or no effector left.
            E[k] = clip_count(e_prev)
            T_high[k] = clip_count(th)
            T_low[k] = clip_count(tl)
            T_neg[k] = clip_count(tn)
            continue

        # ----------------------------------------------------
        # Contact / kill stage
        # ----------------------------------------------------
        # Total kill/contact capacity in this time step.
        kill_capacity = params.max_kills_per_effector_per_day * e_prev * dt

        # Random contacts are distributed according to tumour composition.
        expected_kill_high = kill_capacity * (th / total_T) * phi_high
        expected_kill_low = kill_capacity * (tl / total_T) * phi_low
        expected_kill_neg = kill_capacity * (tn / total_T) * phi_neg

        kill_high = min(
            th,
            noisy_positive_value(
                rng,
                expected_kill_high,
                params.kill_noise_sigma,
            ),
        )
        kill_low = min(
            tl,
            noisy_positive_value(
                rng,
                expected_kill_low,
                params.kill_noise_sigma,
            ),
        )
        kill_neg = min(
            tn,
            noisy_positive_value(
                rng,
                expected_kill_neg,
                params.kill_noise_sigma,
            ),
        )

        th -= kill_high
        tl -= kill_low
        tn -= kill_neg

        # ----------------------------------------------------
        # Effector expansion / contraction stage
        # ----------------------------------------------------
        recognised_tumour_signal = th * phi_high + tl * phi_low
        stimulation = recognised_tumour_signal / (
            recognised_tumour_signal + params.stim_half_saturation_cells
        )

        # AICD / exhaustion pressure increases around a parameterised mid-day.
        aicd_rate = params.aicd_max_rate_per_day * safe_sigmoid(
            (t - params.aicd_mid_day) / params.aicd_width_days
        )

        net_effector_rate = (
            growth_rate_effector * stimulation
            - params.base_death_rate_per_day
            - aicd_rate
        )

        e_expected = e_prev * math.exp(net_effector_rate * dt)

        # Smaller time step, smaller noise.
        e_new = noisy_positive_value(
            rng,
            e_expected,
            params.effector_noise_sigma * math.sqrt(dt),
        )

        E[k] = clip_count(e_new)
        T_high[k] = clip_count(th)
        T_low[k] = clip_count(tl)
        T_neg[k] = clip_count(tn)

    return SimulationResult(
        times=times,
        E=E,
        T_high=T_high,
        T_low=T_low,
        T_neg=T_neg,
        params=params,
        patient=patient,
    )


# ============================================================
# 5. Response summaries
# ============================================================

def summarise_single_simulation(sim: SimulationResult) -> dict:
    """Convert one simulated trajectory into validation-ready summary statistics."""

    params = sim.params
    patient = sim.patient

    T_total = sim.T_high + sim.T_low + sim.T_neg
    E = sim.E

    peak_idx = int(np.argmax(E))
    t_peak = float(sim.times[peak_idx])
    E_peak = float(E[peak_idx])
    E_end = float(E[-1])

    T0 = float(patient.T0)
    T_end = float(T_total[-1])
    end_fraction = T_end / max(T0, 1.0)

    T_nadir = float(np.min(T_total))
    nadir_fraction = T_nadir / max(T0, 1.0)

    # Response definitions for calibration.
    # These are modelling definitions, not clinical RECIST claims.
    ORR = end_fraction <= 0.50
    CR = T_end <= params.clearance_threshold_cells

    residual_neg_fraction = (
        float(sim.T_neg[-1]) / max(T_end, 1.0)
        if T_end > 1.0
        else 0.0
    )

    escape = bool(ORR and not CR and residual_neg_fraction >= 0.50)

    # Time to clearance if achieved.
    clear_indices = np.where(T_total <= params.clearance_threshold_cells)[0]
    if len(clear_indices) > 0:
        time_to_clearance = float(sim.times[int(clear_indices[0])])
    else:
        time_to_clearance = np.nan

    # Qualitative expansion shape:
    # rise -> peak -> contraction
    shape_pass = (
        t_peak > params.dt_days
        and t_peak < params.t_end_days - params.dt_days
        and E_peak >= 1.5 * E[0]
        and E_end <= 0.80 * E_peak
    )

    ET_peak_ratio = E_peak / max(T0, 1.0)

    durable_response = bool(CR or end_fraction <= 0.01)

    return {
        "T0": T0,
        "E0": float(E[0]),
        "t_peak": t_peak,
        "E_peak": E_peak,
        "E_end": E_end,
        "T_end": T_end,
        "end_fraction": end_fraction,
        "T_nadir": T_nadir,
        "nadir_fraction": nadir_fraction,
        "ORR": ORR,
        "CR": CR,
        "escape": escape,
        "residual_neg_fraction": residual_neg_fraction,
        "time_to_clearance": time_to_clearance,
        "shape_pass": shape_pass,
        "ET_peak_ratio": ET_peak_ratio,
        "durable_response": durable_response,
    }


def simulate_cohort(
    params: ModelParams,
    cohort_size: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[SimulationResult]]:
    """Simulate a virtual cohort under a single parameter set theta."""

    rows = []
    sims = []

    for patient_id in range(cohort_size):
        patient = sample_patient(rng)
        sim = simulate_single_patient(params, patient, rng)
        row = summarise_single_simulation(sim)
        row["patient_id"] = patient_id

        rows.append(row)
        sims.append(sim)

    return pd.DataFrame(rows), sims


# ============================================================
# 6. Validation gates
# ============================================================

def cohort_summary(cohort_df: pd.DataFrame) -> dict:
    """Compute summary-level calibration statistics for one parameter set."""

    if len(cohort_df) == 0:
        raise ValueError("cohort_df is empty")

    ORR_rate = float(cohort_df["ORR"].mean())
    CR_rate = float(cohort_df["CR"].mean())

    if cohort_df["ORR"].sum() > 0:
        escape_rate_among_ORR = float(
            cohort_df.loc[cohort_df["ORR"], "escape"].mean()
        )
    else:
        escape_rate_among_ORR = 0.0

    # E:T gradient:
    # top quartile E:T should have better durable response than bottom quartile.
    q25 = cohort_df["ET_peak_ratio"].quantile(0.25)
    q75 = cohort_df["ET_peak_ratio"].quantile(0.75)

    low_group = cohort_df[cohort_df["ET_peak_ratio"] <= q25]
    high_group = cohort_df[cohort_df["ET_peak_ratio"] >= q75]

    low_durable = float(low_group["durable_response"].mean()) if len(low_group) else 0.0
    high_durable = float(high_group["durable_response"].mean()) if len(high_group) else 0.0
    ET_gradient = high_durable - low_durable

    return {
        "t_peak_median": float(cohort_df["t_peak"].median()),
        "E_peak_median": float(cohort_df["E_peak"].median()),
        "E_peak_log10_median": float(np.log10(cohort_df["E_peak"].median())),
        "shape_pass_rate": float(cohort_df["shape_pass"].mean()),
        "ORR_rate": ORR_rate,
        "CR_rate": CR_rate,
        "escape_rate_among_ORR": escape_rate_among_ORR,
        "low_ET_durable_rate": low_durable,
        "high_ET_durable_rate": high_durable,
        "ET_gradient": ET_gradient,
    }


def interval_penalty(value: float, low: float, high: float, scale: float) -> float:
    """Distance penalty for being outside a target interval."""
    if low <= value <= high:
        return 0.0
    if value < low:
        return (low - value) / scale
    return (value - high) / scale


def validation_gates(summary: dict) -> dict:
    """
    Hard validation gates.

    These gates operationalise the README / slide anchors.
    Parameter sets that fail are not used for 2s/dCAR/2v comparison.
    """

    gates = {
        # Level 1 sanity validation
        "gate_shape": summary["shape_pass_rate"] >= 0.80,
        "gate_t_peak": 7.0 <= summary["t_peak_median"] <= 20.0,
        "gate_E_peak": 1e9 <= summary["E_peak_median"] <= 1e10,

        # Level 2 summary-statistics calibration
        "gate_ORR": 0.70 <= summary["ORR_rate"] <= 0.90,
        "gate_CR": 0.50 <= summary["CR_rate"] <= 0.80,
        "gate_escape": 0.10 <= summary["escape_rate_among_ORR"] <= 0.40,

        # E:T relation
        "gate_ET_gradient": summary["ET_gradient"] >= 0.15,
    }

    gates["retained"] = all(gates.values())
    return gates


def calibration_distance(summary: dict) -> float:
    """
    Soft distance for ranking parameter sets.

    Even if no parameter set passes all hard gates in a short run, this helps us
    inspect the closest candidates.
    """

    d = 0.0

    d += interval_penalty(summary["t_peak_median"], 7.0, 20.0, scale=10.0)
    d += interval_penalty(summary["E_peak_log10_median"], 9.0, 10.0, scale=1.0)
    d += interval_penalty(summary["shape_pass_rate"], 0.80, 1.00, scale=0.20)

    d += interval_penalty(summary["ORR_rate"], 0.70, 0.90, scale=0.20)
    d += interval_penalty(summary["CR_rate"], 0.50, 0.80, scale=0.30)
    d += interval_penalty(summary["escape_rate_among_ORR"], 0.10, 0.40, scale=0.30)

    d += interval_penalty(summary["ET_gradient"], 0.15, 1.00, scale=0.15)

    return float(d)


# ============================================================
# 7. ABC / rejection calibration
# ============================================================

def abc_rejection(
    n_parameter_sets: int = 2000,
    cohort_size: int = 50,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main ABC / rejection calibration loop.

    Returns:
        params_df:
            One row per sampled parameter set.
        summary_df:
            Validation summaries and gate outcomes.
    """

    rng = np.random.default_rng(seed)

    param_rows = []
    summary_rows = []

    for param_id in range(n_parameter_sets):
        params = sample_model_params(rng)
        cohort_df, _ = simulate_cohort(params, cohort_size, rng)

        summary = cohort_summary(cohort_df)
        gates = validation_gates(summary)
        distance = calibration_distance(summary)

        param_row = asdict(params)
        param_row["param_id"] = param_id

        summary_row = {
            "param_id": param_id,
            **summary,
            **gates,
            "distance": distance,
        }

        param_rows.append(param_row)
        summary_rows.append(summary_row)

        if (param_id + 1) % 25 == 0:
            retained_so_far = sum(row["retained"] for row in summary_rows)
            print(
                f"Sampled {param_id + 1}/{n_parameter_sets} parameter sets; "
                f"retained so far: {retained_so_far}"
            )

    params_df = pd.DataFrame(param_rows)
    summary_df = pd.DataFrame(summary_rows)

    return params_df, summary_df


# ============================================================
# 8. Plotting
# ============================================================

def params_from_row(row: pd.Series) -> ModelParams:
    """Reconstruct ModelParams from a params_df row."""
    valid_names = {f.name for f in fields(ModelParams)}
    kwargs = {name: row[name] for name in valid_names}
    return ModelParams(**kwargs)


def plot_gate_pass_rates(summary_df: pd.DataFrame, outdir: Path) -> None:
    """Plot validation gate pass rates across sampled parameter sets."""

    gate_cols = [
        "gate_shape",
        "gate_t_peak",
        "gate_E_peak",
        "gate_ORR",
        "gate_CR",
        "gate_escape",
        "gate_ET_gradient",
        "retained",
    ]

    rates = summary_df[gate_cols].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(rates.index, rates.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Pass rate across sampled parameter sets")
    plt.title("Validation gate pass rates")
    plt.tight_layout()
    plt.savefig(outdir / "validation_gate_pass_rates.png", dpi=200)
    plt.close()


def plot_example_curves(
    params: ModelParams,
    outdir: Path,
    seed: int = 999,
    n_examples: int = 8,
    filename_prefix: str = "example",
) -> None:
    """Plot example E(t) and T(t) curves for one selected parameter set.

    filename_prefix lets us save separate curve panels for multiple retained
    parameter sets, instead of overwriting the same two PNG files.
    """

    rng = np.random.default_rng(seed)

    sims = []
    for _ in range(n_examples):
        patient = sample_patient(rng)
        sim = simulate_single_patient(params, patient, rng)
        sims.append(sim)

    # Effector curves
    plt.figure(figsize=(8, 5))
    for sim in sims:
        plt.plot(sim.times, sim.E, alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Days after infusion")
    plt.ylabel("Effector CAR-T cells E(t)")
    plt.title("Example effector expansion: rise → peak → contraction")
    plt.tight_layout()
    plt.savefig(outdir / f"{filename_prefix}_effector_curves.png", dpi=200)
    plt.close()

    # Tumour curves
    plt.figure(figsize=(8, 5))
    for sim in sims:
        T_total = sim.T_high + sim.T_low + sim.T_neg
        plt.plot(sim.times, T_total, alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Days after infusion")
    plt.ylabel("Total tumour cells T(t)")
    plt.title("Example tumour burden trajectories")
    plt.tight_layout()
    plt.savefig(outdir / f"{filename_prefix}_tumour_curves.png", dpi=200)
    plt.close()


def plot_retained_parameter_distributions(
    params_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outdir: Path,
) -> None:
    """Plot a few retained parameter distributions."""

    retained_ids = summary_df.loc[summary_df["retained"], "param_id"]

    if len(retained_ids) == 0:
        print("No retained parameter sets; skipping retained parameter plots.")
        return

    retained_params = params_df[params_df["param_id"].isin(retained_ids)]

    variables = [
        "effector_doubling_time_days",
        "max_kills_per_effector_per_day",
        "kill_threshold_molecules",
        "aicd_max_rate_per_day",
    ]

    for var in variables:
        plt.figure(figsize=(7, 4))
        plt.hist(retained_params[var], bins=20)
        plt.xlabel(var)
        plt.ylabel("Count")
        plt.title(f"Retained distribution: {var}")
        plt.tight_layout()
        plt.savefig(outdir / f"retained_distribution_{var}.png", dpi=200)
        plt.close()


# ============================================================
# 9. Main run
# ============================================================

def main() -> None:
    outdir = Path(BASELINE0_OUTPUT_DIR)
    outdir.mkdir(exist_ok=True)

    print("Running Baseline 0 ABC / rejection calibration...")
    print(f"n_parameter_sets = {N_PARAMETER_SETS}")
    print(f"cohort_size      = {COHORT_SIZE}")
    print(f"random_seed      = {RANDOM_SEED}")

    params_df, summary_df = abc_rejection(
        n_parameter_sets=N_PARAMETER_SETS,
        cohort_size=COHORT_SIZE,
        seed=RANDOM_SEED,
    )

    gate_cols = [
        "gate_shape",
        "gate_t_peak",
        "gate_E_peak",
        "gate_ORR",
        "gate_CR",
        "gate_escape",
        "gate_ET_gradient",
    ]

    # Add a useful diagnostic column: how many gates each parameter set passed.
    summary_df["n_gates_pass"] = summary_df[gate_cols].sum(axis=1)

    retained_df = summary_df[summary_df["retained"]].copy()
    retained_param_ids = retained_df["param_id"].astype(int).tolist()
    retained_params_df = params_df[params_df["param_id"].isin(retained_param_ids)].copy()

    near_miss_df = summary_df[
        (summary_df["n_gates_pass"] == len(gate_cols) - 1)
        & (~summary_df["retained"])
    ].copy()

    # Save main tables.
    params_df.to_csv(outdir / "sampled_parameter_sets.csv", index=False)
    summary_df.to_csv(outdir / "validation_summary.csv", index=False)
    retained_df.to_csv(outdir / "retained_summary.csv", index=False)
    retained_params_df.to_csv(outdir / "retained_parameter_sets.csv", index=False)
    near_miss_df.to_csv(outdir / "near_miss_summary.csv", index=False)

    gate_pass_rates = summary_df[gate_cols + ["retained"]].mean().sort_values(ascending=False)
    gate_pass_rates.rename("pass_rate").to_csv(outdir / "gate_pass_rates.csv")

    top10 = summary_df.sort_values("distance").head(10).copy()
    top10.to_csv(outdir / "top10_closest_parameter_sets.csv", index=False)

    n_retained = len(retained_df)
    n_sampled = len(summary_df)

    print("\n==============================")
    print("ABC rejection calibration done")
    print("==============================")
    print(f"Sampled parameter sets:  {n_sampled}")
    print(f"Retained parameter sets: {n_retained}")
    print(f"Retention rate:          {n_retained / max(n_sampled, 1):.4f}")

    if n_retained > 0:
        print("\nRetained param_id values:")
        print(retained_param_ids)
    else:
        print("\nNo fully retained parameter sets found in this run.")

    print("\nGate pass rates:")
    print(gate_pass_rates)

    print(f"\nNear-miss parameter sets passing {len(gate_cols)-1}/{len(gate_cols)} gates: {len(near_miss_df)}")
    if len(near_miss_df) > 0:
        failed_gate_counts = {
            gate: int((near_miss_df[gate] == False).sum())
            for gate in gate_cols
        }
        print("Failed gate among near-misses:")
        for gate, count in sorted(failed_gate_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {gate}: {count}")

    print("\nTop 10 closest parameter sets by calibration distance:")
    print(
        top10[
            [
                "param_id",
                "distance",
                "t_peak_median",
                "E_peak_median",
                "ORR_rate",
                "CR_rate",
                "escape_rate_among_ORR",
                "ET_gradient",
                "retained",
                "n_gates_pass",
            ]
        ]
    )

    # Plots
    plot_gate_pass_rates(summary_df, outdir)
    plot_retained_parameter_distributions(params_df, summary_df, outdir)

    # Plot example curves for multiple retained sets when available.
    # With 2000 samples, this usually gives around 4-5 retained sets.
    if n_retained > 0:
        selected_param_ids = retained_param_ids[:MAX_EXAMPLE_RETAINED_SETS]
        print(
            f"\nPlotting example curves for up to {MAX_EXAMPLE_RETAINED_SETS} retained parameter sets: "
            f"{selected_param_ids}"
        )
    else:
        selected_param_ids = [int(summary_df.sort_values("distance").iloc[0]["param_id"])]
        print(
            "\nNo fully retained parameter set found in this run. "
            f"Plotting closest candidate param_id={selected_param_ids[0]}"
        )

    for offset, selected_param_id in enumerate(selected_param_ids):
        selected_row = params_df[params_df["param_id"] == selected_param_id].iloc[0]
        selected_params = params_from_row(selected_row)
        prefix = f"retained_param_{selected_param_id}" if n_retained > 0 else f"closest_param_{selected_param_id}"
        plot_example_curves(
            selected_params,
            outdir,
            seed=999 + offset,
            filename_prefix=prefix,
        )

    print(f"\nOutputs written to: {outdir.resolve()}")


# This guard lets another notebook load the functions with:
#     RUN_BASELINE0_MAIN = False
#     %run -i "./baseline0-single-antigen.ipynb"
RUN_BASELINE0_MAIN = globals().get("RUN_BASELINE0_MAIN", True)

if __name__ == "__main__" and RUN_BASELINE0_MAIN:
    main()
