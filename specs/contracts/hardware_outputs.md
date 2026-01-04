# Hardware Outputs Taxonomy

## Purpose

This document defines a strict taxonomy for all output fields in `reports/latest.json` and related JSON artifacts. The goal is to prevent "hardware outputs" from drifting into implied measurement claims—explicitly distinguishing what is **simulated**, **inferred**, **measured**, or a **placeholder**.

---

## Classification Definitions

| Category       | Definition                                                                                                                                           |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Simulated**  | Value produced by Monte Carlo or analytic model with no direct hardware input. The physics model may be simplified or pedagogical.                 |
| **Inferred**   | Value derived from simulated outputs via post-processing (e.g., secret fraction from QBER).                                                        |
| **Measured**   | Value that would come from real hardware in a deployed system. **Not currently present**—this simulator has no hardware interface.                  |
| **Placeholder**| Hardcoded or default value awaiting future implementation or calibration data.                                                                      |

---

## Schema Versioning

The current `schema_version` in `reports/latest.json` is **0.4**.

- Version 0.4: Pass-sweep output with per-step records and summary statistics.
- Version 1.0: Experiment-run output with block-level metrics (used in `latest_experiment.json`, `latest_pass.json`).

Schema version should remain **0.4** for the primary `reports/latest.json` unless explicit breaking changes are introduced. Version increments require changelog documentation.

---

## Output Field Classification: `reports/latest.json`

### Top-Level Fields

| Field                | Category      | Units       | Valid Range        | Notes                                                    |
|----------------------|---------------|-------------|--------------------|----------------------------------------------------------|
| `schema_version`     | Placeholder   | string      | "0.4"              | Version identifier for schema compatibility.             |
| `generated_utc`      | Inferred      | ISO 8601    | Valid timestamp    | Timestamp when JSON was generated.                       |
| `field_classification` | Placeholder | object     | tags per field     | Mapping of field paths to classification tags.           |

### `pass_sweep.records[]` (per time-step)

| Field                | Category      | Units         | Valid Range          | Notes                                                                                      |
|----------------------|---------------|---------------|----------------------|--------------------------------------------------------------------------------------------|
| `time_s`             | Simulated     | seconds       | [0, pass_duration]   | Time since pass start.                                                                     |
| `elevation_deg`      | Simulated     | degrees       | [min_elev, max_elev] | Satellite elevation angle from synthetic pass geometry.                                    |
| `loss_db`            | Simulated     | dB            | [0, ~50]             | Total channel loss from free-space link model. **Not validated against real satellite.**   |
| `p_bg_effective`     | Simulated     | probability   | [0, 1]               | Effective background probability (day/night adjusted).                                     |
| `flip_prob`          | Placeholder   | probability   | [0, 0.5]             | Intrinsic bit-flip probability (user input, not measured).                                 |
| `attack`             | Placeholder   | string        | enum                 | Attack scenario label ("none", "intercept_resend", etc.).                                  |
| `n_sent`             | Simulated     | count         | [0, ∞)               | Number of pulses sent in time step.                                                        |
| `n_received`         | Simulated     | count         | [0, n_sent]          | Number of detection events (signal + background).                                          |
| `n_sifted`           | Simulated     | count         | [0, n_received]      | Number of sifted bits after basis reconciliation.                                          |
| `qber`               | Inferred      | unitless      | [0, 0.5] or NaN      | Quantum bit error rate from sample comparison. NaN if no sifted bits.                      |
| `secret_fraction`    | Inferred      | unitless      | [0, 1]               | Asymptotic secret fraction from `1 - f*h(Q) - h(Q)`.                                       |
| `key_rate_per_pulse` | Inferred      | bits/pulse    | [0, 1]               | Secret key rate normalized per sent pulse.                                                 |
| `n_secret_est`       | Inferred      | count         | [0, n_sifted]        | Estimated extractable secret bits.                                                         |
| `aborted`            | Inferred      | boolean       | true/false           | Whether QBER exceeded abort threshold.                                                     |

#### `pass_sweep.records[].loss_components` (optional, when atmosphere model is used)

| Field                    | Category      | Units         | Valid Range          | Notes                                                                                      |
|--------------------------|---------------|---------------|----------------------|--------------------------------------------------------------------------------------------|
| `system_loss_db`         | Simulated     | dB            | [1, 10]              | System-level losses (optics, coupling, etc.). Copied from link parameters.                |
| `atmosphere_loss_db`     | Simulated     | dB            | [0, ∞)               | Atmospheric attenuation from scenario model. Zero if `--atmosphere-model none`.           |

**Validation Rule:** `abs(loss_db - sum(loss_components.*)) <= float_eps`

**Note:** The `loss_components` structure is added when using `--atmosphere-model` to provide explicit bookkeeping of loss contributions. The total `loss_db` includes geometric + pointing + atmospheric extinction + system + atmosphere scenario loss.

### `pass_sweep.summary`

| Field                       | Category | Units         | Valid Range          | Notes                                                   |
|-----------------------------|----------|---------------|----------------------|---------------------------------------------------------|
| `secure_window_seconds`     | Inferred | seconds       | [0, pass_duration]   | Duration where key extraction is positive.              |
| `secure_start_s`            | Inferred | seconds       | [0, pass_duration]   | Start time of secure window.                            |
| `secure_end_s`              | Inferred | seconds       | [0, pass_duration]   | End time of secure window.                              |
| `secure_start_elevation_deg`| Inferred | degrees       | [min_elev, max_elev] | Elevation at secure window start.                       |
| `secure_end_elevation_deg`  | Inferred | degrees       | [min_elev, max_elev] | Elevation at secure window end.                         |
| `peak_key_rate`             | Inferred | bits/pulse    | [0, 1]               | Maximum key rate during pass.                           |
| `total_secret_bits`         | Inferred | bits          | [0, ∞)               | Cumulative secret bits over pass.                       |
| `mean_key_rate_in_window`   | Inferred | bits/pulse    | [0, 1]               | Average key rate within secure window.                  |

### `parameters` Block

| Field                | Category      | Units         | Valid Range          | Notes                                                              |
|----------------------|---------------|---------------|----------------------|--------------------------------------------------------------------|
| `max_elevation_deg`  | Placeholder   | degrees       | [0, 90]              | User-specified maximum elevation.                                  |
| `min_elevation_deg`  | Placeholder   | degrees       | [0, 90]              | User-specified minimum elevation.                                  |
| `pass_duration_s`    | Placeholder   | seconds       | [0, ∞)               | User-specified pass duration.                                      |
| `time_step_s`        | Placeholder   | seconds       | (0, ∞)               | Simulation time step.                                              |
| `flip_prob`          | Placeholder   | probability   | [0, 0.5]             | Intrinsic error probability.                                       |
| `pulses`             | Placeholder   | count         | [1, ∞)               | Total pulses for pass.                                             |
| `rep_rate`           | Placeholder   | Hz or null    | [0, ∞) or null       | Repetition rate (null if derived from pulses/duration).            |
| `eta`                | Placeholder   | unitless      | [0, 1]               | Detection efficiency parameter.                                    |
| `p_bg`               | Placeholder   | probability   | [0, 1]               | Base background probability (night).                               |
| `is_night`           | Placeholder   | boolean       | true/false           | Night-time flag.                                                   |
| `turbulence`         | Placeholder   | boolean       | true/false           | Turbulence fading enabled.                                         |
| `link_params.*`      | Placeholder   | various       | various              | Free-space link model parameters. See `free_space_link.py`.        |

---

### `cv_gg02_sweep` Block (Scaffold)

The `cv_gg02_sweep` block contains CV-QKD (GG02 protocol) scaffold outputs. This is a **structural demonstration only** and is NOT a validated security proof.

| Field                              | Category      | Units                | Valid Range          | Notes                                                              |
|------------------------------------|---------------|----------------------|----------------------|--------------------------------------------------------------------|
| `parameters.loss_min_db`           | Simulated     | dB                   | [0, ∞)               | Minimum channel loss in sweep.                                     |
| `parameters.loss_max_db`           | Simulated     | dB                   | [0, ∞)               | Maximum channel loss in sweep.                                     |
| `parameters.steps`                 | Simulated     | count                | [1, ∞)               | Number of loss values swept.                                       |
| `parameters.V_A.value`             | Simulated     | shot_noise_units     | (0, ∞)               | Alice's modulation variance.                                       |
| `parameters.V_A.units`             | —             | string               | "shot_noise_units"   | Required.                                                          |
| `parameters.V_A.classification`    | —             | string               | "simulated"          | Required.                                                          |
| `parameters.xi.value`              | Simulated     | shot_noise_units     | [0, ∞)               | Excess noise.                                                      |
| `parameters.xi.units`              | —             | string               | "shot_noise_units"   | Required.                                                          |
| `parameters.xi.classification`     | —             | string               | "simulated"          | Required.                                                          |
| `parameters.eta.value`             | Simulated     | dimensionless        | [0, 1]               | Bob's detection efficiency.                                        |
| `parameters.eta.units`             | —             | string               | "dimensionless"      | Required.                                                          |
| `parameters.eta.classification`    | —             | string               | "simulated"          | Required.                                                          |
| `parameters.v_el.value`            | Simulated     | shot_noise_units     | [0, ∞)               | Electronic noise.                                                  |
| `parameters.v_el.units`            | —             | string               | "shot_noise_units"   | Required.                                                          |
| `parameters.v_el.classification`   | —             | string               | "simulated"          | Required.                                                          |
| `parameters.beta.value`            | Simulated     | dimensionless        | [0, 1]               | Reconciliation efficiency.                                         |
| `parameters.beta.units`            | —             | string               | "dimensionless"      | Required.                                                          |
| `parameters.beta.classification`   | —             | string               | "simulated"          | Required.                                                          |

#### `cv_gg02_sweep.records[]` (per loss value)

| Field                              | Category      | Units                | Valid Range          | Notes                                                              |
|------------------------------------|---------------|----------------------|----------------------|--------------------------------------------------------------------|
| `loss_db.value`                    | Simulated     | dB                   | [0, ∞)               | Channel loss for this record.                                      |
| `loss_db.units`                    | —             | string               | "dB"                 | Required.                                                          |
| `loss_db.classification`           | —             | string               | "simulated"          | Required.                                                          |
| `transmittance.value`              | Simulated     | dimensionless        | [0, 1]               | Channel transmittance T = 10^(-loss_db/10).                        |
| `transmittance.units`              | —             | string               | "dimensionless"      | Required.                                                          |
| `transmittance.classification`     | —             | string               | "simulated"          | Required.                                                          |
| `snr.value`                        | Simulated     | dimensionless        | [0, ∞)               | Signal-to-noise ratio (linear scale).                              |
| `snr.units`                        | —             | string               | "dimensionless"      | Required.                                                          |
| `snr.classification`               | —             | string               | "simulated"          | Required.                                                          |
| `I_AB.value`                       | Simulated     | bits_per_use         | [0, ∞)               | Mutual information between Alice and Bob.                          |
| `I_AB.units`                       | —             | string               | "bits_per_use"       | Required.                                                          |
| `I_AB.classification`              | —             | string               | "simulated"          | Required.                                                          |

#### `cv_gg02_sweep.summary`

| Field                              | Category      | Units                | Valid Range          | Notes                                                              |
|------------------------------------|---------------|----------------------|----------------------|--------------------------------------------------------------------|
| `max_snr.value`                    | Simulated     | dimensionless        | [0, ∞)               | Maximum SNR across all loss values.                                |
| `max_snr.units`                    | —             | string               | "dimensionless"      | Required.                                                          |
| `max_snr.classification`           | —             | string               | "simulated"          | Required.                                                          |
| `max_I_AB.value`                   | Simulated     | bits_per_use         | [0, ∞)               | Maximum mutual information across all loss values.                 |
| `max_I_AB.units`                   | —             | string               | "bits_per_use"       | Required.                                                          |
| `max_I_AB.classification`          | —             | string               | "simulated"          | Required.                                                          |
| `status`                           | —             | string               | "toy", "stub", etc.  | Implementation status indicator.                                   |
| `note`                             | —             | string               | —                    | Human-readable note about implementation limitations.              |

**IMPORTANT**: The `cv_gg02_sweep` block is a **scaffold**. Holevo bound (chi_BE) is NOT computed. Secret key rate is NOT validated. Do NOT use for production security claims. See `docs/12_cvqkd_scaffold.md` for validation gates.

---

## What Is NOT Modeled

The following physical effects are **not currently modeled** or are modeled only in simplified form. Readers should not interpret outputs as capturing these effects:

| Effect                           | Status                                  | Notes                                                                 |
|----------------------------------|-----------------------------------------|-----------------------------------------------------------------------|
| **Pointing jitter**              | Simplified (average loss only)          | No time-resolved jitter; uses RMS pointing error for average loss.    |
| **Atmospheric turbulence**       | Simplified (lognormal fading optional)  | No Cn² profiles, beam wander, or scintillation index.                 |
| **Detector dead-time**           | Optional model exists                   | Not enabled by default; requires explicit `dead_time_pulses > 0`.     |
| **Afterpulsing**                 | Optional model exists                   | Not enabled by default; requires explicit parameters.                 |
| **Spectral filtering**           | Not modeled                             | Assumes monochromatic source; no spectral dispersion.                 |
| **Polarization mode dispersion** | Not modeled                             | Fiber/space channel PMD not simulated.                                |
| **Multiphoton attacks (PNS)**    | Toy model only                          | Uses simplified multiphoton fraction; no photon-number-resolved calc. |
| **Adaptive optics**              | Not modeled                             | No AO tip-tilt or higher-order correction.                            |
| **Real orbit mechanics**         | Synthetic profile only                  | Uses sinusoidal elevation; no TLE/SGP4 propagation.                   |
| **Finite-key composable security** | Hoeffding-based approximation         | Not a full entropic uncertainty bound (Renner-style).                 |
| **Hardware calibration drift**   | Simulated OU process                    | Not from real telemetry; synthetic drift model.                       |

---

## Red Team: How Readers Could Misinterpret These Outputs

### Risk 1: Treating `loss_db` as Measured Link Budget
**Misinterpretation:** Reader assumes `loss_db` represents a validated optical link budget.

**Reality:** The loss model uses a simplified free-space formula with placeholder atmospheric extinction. It is a **scenario generator**, not an engineering prediction.

**Mitigation:**
- Module docstrings explicitly state this is "NOT a physically accurate optical link budget."
- The field name should be read as "simulated channel loss" not "measured loss."
- Documentation (this file) classifies it as **Simulated**.

### Risk 2: Interpreting `secret_fraction` as Composable Security Guarantee
**Misinterpretation:** Reader treats `secret_fraction` as a composable security bound suitable for deployment.

**Reality:** The asymptotic formula `1 - f*h(Q) - h(Q)` is a simplified bound. Finite-key analysis uses Hoeffding bounds, not tight entropic bounds.

**Mitigation:**
- The `finite_key` block in JSON includes explicit epsilon values and bound descriptions.
- Documentation classifies `secret_fraction` as **Inferred** from a toy model.
- Future work should implement tighter bounds (see `optical_link.py` stubs).

### Risk 3: Assuming Hardware-Realistic Detector Model
**Misinterpretation:** Reader assumes detector effects (dead-time, afterpulsing, timing jitter) are comprehensively modeled.

**Reality:** These effects are **optional** and **off by default**. The default detector model only includes efficiency and background.

**Mitigation:**
- `DetectorParams` docstring lists what is and is not modeled.
- This taxonomy document explicitly lists unmodeled effects.
- Output JSON does not include fields for effects that are disabled.

### Risk 4: Extrapolating to Real Satellite Missions
**Misinterpretation:** Reader uses these curves to predict performance of a real satellite QKD mission.

**Reality:** The model is for **pedagogical demonstration** of security-curve concepts. It lacks:
- Validated atmospheric models (MODTRAN)
- Real orbit propagation (SGP4/TLE)
- Measured detector characterization
- Pointing/tracking servo dynamics

**Mitigation:**
- Module docstrings and README explicitly disclaim engineering accuracy.
- All orbit geometry is labeled as "synthetic" in parameter blocks.
- This taxonomy classifies all spatial/temporal parameters as **Placeholder**.

### Risk 5: Misreading `n_secret_est` as Guaranteed Key Bits
**Misinterpretation:** Reader believes `n_secret_est` bits can be extracted with unconditional security.

**Reality:** This is an estimate based on simulated QBER and asymptotic formulas. It does not account for:
- Finite-key penalties (partially addressed in finite-key mode)
- Composable security with explicit epsilon parameters
- Implementation imperfections

**Mitigation:**
- Field is named `n_secret_est` (estimate) not `n_secret_bits`.
- Documentation classifies it as **Inferred**.
- JSON includes `aborted` flag to indicate when no key is extractable.

---

## Recommended Practices

1. **Always check `aborted` flag** before using QBER or key rate values.
2. **Do not cite loss values as engineering predictions** without noting they are from a simplified model.
3. **Enable finite-key mode** for any security-sensitive analysis.
4. **Cross-reference this taxonomy** when interpreting JSON fields.
5. **Report schema version** in any publication using these outputs.

---

## Changelog

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 0.4     | 2026-01-04 | Initial hardware outputs taxonomy document.  |
