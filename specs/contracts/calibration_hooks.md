# Calibration Hooks

## Purpose

Calibration is a realism trap. This document makes calibration **explicit**, **versioned**, and **replayable** to prevent:
- Silent parameter drift across runs
- Unreproducible results due to undocumented tuning
- Confusion between simulation defaults and empirical calibration

---

## Calibratable vs Non-Calibratable Parameters

### Calibratable Parameters

These parameters can be adjusted based on empirical hardware data:

| Parameter | Module | Prior Range | Description |
|-----------|--------|-------------|-------------|
| `eta` | `detector.py` | [0.05, 0.95] | Overall detection efficiency |
| `eta_z` | `detector.py` | [0.05, 0.95] | Z-basis detection efficiency |
| `eta_x` | `detector.py` | [0.05, 0.95] | X-basis detection efficiency |
| `p_bg` | `detector.py` | [1e-6, 1e-2] | Background/dark click probability per pulse |
| `flip_prob` | `bb84.py` | [0.0, 0.05] | Intrinsic bit-flip probability |
| `p_afterpulse` | `detector.py` | [0.0, 0.10] | Afterpulsing probability |
| `dead_time_pulses` | `detector.py` | [0, 100] | Detector dead time in pulse windows |
| `atm_loss_db_zenith` | `free_space_link.py` | [0.1, 2.0] | Atmospheric extinction at zenith (dB) |
| `sigma_point_rad` | `free_space_link.py` | [0.5e-6, 10e-6] | RMS pointing error (radians) |
| `sigma_ln` | `free_space_link.py` | [0.0, 0.5] | Log-amplitude turbulence std dev |
| `system_loss_db` | `free_space_link.py` | [1.0, 10.0] | System losses (dB) |
| `day_background_factor` | `free_space_link.py` | [10, 1000] | Daytime background multiplier |

### Non-Calibratable Parameters

These parameters are fixed by physics, protocol definition, or simulation infrastructure:

| Parameter | Reason |
|-----------|--------|
| `wavelength_m` | Fixed by hardware (laser wavelength) |
| `tx_diameter_m` | Fixed by satellite hardware |
| `rx_diameter_m` | Fixed by ground station hardware |
| `altitude_m` | Fixed by orbital mechanics |
| `earth_radius_m` | Physical constant |
| `ec_efficiency` | Protocol parameter, not hardware |
| `qber_abort_threshold` | Protocol parameter |
| `eps_pe`, `eps_sec`, `eps_cor` | Security parameters (finite-key) |
| `seed` | Reproducibility parameter, not calibration |

---

## Calibration Procedures

### 1. Detection Efficiency (`eta`)

**Prior range:** [0.05, 0.95]

**Calibration procedure:**
1. Send known number of photons through attenuated source
2. Record detection counts at multiple attenuation levels
3. Fit linear relationship: `detections = eta * photons_incident`
4. Use y-intercept to separate `eta` from `p_bg`

**Required data:**
- Attenuator settings (dB)
- Input photon counts (calibrated source)
- Detection counts per attenuation level
- Integration time per measurement

### 2. Background Probability (`p_bg`)

**Prior range:** [1e-6, 1e-2]

**Calibration procedure:**
1. Block signal path completely
2. Record detection events over extended period
3. Compute: `p_bg = dark_counts / (rep_rate * integration_time)`

**Required data:**
- Total dark counts
- Integration time (seconds)
- Repetition rate (Hz)
- Temperature (affects dark count rate)

### 3. Intrinsic Bit-Flip (`flip_prob`)

**Prior range:** [0.0, 0.05]

**Calibration procedure:**
1. Use fiber loopback (bypass free-space channel)
2. Run BB84 with known states
3. Measure QBER in loopback: `flip_prob ≈ QBER_loopback`

**Required data:**
- Loopback QBER measurements
- Number of sifted bits per run
- Environmental conditions (temperature, vibration)

### 4. Atmospheric Extinction (`atm_loss_db_zenith`)

**Prior range:** [0.1, 2.0] dB

**Calibration procedure:**
1. Measure star magnitudes at multiple zenith angles
2. Fit Langley plot: `extinction = A * airmass + const`
3. Extract zenith extinction from y-intercept

**Required data:**
- Reference star magnitudes
- Photometric measurements at 3+ zenith angles
- Weather conditions (visibility, humidity)
- Wavelength band of measurements

### 5. Pointing Error (`sigma_point_rad`)

**Prior range:** [0.5e-6, 10e-6] radians

**Calibration procedure:**
1. Track known target (star or beacon)
2. Record pointing residuals from centroid
3. Compute RMS: `sigma = sqrt(var_x + var_y)`

**Required data:**
- Centroid position time series
- Tracking bandwidth
- Wind speed and vibration data
- Integration time per sample

---

## Calibration Record Schema

All calibration records are stored in `reports/calibration/<timestamp>.json`:

```json
{
  "calibration_version": "1.0",
  "schema_version": "0.4",
  "generated_utc": "2026-01-04T12:00:00Z",
  "git_commit": "d17561f",
  "seed_policy": "fixed",
  "seed_value": 42,

  "source": {
    "type": "empirical",
    "dataset_path": "data/2026-01-03_ground_test.csv",
    "dataset_hash_sha256": "abc123...",
    "n_records": 150,
    "date_range": ["2026-01-03T00:00:00Z", "2026-01-03T06:00:00Z"]
  },

  "fit_method": {
    "name": "grid_search_least_squares",
    "grids": {
      "eta_scale": {"min": 0.8, "max": 1.2, "n_points": 21},
      "p_bg": {"min": 1e-5, "max": 1e-3, "n_points": 20},
      "flip_prob": {"min": 0.0, "max": 0.02, "n_points": 11}
    }
  },

  "parameters": {
    "eta": {"value": 0.22, "uncertainty": 0.02, "unit": "unitless"},
    "p_bg": {"value": 1.2e-4, "uncertainty": 3e-5, "unit": "per_pulse"},
    "flip_prob": {"value": 0.003, "uncertainty": 0.001, "unit": "unitless"},
    "atm_loss_db_zenith": {"value": 0.45, "uncertainty": 0.1, "unit": "dB"},
    "sigma_point_rad": {"value": 2.5e-6, "uncertainty": 0.5e-6, "unit": "rad"}
  },

  "fit_quality": {
    "rmse": 0.0025,
    "r2": 0.94,
    "residual_std": 0.003,
    "condition_number": 42.5,
    "identifiable": true
  },

  "notes": "Ground test campaign, clear night, 15C ambient"
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `calibration_version` | string | Calibration record schema version |
| `schema_version` | string | Simulator schema version (must match) |
| `generated_utc` | string | ISO 8601 timestamp |
| `git_commit` | string | Short git commit hash |
| `seed_policy` | string | "fixed", "random", or "derived" |
| `source.type` | string | "empirical", "synthetic", or "prior" |
| `parameters` | object | Calibrated parameter values with uncertainty |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `source.dataset_path` | string | Path to source data file |
| `source.dataset_hash_sha256` | string | SHA-256 hash of source data |
| `fit_method` | object | Details of fitting procedure |
| `fit_quality` | object | Goodness-of-fit metrics |
| `notes` | string | Free-form notes |

---

## Versioning Rules

### Schema Version (`schema_version`)

- Matches the simulator's output schema version (currently `0.4`)
- Calibration record is only valid for matching schema versions
- Schema mismatch triggers warning or error at runtime

### Calibration Version (`calibration_version`)

- Starts at `1.0` for initial release
- Increments for breaking changes to calibration record format
- Minor increments (1.0 -> 1.1) for backward-compatible additions

### Git Commit Hash (`git_commit`)

- Short hash (7 characters) of simulator code at calibration time
- Allows tracing calibration to exact code version
- Recorded automatically by calibration runner

### Seed Policy

| Policy | Description |
|--------|-------------|
| `fixed` | Seed explicitly set in calibration record |
| `random` | Seed was random at calibration time (record the value) |
| `derived` | Seed derived from dataset hash or timestamp |

---

## Replay Rules

A simulation run can reference a calibration record to ensure reproducibility.

### Referencing a Calibration Record

In simulation config or CLI:

```bash
./py -m sat_qkd_lab.run --calibration reports/calibration/2026-01-04T12:00:00Z.json
```

Or in JSON config:

```json
{
  "calibration_ref": "reports/calibration/2026-01-04T12:00:00Z.json"
}
```

### Replay Validation

When loading a calibration record, the simulator must:

1. **Check schema version match**: Warn if `schema_version` differs
2. **Record provenance**: Store calibration path in output JSON
3. **Hash verification** (optional): Verify `dataset_hash_sha256` if source data available
4. **Apply parameters**: Override defaults with calibrated values

### Output Provenance

Simulation outputs that use calibration must include:

```json
{
  "calibration": {
    "applied": true,
    "calibration_ref": "reports/calibration/2026-01-04T12:00:00Z.json",
    "calibration_version": "1.0",
    "git_commit_at_calibration": "d17561f",
    "parameters_used": ["eta", "p_bg", "flip_prob"]
  }
}
```

---

## Storage Layout

```
reports/
└── calibration/
    ├── 2026-01-03T10:00:00Z.json
    ├── 2026-01-04T12:00:00Z.json
    └── latest.json -> 2026-01-04T12:00:00Z.json
```

- Timestamped files for historical record
- `latest.json` symlink points to most recent calibration
- Calibration files are append-only (never modified after creation)

---

## Existing Calibration Infrastructure

The simulator already provides calibration hooks in:

| Module | Function | Description |
|--------|----------|-------------|
| `calibration.py` | `CalibrationModel.from_file()` | Load calibration from JSON |
| `calibration.py` | `CalibrationModel.apply()` | Apply affine/piecewise calibration |
| `calibration_fit.py` | `fit_telemetry_parameters()` | Grid-search parameter fitting |
| `calibration_fit.py` | `compute_fit_quality()` | R², identifiability, uncertainty |
| `telemetry.py` | `load_telemetry()` | Load empirical data for fitting |

### Current Calibration Methods

1. **Affine**: `value_calibrated = value * scale + offset`
2. **Piecewise**: Linear interpolation from lookup table

---

## Red Team: Calibration Pitfalls

### Pitfall 1: Circular Calibration
**Risk:** Calibrating to simulation outputs, not hardware data.

**Mitigation:**
- `source.type` must be `empirical` for production use
- `synthetic` and `prior` types are for testing only

### Pitfall 2: Overfitting
**Risk:** Fitting too many parameters to limited data.

**Mitigation:**
- Check `fit_quality.condition_number` (warn if > 100)
- Check `fit_quality.identifiable` flag
- Document degrees of freedom vs. data points

### Pitfall 3: Version Drift
**Risk:** Using old calibration with new code.

**Mitigation:**
- `git_commit` enables traceability
- `schema_version` mismatch triggers warning
- Calibration expiration policy (recommend recalibration after N days)

### Pitfall 4: Silent Defaults
**Risk:** Forgetting to apply calibration, using defaults.

**Mitigation:**
- Output JSON includes `calibration.applied` flag
- Uncalibrated runs are clearly labeled
- CLI warns if no calibration file provided for production runs

---

## Changelog

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 1.0     | 2026-01-04 | Initial calibration hooks specification.     |
