# sat-qkd-security-curves — Link Model Card (Physical Assumptions Contract)

**Version:** 1.0
**Status:** Stable public spec

## Scope

This document defines the physical assumptions underlying the satellite-to-ground optical link model used in security curve generation. It provides traceability from each assumption to its code location and clarifies what constitutes "within validity envelope" for each parameter.

**What is modeled:**
- Satellite-to-ground free-space optical link geometry (LEO)
- Diffraction-limited beam propagation with Gaussian approximation
- Pointing jitter with 2D Rayleigh distribution (magnitude)
- Atmospheric extinction using Kasten-Young airmass formula
- Lognormal turbulence fading (weak-to-moderate regime)
- Day/night background noise scaling
- Detector efficiency and dark count effects

**What is NOT modeled (out of scope):**
- Adaptive optics correction
- Fried parameter (r0) or isoplanatic angle dynamics
- Real-time atmospheric sensing or weather fusion
- Beam wander or tip-tilt beyond pointing jitter
- Multi-aperture or coherent combining receivers
- Validated radiative transfer (MODTRAN-level accuracy)

**Validity envelope:**
Assumptions are valid for:
- LEO orbits (400–600 km altitude)
- Elevation angles 10°–90°
- Near-infrared wavelengths (800–900 nm)
- Clear-to-moderate atmospheric conditions
- Ground station apertures 0.5–1.5 m
- Pointing jitter 1–10 µrad RMS

---

## 1. Executive Summary

The link model stack transforms satellite geometry into channel loss through the following pipeline:

```
Elevation angle (deg)
    ↓
Slant range geometry (spherical Earth)
    ↓
Geometric coupling (Gaussian beam + aperture)
    ↓
Pointing loss (Rayleigh 2-axis jitter → average coupling)
    ↓
Atmospheric extinction (Kasten-Young airmass × zenith loss)
    ↓
System losses (optics, detector coupling)
    ↓
Total loss_db
    ↓
Detection probability → QBER → Secret fraction
```

**Fading/turbulence** enters as a multiplicative transmittance factor (lognormal or OU process) applied after deterministic loss calculation.

**Background noise** enters as additive click probability per pulse window, scaled by day/night mode.

### Why This Matters for Security Curves

The link model determines where the "security cliff" occurs:
- At low loss: signal dominates, QBER is low, security margin is high
- At high loss: background clicks dominate, QBER rises toward 0.5, security vanishes

If link assumptions are incorrect (e.g., unrealistically low background, unrealistically good pointing), the curves will show security where none exists. If assumptions are overly pessimistic, viable operating regimes may appear infeasible.

---

## 2. Assumption Traceability Table

| Parameter | Symbol | Default | Units | Distribution/Model | Dim | Code Source | Override | Affects | Sensitivity | Safety |
|-----------|--------|---------|-------|-------------------|-----|-------------|----------|---------|-------------|--------|
| Wavelength | λ | 850e-9 | m | Constant | — | `free_space_link.py:FreeSpaceLinkParams.wavelength_m` | `--wavelength` | loss (diffraction) | Low | Green |
| TX aperture diameter | D_tx | 0.30 | m | Constant | — | `free_space_link.py:FreeSpaceLinkParams.tx_diameter_m` | `--tx-diameter` | loss (divergence) | Med | Green |
| RX aperture diameter | D_rx | 1.0 | m | Constant | — | `free_space_link.py:FreeSpaceLinkParams.rx_diameter_m` | `--rx-diameter` | loss (coupling) | Med | Green |
| Beam divergence | θ_div | Computed | rad | 1.22λ/D_tx (diffraction) | — | `free_space_link.py:effective_divergence_rad` | `beam_divergence_rad` param | loss | Med | Green |
| Pointing jitter RMS | σ_point | 2e-6 | rad | Rayleigh (2D Gaussian magnitude) | 2-axis | `free_space_link.py:FreeSpaceLinkParams.sigma_point_rad` | `--sigma-point` | loss | High | Yellow |
| Satellite altitude | h | 500e3 | m | Constant | — | `free_space_link.py:FreeSpaceLinkParams.altitude_m` | `--altitude` | slant range | Low | Green |
| Earth radius | R_E | 6371e3 | m | Constant | — | `free_space_link.py:FreeSpaceLinkParams.earth_radius_m` | param only | geometry | Low | Green |
| Zenith atm. loss | L_atm_z | 0.5 | dB | Constant | — | `free_space_link.py:FreeSpaceLinkParams.atm_loss_db_zenith` | `--atm-loss-db` | loss | Med | Green |
| Turbulence fading | σ_ln | 0.0 | — | Lognormal(−σ²/2, σ) | — | `free_space_link.py:FreeSpaceLinkParams.sigma_ln` | `--sigma-ln` | loss variance | High | Yellow |
| System loss | L_sys | 3.0 | dB | Constant | — | `free_space_link.py:FreeSpaceLinkParams.system_loss_db` | `--system-loss-db` | loss | Low | Green |
| Day/night mode | is_night | True | bool | Discrete | — | `free_space_link.py:FreeSpaceLinkParams.is_night` | `--day` flag | background | Med | Green |
| Day background factor | f_day | 100.0 | — | Multiplicative | — | `free_space_link.py:FreeSpaceLinkParams.day_background_factor` | `--day-bg-factor` | QBER | High | Yellow |
| Detection efficiency | η | 0.2 | — | Constant | — | `detector.py:DetectorParams.eta` | `--eta` | yields, QBER | High | Yellow |
| Background prob. | p_bg | 1e-4 | /pulse | Constant | — | `detector.py:DetectorParams.p_bg` | `--p-bg` | QBER | High | Yellow |
| Afterpulse prob. | p_ap | 0.0 | — | Constant | — | `detector.py:DetectorParams.p_afterpulse` | `--afterpulse-prob` | yields | Med | Yellow |
| Dead time | τ_dead | 0 | pulses | Constant | — | `detector.py:DetectorParams.dead_time_pulses` | `--dead-time-pulses` | yields | Med | Yellow |
| OU fading mean | µ_OU | 1.0 | — | OU process | — | `ou_fading.py:simulate_ou_transmittance` | `--fading-ou-mean` | loss variance | Med | Green |
| OU fading sigma | σ_OU | 0.1 | — | OU process | — | `ou_fading.py:simulate_ou_transmittance` | `--fading-ou-sigma` | loss variance | Med | Yellow |
| OU reversion rate | θ_OU | (1/30) | 1/s | OU process | — | CLI: `--fading-ou-tau-s` → θ = 1/τ | `--fading-ou-tau-s` | correlation | Med | Green |
| Pointing jitter (alt) | σ_jit | 2.0 | µrad | OU-driven 1D | 1-axis | `pointing.py:PointingParams.pointing_jitter_urad` | `--pointing-jitter-urad` | loss | High | Yellow |
| Acquisition time | t_acq | 0.0 | s | Step function | — | `pointing.py:PointingParams.acq_seconds` | `--pointing-acq-s` | window | Low | Green |
| Dropout probability | p_drop | 0.0 | /s | Poisson rate proxy | — | `pointing.py:PointingParams.dropout_prob` | `--pointing-dropout-prob` | window | Med | Yellow |
| Timing jitter | σ_t | 0.0 | s | Gaussian | — | `timing.py:TimingModel.jitter_sigma_s` | `--jitter-ps` (ps) | coincidence | Med | Yellow |
| TDC quantization | Δ_TDC | 0.0 | s | Uniform | — | `timing.py:TimingModel.tdc_seconds` | param only | coincidence | Low | Green |

---

## 3. Atmospheric/Extinction/Airmass Assumptions

### Airmass Formula

The repo implements the **Kasten-Young airmass formula** in `free_space_link.py:atmospheric_extinction_db()`:

```python
airmass = 1.0 / (sin(el_rad) + 0.50572 * (el + 6.07995) ** (-1.6364))
```

This is a well-known empirical formula accurate to ~0.5° elevation.

### Elevation-to-Airmass Mapping

| Elevation (deg) | Airmass (approx) |
|-----------------|------------------|
| 90 (zenith) | 1.0 |
| 60 | 1.15 |
| 45 | 1.41 |
| 30 | 2.0 |
| 20 | 2.9 |
| 10 | 5.6 |

### Total Atmospheric Loss

Atmospheric loss is computed as:

```
L_atm(el) = L_atm_zenith × airmass(el)
```

Default `L_atm_zenith = 0.5 dB` corresponds to good clear-sky conditions at 850 nm.

### Alternative Atmosphere Models (atmosphere.py)

The `atmosphere.py` module provides additional scenario generators:

| Model | Description | Use case |
|-------|-------------|----------|
| `"none"` | Zero attenuation | Baseline/vacuum |
| `"simple_clear_sky"` | 0.2 dB/km constant | Simple scenarios |
| `"kruse"` | Visibility-based empirical | Aerosol/haze scenarios |

These are **scenario generators**, not validated radiative transfer models.

---

## 4. Pointing Jitter + Fading Assumptions

### Pointing Jitter Distribution

**Primary model** (`free_space_link.py`):
- Distribution: **2D Gaussian** (Rayleigh magnitude)
- Parameterization: RMS σ_point in radians
- Dimensionality: **2-axis** (azimuth + elevation combined into radial)
- Entry point: `pointing_loss_db()` computes average loss factor

The average pointing loss for Gaussian beam with Rayleigh jitter:
```
<η_point> = 1 / (1 + (σ_point/θ_div)²)
L_point_dB = −10 log₁₀(<η_point>)
```

**Secondary model** (`pointing.py`):
- Distribution: **1D OU process** for time-varying jitter
- Generates lock state and transmittance multiplier per time step
- Includes acquisition delay, dropout, and relock dynamics

### Fading Models

**Lognormal fading** (`free_space_link.py`, `fading_samples.py`):
- Distribution: Lognormal with unit mean
- Parameters: σ_ln (log-amplitude standard deviation)
- Formula: T = exp(2χ) where χ ~ N(−σ²/2, σ²)
- Regime: Weak-to-moderate turbulence

**OU fading** (`ou_fading.py`):
- Distribution: Ornstein-Uhlenbeck process
- Parameters: µ (mean), θ (reversion rate = 1/τ), σ (volatility)
- Produces correlated transmittance time series
- Clamped to [0, 1]

### Integration Points

Fading enters in `pass_model.py`:
- Lognormal samples multiply the deterministic transmittance
- OU process provides time-correlated transmittance evolution

---

## 5. Detector + Background + Timing Assumptions

### Background Rate Model

Background clicks are modeled as a constant probability per pulse window:

| Parameter | Default | Source |
|-----------|---------|--------|
| Base p_bg | 1e-4 | `detector.py:DetectorParams.p_bg` |
| Day factor | 100× | `free_space_link.py:day_background_factor` |

Effective background: `p_bg_eff = p_bg × (1 if night else day_factor)`

This is a **lumped model** combining:
- Detector dark counts
- Stray light
- Sky radiance (crude day/night proxy)

### Dark Count Defaults

Dark counts are included in `p_bg`. The `optics.py` module provides additional temperature scaling:
```
DCR(T) = DCR_base × (1 + 0.02 × (T − 20°C))
```

### Detector Efficiency

| Parameter | Default | Range |
|-----------|---------|-------|
| η | 0.2 | [0, 1] |
| η_z (Z-basis) | η | [0, 1] |
| η_x (X-basis) | η | [0, 1] |

This is a lumped efficiency covering:
- Detector quantum efficiency
- Optical coupling
- Filtering losses

### Timing Jitter

The `timing.py` module models clock and timing effects:

| Parameter | Default | Effect |
|-----------|---------|--------|
| jitter_sigma_s | 0.0 | Gaussian timing jitter |
| tdc_seconds | 0.0 | TDC quantization step |
| drift_ppm | 0.0 | Clock drift |

Timing jitter affects coincidence window matching, which impacts:
- CAR (coincidence-to-accidental ratio)
- Apparent QBER (timing mismatch → wrong bit assignments)

---

## 6. Override Safety Notes (Link Model Card)

### Red Zone (changes can invalidate security claims)

| Change | Risk |
|--------|------|
| Setting p_bg = 0 | Unrealistic: QBER floor vanishes, curves show security where none exists |
| Setting η = 1.0 without justification | Unrealistic: hides detection losses |
| Negative loss_db values | Unphysical: gain in passive channel |
| sigma_point = 0 with real tracking | Hides pointing loss, optimistic curves |
| Modifying sifting factor in protocol | Wrong key rate calculation (security issue) |

### Yellow Zone (requires paired validation)

| Change | Required Validation |
|--------|---------------------|
| σ_point outside [1, 10] µrad | Compare against real tracking data |
| σ_ln > 0.5 | Verify weak turbulence assumption still holds |
| p_bg outside [1e-6, 1e-2] | Check against detector spec sheets |
| η outside [0.1, 0.9] | Justify with detector datasheet |
| Day background factor ≠ 100 | Calibrate against measured day/night ratio |
| Afterpulsing enabled | Run `test_decoy_realism.py` to verify yields |

### Green Zone (safe parameter changes)

| Change | Safety |
|--------|--------|
| Wavelength 800–1600 nm | Valid for near-IR optics |
| TX/RX aperture 0.1–2.0 m | Geometry scales correctly |
| Altitude 400–800 km | LEO range |
| Zenith atm. loss 0.2–2.0 dB | Reasonable clear-to-hazy |
| System loss 1–10 dB | Covers typical optical chains |

### Security-Relevant Invariants

These invariants MUST hold:

1. **p_bg ≥ 0**: Background cannot be negative
2. **0 ≤ η ≤ 1**: Detection efficiency is a probability
3. **loss_db ≥ 0**: Passive channel cannot amplify
4. **QBER ∈ [0, 0.5]**: By definition
5. **Sifting factor = 0.5 for BB84**: Hardcoded, do not override
6. **Finite-key epsilon values ∈ (0, 1)**: Probabilities

---

## 7. Validation Hooks + Recommended Checks

### Tests Protecting Link Model

| Test File | Component Protected |
|-----------|---------------------|
| `test_optical_link_integration.py` | Free-space link budget |
| `test_fading_model.py` | Lognormal fading |
| `test_ou_fading.py` | OU fading process |
| `test_fading_samples.py` | Fading sample statistics |
| `test_pointing_dynamics.py` | Pointing lock/unlock |
| `test_atmosphere_models.py` | Kruse/clear-sky models |

### Tests Protecting Detector Model

| Test File | Component Protected |
|-----------|---------------------|
| `test_detector_effects.py` | Afterpulsing, dead time |
| `test_detector_attacks.py` | Attack-detector interaction |
| `test_decoy_realism.py` | Decoy + detector combined |

### Tests Protecting Timing

| Test File | Component Protected |
|-----------|---------------------|
| `test_clock_sync.py` | Offset/drift estimation |
| `test_timing_sync_layer.py` | Timing model application |
| `test_timetag_coincidence.py` | Coincidence matching |

### Coverage Gaps (noted)

- No direct unit test for Kasten-Young airmass formula accuracy
- No cross-validation against measured satellite link data
- No turbulence profile (Cn2) tests

### Sensitivity Sweep Checklist

Run these sweeps to sanity-check curve behavior:

1. **p_bg sweep**: 1e-6 → 1e-2 (should see QBER rise at high loss)
2. **η sweep**: 0.1 → 0.5 (should see curves shift right/left)
3. **σ_point sweep**: 1 → 10 µrad (should see loss increase)
4. **σ_ln sweep**: 0 → 0.5 (should see variance increase, mean shift)
5. **zenith atm. loss sweep**: 0.2 → 2.0 dB (should see low-elevation loss increase)

---

## 8. References

Anchor papers for interpreting link model assumptions:

1. Liao, S.-K., et al. (2017). Satellite-to-ground quantum key distribution. *Nature* 549, 43–47.

2. Bourgoin, J.-P., et al. (2013). A comprehensive design and performance analysis of low Earth orbit satellite quantum communication. *New Journal of Physics* 15, 023006.

3. Andrews, L. C., & Phillips, R. L. (2005). *Laser Beam Propagation through Random Media* (2nd ed.). SPIE Press.

4. Bedington, R., Arrazola, J. M., & Ling, A. (2017). Progress in satellite quantum key distribution. *npj Quantum Information* 3, 30.

5. Kasten, F., & Young, A. T. (1989). Revised optical air mass tables and approximation formula. *Applied Optics* 28, 4735–4738.

---

## 9. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-05 | Initial link model card |
