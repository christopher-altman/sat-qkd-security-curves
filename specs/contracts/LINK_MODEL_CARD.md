# sat-qkd-security-curves — Link Model Card (Physical Assumptions Contract)

**Version:** 1.0
**Status:** Stable public spec

## Scope

This contract codifies the physical assumptions that feed the satellite-to-ground security curves produced by the `sweep` and `pass-sweep` workflows. The canonical link stack is implemented in `src/sat_qkd_lab/free_space_link.py`, wrapped by `src/sat_qkd_lab/pass_model.py`/`sweep.py`, and exposed through `src/sat_qkd_lab/run.py` CLI commands. It includes

- Free-space geometry (LEO/ground) with diffraction-limited optics
- Pointing jitter and optional acquisition/dropout dynamics
- Atmospheric extinction (Kasten-Young airmass) and optional aerosol scenarios
- Turbulence/fading (lognormal and OU) and time-correlated backgrounds
- Detector efficiency, dark/background clicks, and coincidence gating

**Not modeled:** adaptive optics, real-time weather fusion, multi-aperture receivers, detailed radiative transfer (MODTRAN), full polarization subtleties beyond basis-bias/rotation toggles, and satellite orbit propagation beyond fixed altitude.

## Defaults & Assumptions

The table below lists the physics knobs that feed η, QBER, sift-rate, and finite-key penalty. Each default is backed by a dataclass field, module constant, or CLI default.

| Parameter | Default | Units | Where defined | CLI/constructor override |
|-----------|---------|-------|----------------|-------------------------|
| Wavelength (λ) | 850e-9 | m | `src/sat_qkd_lab/free_space_link.py:FreeSpaceLinkParams.wavelength_m` | `--wavelength` |
| Transmitter aperture (D_tx) | 0.30 | m | `FreeSpaceLinkParams.tx_diameter_m` | `--tx-diameter` |
| Receiver aperture (D_rx) | 1.0 | m | `FreeSpaceLinkParams.rx_diameter_m` | `--rx-diameter` |
| Beam divergence half-angle | `1.22 × λ / D_tx` (diffraction limit) | rad | `FreeSpaceLinkParams.effective_divergence_rad` | Set `beam_divergence_rad` when instantiating `FreeSpaceLinkParams` |
| Pointing jitter (σ_point, 2D Rayleigh magnitude) | 2e-6 | rad | `FreeSpaceLinkParams.sigma_point_rad` | `--sigma-point` |
| Satellite altitude (h) | 500e3 | m | `FreeSpaceLinkParams.altitude_m` | `--altitude` |
| Earth radius (R_E) | 6371e3 | m | `FreeSpaceLinkParams.earth_radius_m` | fixed in code (no CLI) |
| Zenith atmospheric loss (L_atm_zenith) | 0.5 | dB | `FreeSpaceLinkParams.atm_loss_db_zenith` | `--atm-loss-db` |
| System loss (optics, coupler, detector) | 3.0 | dB | `FreeSpaceLinkParams.system_loss_db` | `--system-loss-db` |
| Day/night mode | Night (True) | flag | `FreeSpaceLinkParams.is_night` via `run.py` pass-sweep logic | `--day` / `--is-night` |
| Day background factor | 100 | × | `FreeSpaceLinkParams.day_background_factor` | `--day-bg-factor` |
| Base background probability (p_bg) | 1e-4 | per pulse window | `src/sat_qkd_lab/detector.py:DetectorParams.p_bg` | `--p-bg` |
| Detection efficiency (η) | 0.2 | — | `DetectorParams.eta` | `--eta` |
| Afterpulse probability | 0.0 | — | `DetectorParams.p_afterpulse` | `--afterpulse-prob` |
| Afterpulse window | 0 | pulses | `DetectorParams.afterpulse_window` | `--afterpulse-window` |
| Afterpulse decay | 0.0 | — | `DetectorParams.afterpulse_decay` | `--afterpulse-decay` |
| Dead time | 0 | pulses | `DetectorParams.dead_time_pulses` | `--dead-time-pulses` |
| Optical filter bandwidth | 1.0 | nm | `src/sat_qkd_lab/optics.py:OpticalParams.filter_bandwidth_nm` | `--filter-bandwidth-nm` |
| Detector temperature | 20.0 | °C | `OpticalParams.detector_temp_c` | `--detector-temp-c` |
| Lognormal scintillation std (σ_ln) | 0.0 (0.3 when `--turbulence` is active) | — | `FreeSpaceLinkParams.sigma_ln` / `run.py` pass-sweep | `--sigma-ln` (guards `--turbulence`) |
| Fading samples (per-step lognormal) | σ_ln=0.3, samples=50, trials=1, seed=0 | — | `src/sat_qkd_lab/pass_model.py:FadingParams` | `--fading`, `--fading-sigma-ln`, `--fading-samples`, `--fading-trials`, `--fading-seed` |
| OU fading (time-correlated transmittance) | μ=1.0, σ=0.1, τ=30 s, outage threshold 0.2 | — | `src/sat_qkd_lab/run.py` + `src/sat_qkd_lab/ou_fading.py:simulate_ou_transmittance` | `--fading-ou`, `--fading-ou-mean`, `--fading-ou-sigma`, `--fading-ou-tau-s`, `--fading-ou-outage-threshold` |
| Pointing acquisition / dropout | acq=0 s, dropout=0 s⁻¹, relock=0 s, jitter=2 µrad | s / µrad | `src/sat_qkd_lab/pointing.py:PointingParams` | `--pointing`, `--acq-seconds`, `--dropout-prob`, `--relock-seconds`, `--pointing-jitter-urad` |
| Time-correlated background process | mean=1.0, σ=0.2, τ=60 s | — | `src/sat_qkd_lab/background_process.py` | `--background-process`, `--bg-ou-mean`, `--bg-ou-sigma`, `--bg-ou-tau-s` |
| Atmosphere scenario + visibility | `none` (options `simple_clear_sky`, `kruse` with `visibility km`) | — | `src/sat_qkd_lab/atmosphere.py` | `--atmosphere-model`, `--visibility-km`, `--wavelength-nm` |
| Coincidence window (τ) | 200 | ps | `src/sat_qkd_lab/run.py` commands `clock-sync`, `sync-estimate`, `coincidence-sim` | `--tau-ps` |
| Timing jitter (σ_t) | 0 (but 20 ps for `clock-sync`/`sync-estimate`, 80 ps for `coincidence-sim`) | s | `src/sat_qkd_lab/timing.py:TimingModel.jitter_sigma_s` + `run.py` `--jitter-ps` | `--jitter-ps` |
| TDC quantization | 0 | s | `TimingModel.tdc_seconds` | `--tdc-ps` |
| HIL ingestion coincidence window | 200 | ps | `run.py` `--ingest-tau-ps` | `--ingest-tau-ps` |

### Derived noise scaling

- **Effective background per pulse:** `p_bg_eff = background_rate_hz × (mode factor) × p_bg`, where `background_rate_hz` multiplies `p_bg` by filter bandwidth and day/night mode as implemented in `src/sat_qkd_lab/optics.py`. The night factor is 1× and the day factor is 100×, consistent with the `--day` flag.
- **Dark-count temperature scaling:** `dark_count_rate_hz` multiplies the base rate by `1 + 0.02 × (T − 20 °C)` (clamped ≥ 0). The resulting rate is used to set `DetectorParams.p_bg` before simulation.
- **Effective background for pass sweeps:** `effective_background_prob` in `run.py`/`free_space_link.py` multiplies `p_bg` by the day/night factor at each time step; optional `--background-process` adds OU fluctuations.

### Atmospheric extinction & geometry

- **Airmass:** `free_space_link.atmospheric_extinction_db()` uses the Kasten–Young formula (`airmass = 1/(sin(el) + 0.50572*(el+6.07995)^(-1.6364))`) for elevation angles down to ~1°. Total atmospheric loss is `L_atm_zenith × airmass`.
- **Geometric coupling:** Gaussian beam radius grows as `range × θ` and receiver coupling uses the 1 − exp(−2(a/w)^2) truncation. The divergence θ is either the diffraction limit or `beam_divergence_rad` override.
- **Atmosphere scenario scripts:** `--atmosphere-model` adds additive loss from `simple_clear_sky` (0.2 dB/km) or `kruse` visibility-based formulas; the default `none` means no extra attenuation beyond Kasten–Young.

### Pointing, fading, and security windows

- **Pointing loss:** `free_space_link.pointing_loss_db()` averages Rayleigh-distributed jitter, giving `η_pointing = 1/(1 + (σ_point/θ)^2)`. A zero jitter override yields 0 dB pointing loss but is guarded by the override safety rules below.
- **Pointing dynamics (optional `--pointing`):** `simulate_pointing_profile` produces lock-state, transmittance multipliers, and dropout statistics for acquisition time, dropout probability, and relock delay.
- **Lognormal turbulence:** `sweep` and `pass_model` multiply deterministic transmittance by lognormal samples (unit mean) when `--turbulence` or `--fading` are enabled. The standard deviation is controlled by `σ_ln`.
- **Ornstein–Uhlenbeck fading:** `--fading-ou` generates time-correlated transmittance traces (`simulate_ou_transmittance`) that are clipped to [0, 1] and yield outage clusters via `compute_outage_clusters`.
- **Background process:** `--background-process` applies OU scaling to `p_bg_eff`, letting mean, sigma, and correlation time tune the temporal noise.

### Timing & coincidences

- **Timing model:** `TimingModel` defaults to zero offset, zero drift, zero jitter, and zero TDC quantization. CLI commands (e.g., `clock-sync`, `sync-estimate`, `coincidence-sim`) inject picosecond-level jitter (`--jitter-ps`) and window (`--tau-ps`).
- **Coincidence matching:** `coincidence.py` enforces `|Δt| ≤ τ` around the gate. The ingestion/HIL playback window uses `--ingest-tau-ps`, and `--tdc-ps` applies quantization via `TimingModel.tdc_seconds`.

## What Changes What

| Parameter(s) | Primary outputs affected |
|---------------|---------------------------|
| λ, D_tx, D_rx, beam divergence, altitude, geometric + pointing loss | Channel transmittance η (loss_db) → direct shift of QBER and secret fraction |
| σ_point, pointing OU dynamics (`--pointing` args) | Coupling loss, dropouts, secure window duration → η, QBER, sift rate |
| L_atm_zenith, airmass (elevation) | Atmospheric loss → η, QBER, key rate per pulse |
| System loss | Lumped attenuation → η, secret fraction |
| σ_ln (`--turbulence`/`--fading`), fading OU parameters | Loss variance → QBER spread, finite-key penalty, outage statistics |
| p_bg, day/night factor, filter bandwidth, detector temperature, background OU | Background click rate → QBER floor, incidental sift bits, secret fraction |
| η, afterpulse probability/window/decay, dead time | Signal yield + accidental clicks → sift rate, finite-key penalty, detectable QBER offsets |
| Coincidence window τ, timing jitter, TDC quantization | Accidental coincidences vs signal peaks → QBER, CAR, finite-key surplus |

## Override Safety

| Zone | Guidance |
|------|----------|
| **Green (safe)** | Wavelength in 800–1600 nm, D_tx/D_rx in 0.1–2.0 m, altitude 400–800 km, zenith loss 0.2–2.0 dB, system loss 1–10 dB, day/night ratio tuning via `--day-bg-factor`. These change loss gradually without violating passive-channel constraints. |
| **Yellow (requires validation)** | Pointing jitter outside [1, 10] µrad (`--sigma-point`), `σ_ln > 0.5`, `--fading-sigma-ln`, `--fading-ou-mean/sigma/tau`, `p_bg` outside [1e-6, 1e-2], η outside [0.1, 0.9], enabling afterpulses/dead time without recalibrating finite-key bounds, changing `--filter-bandwidth-nm` or `--detector-temp-c` far from lab values, or toggling `--background-process` without verifying time scales. |
| **Red (dangerous)** | Setting `p_bg` to 0, `η` to 1, `σ_point` to 0 without real tracking data, negative loss values (e.g., negative `system_loss`), ignoring finite-key abort thresholds (e.g., forcing QBER ≳ 11%), or breaking the pass geometry (e.g., sending fewer pulses than time steps). These invalidate security claims. |

## Anchors

- Liao, S.-K., et al. (2017). Satellite-to-ground quantum key distribution. *Nature* 549, 43–47. The assumptions target the same near-infrared, low-background regime that the Micius satellite explored.
- Bourgoin, J.-P., et al. (2013). A comprehensive design and performance analysis of low Earth orbit satellite quantum communication. *New Journal of Physics* 15, 023006. The link envelope and turbulence reasoning follow the broad design space laid out in that work.
