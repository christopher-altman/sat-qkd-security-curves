# sat-qkd-security-curves — Operator UX Contract

**Version:** 1.0
**Status:** Specification
**Scope:** Streamlit dashboard as primary operator interface; CLI remains primary researcher interface.

---

## 1. Introduction

This contract defines the Operator UX for the sat-qkd-security-curves system. The dashboard serves as a layer between complex QKD simulation/analysis capabilities and users who need actionable outputs without deep protocol expertise.

### 1.1 Design Principles

1. **Progressive disclosure:** Simple by default, complexity on demand
2. **Fail-safe defaults:** Invalid configurations are prevented, not just warned
3. **Auditability:** Every run produces an export packet for traceability
4. **Two audiences:** Operators get GO/NO-GO; Researchers get full control

---

## 2. Personas

### 2.1 Link Engineer

**Goal:** Elevation/loss/fading sensitivity, pass-time performance envelopes

**Primary Tasks:**
1. Run pass-sweep simulations across elevation profiles
2. Visualize loss vs elevation and atmospheric effects
3. Compare day vs night passes
4. Assess pointing jitter impact on link margin
5. Evaluate turbulence/fading regimes
6. Export link budget summaries for mission planning

**Success Criteria:**
- Can identify minimum viable elevation for key generation
- Can quantify loss margin under worst-case fading
- Can compare aperture/pointing configurations

**Failure Modes:**
- Misinterprets simulated loss as measured loss
- Ignores fading variance (uses only mean)
- Sets unrealistic pointing jitter values

**Guardrails Required:**
- Warn when pointing jitter < 0.5 µrad (unrealistic for LEO)
- Warn when altitude < 200 km or > 2000 km
- Display "SIMULATED" watermark on all link outputs

---

### 2.2 Cryptographer

**Goal:** Finite-key/composable viability and security budget sensitivity

**Primary Tasks:**
1. Analyze finite-key penalties vs block size
2. Adjust epsilon budgets (ε_pe, ε_sec, ε_cor)
3. Compare asymptotic vs finite-key rates
4. Evaluate QBER headroom to abort threshold
5. Assess decoy-state bounds
6. Review security parameter breakdown

**Success Criteria:**
- Can identify minimum block size for positive key rate
- Can trace key failure to dominant epsilon term
- Can compare protocol variants (BB84 vs decoy)

**Failure Modes:**
- Uses asymptotic rates for small block sizes
- Misallocates epsilon budget (all to one term)
- Ignores abort rate in viability assessment

**Guardrails Required:**
- Force finite-key mode when n_sent < 10^6
- Warn when ε_total > 10^-6 (too loose)
- Warn when ε_total < 10^-15 (numerical precision)

---

### 2.3 Hardware / Timing Engineer

**Goal:** Coincidence behavior under timing jitter/background/dark counts

**Primary Tasks:**
1. Run coincidence simulations with timing parameters
2. Analyze CAR (coincidence-to-accidental ratio) vs loss
3. Evaluate afterpulsing and dead time effects
4. Assess CHSH S-value for entanglement-based setups
5. Tune coincidence window width
6. Compare detector parameter configurations

**Success Criteria:**
- Can identify timing jitter threshold for viable CAR
- Can predict coincidence rate degradation with loss
- Can optimize coincidence window for signal/noise tradeoff

**Failure Modes:**
- Confuses CAR with visibility
- Sets coincidence window too wide (accidentals dominate)
- Ignores afterpulsing in high-rate scenarios

**Guardrails Required:**
- Warn when jitter_ps > tau_ps (window smaller than jitter)
- Warn when dead_time_ps < 10 ps (unrealistic)
- Bound coincidence window to [0.1, 10] × jitter_ps

---

### 2.4 Ops / Nontechnical Subject

**Goal:** Run predefined scenarios, see GO/NO-GO and simple explanations, export results

**Primary Tasks:**
1. Select a scenario card (e.g., "Night Pass / Low Turbulence")
2. Click "Run" and wait for results
3. Read GO/NO-GO verdict with plain-language explanation
4. Export results packet for records
5. Compare two scenario runs side-by-side

**Success Criteria:**
- Can complete a scenario run without CLI interaction
- Can explain verdict to stakeholder in one sentence
- Can produce audit-ready export packet

**Failure Modes:**
- Modifies advanced parameters without understanding
- Misinterprets "SIMULATED" as "MEASURED"
- Skips export step, losing traceability

**Guardrails Required:**
- Advanced controls hidden by default
- Prominent "SIMULATED DATA" banner
- Auto-prompt for export after each run
- Plain-language verdict (not just numbers)

---

## 3. Information Architecture (Screens)

### 3.1 Screen A: Home / Scenario Runner

**Purpose:** Entry point for all users; scenario selection and execution

**Components:**
- Mode toggle: Operator Mode / Researcher Mode
- Scenario cards grid (6-8 predefined scenarios)
- "Run Selected Scenario" button (large, prominent)
- Status indicator: Idle / Running / Complete / Error
- Quick results panel:
  - GO / NO-GO badge (green/red)
  - Total loss (dB)
  - QBER estimate
  - Key viability summary (1 sentence)
- "SIMULATION" watermark (always visible)

**Scenario Cards (predefined):**

| Card Name | Description | Presets |
|-----------|-------------|---------|
| Night Pass (Clear) | Best-case LEO pass | night, no turbulence, 60° max elev |
| Night Pass (Turbulent) | Moderate fading | night, σ_ln=0.3, 60° max elev |
| Day Pass (Clear) | Daytime background | day, no turbulence, 60° max elev |
| Day Pass (Turbulent) | Worst-case daytime | day, σ_ln=0.5, 60° max elev |
| Low Elevation | Grazing pass | night, 30° max elev |
| High Jitter | Pointing-stressed | night, σ_point=10 µrad |
| Finite-Key Stress | Small block size | night, n_sent=10^5, finite-key |
| Custom | User-defined | (unlocks in Researcher mode) |

---

### 3.2 Screen B: Curves & Sweeps

**Purpose:** Visualize security curves across parameter ranges

**Components:**
- Plot selector: Loss vs Key Rate, QBER vs Loss, Elevation vs Key Rate
- Parameter sweep controls (Researcher mode only):
  - Loss range (min/max dB)
  - Steps (5-100)
  - Flip probability
  - Pulse count
- Compare runs toggle (overlay up to 3 runs)
- Export plot button (PNG + data CSV)

**Plots Available:**
- `qber_vs_loss_ci.png` — QBER with confidence intervals
- `secret_fraction_vs_loss_ci.png` — Secret fraction with CI
- `key_rate_vs_loss_ci.png` — Key rate per pulse
- `qber_headroom_vs_loss.png` — Headroom to abort threshold
- `key_rate_vs_elevation.png` — Key rate over pass

---

### 3.3 Screen C: Security (Finite-Key)

**Purpose:** Deep-dive into finite-key analysis and epsilon budget

**Components:**
- Epsilon budget breakdown (pie chart or bar):
  - ε_pe (parameter estimation)
  - ε_sec (secrecy)
  - ε_cor (correctness)
  - ε_total
- Finite-key penalty visualization:
  - Asymptotic rate vs finite-key rate
  - Penalty factor (%)
- "Why Key Failed" breakdown (when applicable):
  - Dominant term identification
  - Plain-language explanation
- Controls (Researcher mode):
  - ε_pe, ε_sec, ε_cor sliders (log scale)
  - PE fraction slider
  - EC efficiency input

**Key Outputs Displayed:**

| Field | JSON Key | Description |
|-------|----------|-------------|
| QBER Upper Bound | `qber_upper` | Conservative QBER estimate |
| Extractable Bits | `ell_bits` | Finite-key secret bits |
| Penalty Factor | `finite_size_penalty` | 1 - (finite/asymptotic) |
| Status | `finite_key.status` | "secure" or "insecure" |
| Reason | `finite_key.reason` | Why key failed (if applicable) |

---

### 3.4 Screen D: Link Budget / Physics

**Purpose:** Visualize physical link assumptions and geometry

**Components:**
- Elevation profile plot (time vs elevation)
- Loss breakdown:
  - Geometric/diffraction loss
  - Atmospheric loss
  - Pointing loss
  - System loss
  - Total loss
- Assumptions snapshot (human-readable table):
  - Wavelength, apertures, altitude
  - Pointing jitter, atmospheric model
  - Detector parameters
- Fading summary (when enabled):
  - Mean transmittance
  - Variance
  - Correlation time
  - Outage statistics

**Plots Available:**
- `loss_vs_elevation.png` — Total loss vs elevation
- `fading_ou_time.png` — Fading evolution (if OU enabled)
- `secure_window_per_pass.png` — Secure time window

---

### 3.5 Screen E: Telemetry / Observables

**Purpose:** Display pass-time observables and derived quantities

**Components:**
- Time-series plots:
  - Background rate vs time
  - Polarization angle vs time
  - CAR vs time (if coincidence data)
- Derived quantities:
  - Estimated QBER (with confidence band)
  - Headroom to abort
- Data source indicator:
  - "SIMULATED" or "TELEMETRY INGEST" badge
- Incident cards (if change points detected)

**Note:** This screen is most relevant when replaying telemetry or running the experiment harness. For pure simulation runs, it shows simulated observables.

---

### 3.6 Screen F: Exports

**Purpose:** Manage export packets for auditability

**Components:**
- "Create Export Packet" button
- Export history table:
  - Timestamp
  - Scenario name
  - GO/NO-GO verdict
  - File path
- Export preview (expandable):
  - Summary preview
  - File list
- Download/Copy path buttons

---

### 3.7 Screen G: Glossary / Help

**Purpose:** Define terms and map to JSON outputs

**Components:**
- Searchable glossary table
- Term → JSON key mapping
- "Operator explanation" column (plain language)
- Links to relevant documentation

---

## 4. Modes: Operator vs Researcher

### 4.1 Mode Selection

- Toggle in sidebar (top position)
- Default: **Operator Mode**
- Persists across session (stored in browser/Streamlit state)

### 4.2 Operator Mode

**Visible Controls:**
- Scenario cards (predefined only)
- Run button
- Export button
- Mode toggle

**Hidden Controls:**
- All parameter sliders
- Custom scenario card
- Advanced plots
- Raw JSON viewer
- Epsilon budget controls

**UI Characteristics:**
- Large fonts, high contrast
- GO/NO-GO prominently displayed
- Plain-language explanations
- Guardrails enforced (cannot bypass)
- "SIMULATED DATA" banner always visible

### 4.3 Researcher Mode

**Additional Controls:**
- All parameter sliders with full ranges
- Custom scenario card
- Raw JSON viewer (expandable)
- Epsilon budget fine-tuning
- Compare multiple runs
- Advanced plots (headroom, penalty breakdown)
- Override warnings (with confirmation)

**UI Characteristics:**
- Compact layout, more data density
- Numeric precision displayed
- Technical terminology
- Guardrails warn but allow override

---

## 5. Guardrails / Invalid Configuration Prevention

### 5.1 Parameter Bounds

| Parameter | Min | Max | Default | Rationale |
|-----------|-----|-----|---------|-----------|
| loss_min | 0 | 80 | 20 | Physical range |
| loss_max | 0 | 80 | 60 | Physical range |
| steps | 3 | 100 | 21 | Computational limits |
| flip_prob | 0 | 0.5 | 0.005 | Physical meaning |
| pulses | 1000 | 10^9 | 200000 | Memory/time limits |
| eta | 0.01 | 1.0 | 0.2 | Detector physics |
| p_bg | 0 | 0.1 | 1e-4 | Realistic range |
| sigma_point | 0.1 | 100 | 2.0 | µrad, realistic range |
| eps_pe | 1e-15 | 1e-3 | 1e-10 | Numerical precision |
| eps_sec | 1e-15 | 1e-3 | 1e-10 | Numerical precision |
| eps_cor | 1e-15 | 1e-3 | 1e-15 | Numerical precision |

### 5.2 Mutual Exclusions

- `--day` and `--night` are mutually exclusive (day flag presence determines mode)
- `--n-sent` overrides `--pulses` when both provided
- `--rep-rate` requires `--pass-seconds`

### 5.3 Warnings (Yellow Zone)

| Condition | Warning Message |
|-----------|-----------------|
| p_bg = 0 | "Zero background is unrealistic; results may be optimistic" |
| eta > 0.9 | "Detection efficiency > 90% is rarely achieved" |
| sigma_point < 0.5 | "Pointing jitter < 0.5 µrad is extremely challenging" |
| n_sent < 10^5 with finite-key | "Very small block; finite-key penalty may dominate" |
| eps_total > 1e-6 | "Security parameter is loose; consider tightening" |

### 5.4 Errors (Red Zone — Blocked)

| Condition | Error Message |
|-----------|---------------|
| loss_min > loss_max | "Minimum loss cannot exceed maximum" |
| eta <= 0 | "Detection efficiency must be positive" |
| flip_prob >= 0.5 | "Flip probability must be < 0.5" |
| steps < 3 | "At least 3 sweep steps required" |

### 5.5 Operator-Safe Defaults

When in Operator Mode, the following defaults are enforced:

```python
OPERATOR_SAFE_DEFAULTS = {
    "loss_min": 20.0,
    "loss_max": 60.0,
    "steps": 21,
    "flip_prob": 0.005,
    "pulses": 200_000,
    "eta": 0.2,
    "p_bg": 1e-4,
    "sigma_point": 2.0,  # µrad
    "finite_key": False,  # asymptotic by default
    "eps_pe": 1e-10,
    "eps_sec": 1e-10,
    "eps_cor": 1e-15,
}
```

---

## 6. Export Packet Contract

### 6.1 Purpose

Export packets provide a self-contained, auditable record of a simulation run. They enable:
- Reproducibility (all inputs captured)
- Traceability (timestamp, git commit)
- Sharing (single archive)

### 6.2 Packet Contents

```
exports/YYYY-MM-DD_HHMMSS_<scenario>/
├── latest.json              # Full simulation output
├── assumptions.json         # Assumptions manifest snapshot
├── summary.md               # Human-readable summary (see below)
├── figures/
│   ├── qber_vs_loss_ci.png
│   ├── key_rate_vs_loss_ci.png
│   ├── key_rate_vs_elevation.png
│   └── ... (all generated plots)
└── metadata.json            # Export metadata
```

### 6.3 summary.md Format

```markdown
# Simulation Summary

**Scenario:** Night Pass (Clear)
**Timestamp:** 2026-01-05T12:34:56Z
**Git Commit:** abc1234

## Verdict

**GO** — Key generation is viable under these conditions.

## Key Outputs

| Metric | Value |
|--------|-------|
| Peak Key Rate | 0.0012 bits/pulse |
| QBER (mean) | 2.5% |
| Headroom to Abort | 8.5% |
| Secure Window | 180 s |

## Top 5 Assumptions

1. Detector efficiency: 0.2
2. Background probability: 1e-4 per pulse
3. Pointing jitter: 2 µrad (2D Rayleigh)
4. Wavelength: 850 nm
5. Satellite altitude: 500 km

## Warnings

- None

---
*Export generated by sat-qkd-security-curves dashboard*
```

### 6.4 metadata.json Format

```json
{
  "export_version": "1.0",
  "timestamp_utc": "2026-01-05T12:34:56Z",
  "scenario_name": "Night Pass (Clear)",
  "verdict": "GO",
  "git_commit": "abc1234",
  "dashboard_mode": "operator",
  "files_included": [
    "latest.json",
    "assumptions.json",
    "summary.md",
    "figures/qber_vs_loss_ci.png",
    "figures/key_rate_vs_loss_ci.png"
  ]
}
```

### 6.5 Export Location

- Default: `exports/` directory in output path
- Naming convention: `YYYY-MM-DD_HHMMSS_<scenario_slug>/`
- Scenario slug: lowercase, underscores for spaces (e.g., `night_pass_clear`)

---

## 7. Tooltip/Glossary Mapping to JSON Outputs

### 7.1 Core Metrics

| UI Element | Definition | JSON Key | Operator Explanation |
|------------|------------|----------|---------------------|
| QBER | Quantum Bit Error Rate; fraction of sifted bits with errors | `qber_mean` | "How many bits came out wrong — lower is better" |
| Secret Fraction | Fraction of sifted bits extractable as secret key | `secret_fraction_mean` | "How much of your data becomes actual secret key" |
| Key Rate | Secret bits per emitted pulse | `key_rate_per_pulse_mean` | "How efficiently you're generating secret key" |
| Headroom | Distance from QBER to abort threshold (0.11) | `headroom` | "Safety margin before the protocol must abort" |
| Abort Rate | Fraction of trials that exceeded QBER threshold | `abort_rate` | "How often the protocol had to give up" |

### 7.2 Finite-Key Terms

| UI Element | Definition | JSON Key | Operator Explanation |
|------------|------------|----------|---------------------|
| QBER Upper | Conservative upper bound on QBER | `qber_upper` | "Worst-case error rate we must assume" |
| Extractable Bits | Finite-key secure bits | `ell_bits` | "Actual secret bits you can safely use" |
| Penalty Factor | Reduction from asymptotic rate | `finite_size_penalty` | "Tax paid for having limited data" |
| ε_total | Total security failure probability | `eps_total` | "Chance something goes wrong — smaller is safer" |
| Status | Secure or insecure determination | `finite_key.status` | "Final verdict on security" |
| Reason | Why key extraction failed | `finite_key.reason` | "What went wrong (if anything)" |

### 7.3 Link/Physical Terms

| UI Element | Definition | JSON Key | Operator Explanation |
|------------|------------|----------|---------------------|
| Loss (dB) | Total channel attenuation | `loss_db` | "How much signal you lose in transit" |
| Elevation | Angle above horizon | `elevation_deg` | "How high the satellite appears in the sky" |
| Secure Window | Duration where key generation is viable | `summary.secure_window_seconds` | "How long you can actually generate key" |
| CAR | Coincidence-to-Accidental Ratio | `car` | "Signal quality for entangled photon pairs" |
| Visibility | Fringe visibility (entanglement quality) | `visibility` | "How well-correlated your photon pairs are" |

### 7.4 Coincidence Terms

| UI Element | Definition | JSON Key | Operator Explanation |
|------------|------------|----------|---------------------|
| Coincidences | Detected photon pairs | `coincidences` | "Photon pairs that arrived together" |
| Accidentals | Random coincidences (noise) | `accidentals` | "False alarms from random timing" |
| CHSH S | Bell inequality witness | `chsh_s` | "Proof of quantum correlations (should be > 2)" |

### 7.5 Planned Additions (Not Yet in JSON)

| UI Element | Planned JSON Key | Status |
|------------|------------------|--------|
| Assumptions Snapshot | `assumptions_snapshot` | Phase-3 deliverable |
| Export Packet Path | `export_path` | Phase-3 deliverable |
| Scenario Preset Name | `scenario_preset` | Phase-3 deliverable |

---

## 8. CLI Wrapping Strategy

### 8.1 Scenario Card → CLI Mapping

| Scenario Card | CLI Command | Fixed Flags | Adjustable (Researcher) |
|---------------|-------------|-------------|------------------------|
| Night Pass (Clear) | `pass-sweep` | `--max-elevation 60` | `--pass-duration`, `--rep-rate` |
| Night Pass (Turbulent) | `pass-sweep` | `--max-elevation 60 --turbulence --sigma-ln 0.3` | `--sigma-ln` |
| Day Pass (Clear) | `pass-sweep` | `--day --max-elevation 60` | `--day-bg-factor` |
| Day Pass (Turbulent) | `pass-sweep` | `--day --turbulence --sigma-ln 0.5 --max-elevation 60` | `--sigma-ln` |
| Low Elevation | `pass-sweep` | `--max-elevation 30` | `--min-elevation` |
| High Jitter | `pass-sweep` | `--sigma-point 10` | `--sigma-point` |
| Finite-Key Stress | `sweep` | `--finite-key --pulses 100000` | `--eps-pe`, `--eps-sec` |
| Custom | (user-defined) | (none) | (all) |

### 8.2 Dashboard Tab → CLI Mapping

| Dashboard Tab | Primary CLI Command | Notes |
|---------------|---------------------|-------|
| Sweep | `sweep` | Loss sweep, CI mode |
| Pass | `pass-sweep` | Elevation-based sweep |
| Experiment | `experiment-run` | Blinded A/B harness |
| Forecast | `forecast-run` | Forecast scoring |
| Coincidence | `coincidence-sim` | Timing/coincidence analysis |

### 8.3 Wrapped Parameters

All CLI parameters are exposed through the dashboard in Researcher Mode. In Operator Mode, only scenario presets are available, which map to fixed CLI flag combinations.

---

## 9. Acceptance Criteria

### 9.1 Operator Workflow Acceptance

- [ ] Operator can select a scenario card and run simulation without typing CLI commands
- [ ] Operator can view GO/NO-GO verdict with plain-language explanation
- [ ] Operator can export a complete packet with one click
- [ ] Operator cannot accidentally set invalid parameter combinations
- [ ] "SIMULATED DATA" watermark is visible on all outputs

### 9.2 Researcher Workflow Acceptance

- [ ] Researcher can access all CLI parameters through UI controls
- [ ] Researcher can compare up to 3 runs side-by-side
- [ ] Researcher can view raw JSON output
- [ ] Researcher can override warnings with confirmation
- [ ] Researcher can create custom scenarios

### 9.3 Export Packet Acceptance

- [ ] Export packet contains all required files (JSON, plots, summary)
- [ ] Export packet is self-contained (no external dependencies)
- [ ] summary.md is human-readable without technical expertise
- [ ] metadata.json includes git commit for reproducibility

### 9.4 Glossary Acceptance

- [ ] All top-level metrics shown in UI have glossary entries
- [ ] Glossary maps to actual JSON keys (not invented keys)
- [ ] Operator explanations use plain language (no jargon)

### 9.5 Integration Acceptance

- [ ] Dashboard launches with `./py -m sat_qkd_lab.dashboard`
- [ ] Dashboard respects `--outdir` for output location
- [ ] Dashboard blinded by default (no label leakage)

---

## 10. Implementation Notes for Sonnet

### 10.1 Risk Areas

1. **Mode persistence:** Streamlit session state can be tricky; ensure mode toggle persists correctly
2. **Guardrail enforcement:** Must happen both client-side (disable controls) and server-side (validate before run)
3. **Export packet atomicity:** Use temp directory + rename to avoid partial exports
4. **Plot caching:** Large sweeps generate many plots; consider caching strategy

### 10.2 Dependencies on Existing Code

- `dashboard.py` already implements basic tab structure
- `run.py` provides all CLI argument parsing (reuse validators)
- `assumptions.py` provides `build_assumptions_manifest()` for export
- `plotting.py` provides all plot generation functions

### 10.3 New Code Required

1. Scenario preset definitions (dataclass or dict)
2. Export packet generation function
3. summary.md template rendering
4. Mode toggle state management
5. Guardrail validation layer
6. Glossary data structure

### 10.4 Testing Strategy

- Unit tests for guardrail validation
- Integration tests for scenario → CLI mapping
- E2E tests for export packet completeness
- Visual regression tests for key plots (optional)

---

## Appendix A: Current Dashboard State

The existing `dashboard.py` provides:
- 8 tabs: Sweep, Pass, Experiment, Forecast, Ops, Instrument, Protocol, Incidents
- Basic Streamlit controls for each tab
- Output directory configuration
- Unblind toggle
- Plot index viewer

This contract extends and restructures the existing dashboard to support the persona-based UX design.

---

## Appendix B: JSON Output Files Reference

| File | CLI Command | Key Contents |
|------|-------------|--------------|
| `reports/latest.json` | `sweep` | `loss_sweep_ci`, `assumptions_manifest`, `parameters` |
| `reports/latest_pass.json` | `pass-sweep` | `time_series`, `summary`, `inputs` |
| `reports/latest_experiment.json` | `experiment-run` | `block_results`, `analysis`, `blinding` |
| `reports/latest_coincidence.json` | `coincidence-sim` | `records`, `summary`, `artifacts` |
| `reports/schedule_blinded.json` | `experiment-run` | Schedule without labels |

---

## Appendix C: Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-05 | Initial specification |
