# CV-QKD (GG02) Scaffold

## Purpose

This document describes the **scaffold implementation** of Continuous-Variable Quantum Key Distribution (CV-QKD) using the GG02 protocol (Gaussian-modulated coherent states). This is a **structural demonstration** to signal capability and intent, NOT a production-grade security proof.

---

## What Is Implemented

### Module: `src/sat_qkd_lab/cv/gg02.py`

1. **Parameter dataclass**: `GG02Params`
   - `V_A`: Alice's modulation variance (shot noise units)
   - `T`: Channel transmittance (0 to 1)
   - `xi`: Excess noise referred to channel input (shot noise units)
   - `eta`: Bob's detection efficiency (0 to 1)
   - `v_el`: Bob's electronic noise (shot noise units)
   - `beta`: Reconciliation efficiency (0 to 1)

2. **Pure functions**:
   - `compute_snr(params)`: Signal-to-noise ratio estimate
   - `compute_mutual_information(params)`: Mutual information I(A:B) using standard Gaussian channel formula
   - `compute_holevo_bound(params)`: **STUBBED** (returns `None`)
   - `compute_secret_key_rate(params)`: Returns `GG02Result` with SNR, I(A:B), and status

3. **Result dataclass**: `GG02Result`
   - `snr`: Signal-to-noise ratio (linear scale)
   - `I_AB`: Mutual information (bits per channel use)
   - `chi_BE`: Holevo bound (None if not implemented)
   - `secret_key_rate`: Secret key rate estimate (None if chi_BE unavailable)
   - `status`: One of `"toy"`, `"stub"`, `"not_implemented"`

---

## CLI Command: `cv-sweep`

### Usage

```bash
./py -m sat_qkd_lab.run cv-sweep [OPTIONS]
```

### Arguments

| Argument         | Default | Description                                      |
|------------------|---------|--------------------------------------------------|
| `--loss-min`     | 0.0     | Minimum channel loss in dB                       |
| `--loss-max`     | 30.0    | Maximum channel loss in dB                       |
| `--steps`        | 31      | Number of loss values to sweep                   |
| `--v-a`          | 10.0    | Alice's modulation variance (shot noise units)   |
| `--xi`           | 0.01    | Excess noise (shot noise units)                  |
| `--eta`          | 0.6     | Bob's detection efficiency                       |
| `--v-el`         | 0.01    | Electronic noise (shot noise units)              |
| `--beta`         | 0.95    | Reconciliation efficiency                        |
| `--outdir`       | `.`     | Output directory for reports/figures             |

### Outputs

1. **Plot**: `figures/cv_snr_vs_loss.png`
   - SNR (linear scale) vs channel loss (dB)

2. **Plot**: `figures/cv_mutual_info_vs_loss.png`
   - Mutual information I(A:B) (bits/use) vs channel loss (dB)

3. **Report**: `reports/latest.json`
   - Appends `cv_gg02_sweep` section to existing report (or creates new)
   - Includes parameters, records (per loss value), summary, and artifacts

---

## What Is NOT Implemented

The following are **explicitly NOT implemented** in this scaffold:

1. **Holevo bound computation** (`chi_BE`):
   - Requires covariance matrix analysis for optimal collective attack
   - Requires symplectic eigenvalue computation
   - Requires von Neumann entropy calculations
   - Returns `None` in current implementation

2. **Secret key rate**:
   - Cannot be computed without a validated Holevo bound
   - Returns `None` in current implementation

3. **Composable security proof**:
   - This is a pedagogical model, not a composable security proof

4. **Reverse reconciliation**:
   - Only toy direct reconciliation formula is used

5. **Finite-size effects**:
   - No finite-key corrections are applied

---

## Validation Gates

Before promoting this scaffold to production, the following gates MUST pass:

### Gate 1: Holevo Bound Implementation
- Implement `compute_holevo_bound()` with validated formulas
- Add unit tests comparing against published benchmarks
- Document which attack model is used (e.g., collective, coherent)

### Gate 2: Secret Key Rate Validation
- Compute secret key rate as `K = beta * I(A:B) - chi(B:E)`
- Validate against literature results for standard GG02 scenarios
- Add tests for positive/negative key rate regions

### Gate 3: Reconciliation Direction
- Implement reverse reconciliation (if needed)
- Document trade-offs between direct and reverse reconciliation
- Add tests for both modes

### Gate 4: Finite-Size Effects
- Add finite-key corrections for statistical fluctuations
- Validate against composable security bounds (if available)

### Gate 5: Threat Model Documentation
- Explicitly state which attacks are/aren't covered
- Document assumptions about Eve's capabilities
- Clarify channel noise vs. adversarial noise attribution

---

## Loss-to-Transmittance Mapping

Channel loss (dB) is converted to transmittance using:

```
T = 10^(-loss_db / 10)
```

This mapping is recorded in the assumptions manifest under `cv_qkd.loss_to_transmittance_mapping`.

---

## Status Field Semantics

Results include a `status` field to clarify the level of validation:

| Status              | Meaning                                                    |
|---------------------|------------------------------------------------------------|
| `"toy"`             | Pedagogical implementation; not for security claims        |
| `"stub"`            | Partially implemented; missing critical components         |
| `"not_implemented"` | Placeholder; no computation performed                      |

Current implementation returns `"stub"` because the Holevo bound is not computed.

---

## References (Titles Only)

1. **Grosshans & Grangier (2002)**: "Continuous Variable Quantum Cryptography Using Coherent States" - PRL 88, 057902
2. **Weedbrook et al. (2012)**: "Gaussian quantum information" - Rev. Mod. Phys. 84, 621
3. **Leverrier et al. (2010)**: "Finite-size analysis of a continuous-variable quantum key distribution" - Phys. Rev. A 81, 062343
4. **Pirandola et al. (2020)**: "Advances in quantum cryptography" - Adv. Opt. Photon. 12, 1012

---

## Disclaimers

1. **Not a security proof**: This scaffold is for structural demonstration and API design. Do NOT use for production security claims.

2. **Not validated against experiments**: The formulas are from standard references but have not been validated against real CV-QKD hardware.

3. **Holevo bound is stubbed**: Without a validated Holevo bound, the secret key rate cannot be computed reliably.

4. **Toy threat model**: The assumptions about Eve's capabilities are placeholders, not rigorous threat models.

---

## Integration with Assumptions Manifest

The assumptions manifest (`./py -m sat_qkd_lab.run assumptions`) includes a `cv_qkd` section that documents:

- Protocol: GG02 (Gaussian-modulated coherent states)
- Status: scaffold
- Threat model: placeholder; not a security guarantee
- What is computed: SNR, I(A:B)
- What is NOT computed: chi(B:E), validated secret key rate
- Loss-to-transmittance mapping: `T = 10^(-loss_db/10)`

---

## Future Work

1. Implement Holevo bound computation
2. Validate against published benchmarks
3. Add finite-size corrections
4. Document composable security framework (if applicable)
5. Add reverse reconciliation mode
6. Integrate with existing BB84 security curve infrastructure for comparison plots
