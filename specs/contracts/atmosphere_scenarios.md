# Atmosphere Scenarios (NOT Forecasts)

## Purpose

This document describes atmospheric attenuation modeling for **scenario generation** in QKD link planning. These are parameterized models used to explore plausible atmospheric conditions, NOT meteorological forecasts or measurements.

---

## What This IS

- **Scenario Generator**: Parameterized models for exploring atmospheric effects on QKD links
- **Planning Tool**: Helps assess QKD performance under different visibility conditions
- **Loss Component**: Adds atmospheric attenuation to existing link budget

## What This IS NOT

- ❌ **NOT a meteorological forecast**
- ❌ **NOT a sensor fusion pipeline**
- ❌ **NOT a validated atmospheric propagation model**
- ❌ **NOT real-time weather observation**
- ❌ **NOT a radiative transfer model (MODTRAN, etc.)**

---

## Available Models

### `none` (default)

No atmospheric attenuation beyond the existing `--atm-loss-db` parameter.

```bash
./py -m sat_qkd_lab.run pass-sweep --atmosphere-model none
```

Use when:
- You want baseline behavior (no change from previous versions)
- Atmospheric effects are already captured in `--atm-loss-db`

### `simple_clear_sky`

Constant baseline attenuation of **0.2 dB/km** for clear-sky scenarios.

```bash
./py -m sat_qkd_lab.run pass-sweep --atmosphere-model simple_clear_sky
```

Use when:
- You want a simple clear-weather scenario
- Testing atmospheric integration without visibility complexity

### `kruse`

Visibility-based empirical model (Kruse approximation) for aerosol scattering.

```bash
./py -m sat_qkd_lab.run pass-sweep \\
    --atmosphere-model kruse \\
    --visibility-km 23.0 \\
    --wavelength-nm 850.0
```

Use when:
- You want to explore visibility-dependent scenarios
- Modeling haze, fog, or varying atmospheric clarity

**Visibility Guidelines:**
- `>40 km`: Excellent visibility (clear day)
- `20-40 km`: Good visibility (typical clear conditions)
- `10-20 km`: Moderate visibility (light haze)
- `<10 km`: Poor visibility (fog, smog, heavy haze)

---

## Atmospheric Loss Integration

### Formula

The atmospheric attenuation (dB/km) is integrated over the slant path to produce total atmospheric loss:

```
atmosphere_loss_db = attenuation_db_km(...) * slant_path_km
```

### Slant Path Approximation

Plane-parallel toy approximation (NOT a validated atmospheric model):

```
slant_path_km = 1.0 / max(sin(elevation_deg * π / 180), 0.1)
```

**Typical values:**
- 90° (zenith): ~1 km
- 45°: ~1.4 km
- 30°: ~2 km
- 10°: ~5.7 km (clamped by min_sin=0.1)

**This is a TOY approximation.** It does NOT account for:
- Earth curvature
- Atmospheric refraction
- Vertical density profiles
- Actual atmospheric scale height

---

## Loss Bookkeeping

When `--atmosphere-model` is not `none`, the code adds atmospheric loss to the total and records it explicitly:

### JSON Structure

```json
{
  "pass_sweep": {
    "records": [
      {
        "time_s": 0.0,
        "elevation_deg": 45.0,
        "loss_db": 12.5,
        "loss_components": {
          "system_loss_db": 3.0,
          "atmosphere_loss_db": 0.28
        }
      }
    ]
  }
}
```

### Validation Rule

```python
abs(loss_db - sum(loss_components.*)) <= float_eps
```

**Note:** The current implementation adds atmosphere loss to the existing `loss_db` from `total_link_loss_db`, which already includes geometric, pointing, atmospheric extinction, and system losses. The `loss_components` structure records the additive atmosphere scenario contribution separately.

---

## CLI Usage

### Basic Usage

```bash
# Default (no atmosphere scenario loss)
./py -m sat_qkd_lab.run pass-sweep

# Simple clear-sky scenario
./py -m sat_qkd_lab.run pass-sweep --atmosphere-model simple_clear_sky

# Kruse model with good visibility
./py -m sat_qkd_lab.run pass-sweep \\
    --atmosphere-model kruse \\
    --visibility-km 30.0 \\
    --wavelength-nm 850.0

# Kruse model with poor visibility (haze)
./py -m sat_qkd_lab.run pass-sweep \\
    --atmosphere-model kruse \\
    --visibility-km 10.0 \\
    --wavelength-nm 850.0
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--atmosphere-model` | `none` | Model: `none`, `kruse`, `simple_clear_sky` |
| `--visibility-km` | 23.0 | Meteorological visibility in km (for kruse) |
| `--wavelength-nm` | 850.0 | Optical wavelength in nm |

---

## Assumptions Manifest Integration

The atmosphere section is automatically added to `reports/latest.json`:

```json
{
  "assumptions_manifest": {
    "atmosphere": {
      "purpose": "Scenario generator for atmospheric attenuation modeling",
      "status": "scenario_generator",
      "not_a_forecast": "This is NOT a meteorological forecast...",
      "models": { ... },
      "disclaimers": [ ... ]
    }
  }
}
```

To view:
```bash
./py -m sat_qkd_lab.run assumptions | jq .atmosphere
```

---

## Example Scenarios

### Scenario 1: Clear Day

```bash
./py -m sat_qkd_lab.run pass-sweep \\
    --atmosphere-model kruse \\
    --visibility-km 40.0 \\
    --max-elevation 70 \\
    --pass-duration 300
```

### Scenario 2: Moderate Haze

```bash
./py -m sat_qkd_lab.run pass-sweep \\
    --atmosphere-model kruse \\
    --visibility-km 15.0 \\
    --max-elevation 70 \\
    --pass-duration 300
```

### Scenario 3: Fog/Heavy Haze

```bash
./py -m sat_qkd_lab.run pass-sweep \\
    --atmosphere-model kruse \\
    --visibility-km 5.0 \\
    --max-elevation 70 \\
    --pass-duration 300
```

Compare the `loss_db` and `loss_components.atmosphere_loss_db` values across scenarios to see the impact of varying visibility.

---

## Intended Use Cases

✅ **DO use for:**
- Exploring QKD link performance under different visibility conditions
- Scenario-based link budget planning
- Sensitivity analysis for atmospheric variability
- Comparing clear-sky vs. hazy conditions

❌ **DO NOT use for:**
- Meteorological forecasting
- Real-time weather prediction
- Mission-critical link availability predictions
- Replacing validated atmospheric propagation tools

---

## Limitations

1. **Plane-Parallel Approximation**: Slant path calculation is a toy geometric model
2. **No Refraction**: Does not account for atmospheric bending of light
3. **No Vertical Profiles**: Assumes uniform atmosphere (no density stratification)
4. **No Temporal Dynamics**: Static visibility per sweep (no weather evolution)
5. **Empirical Model**: Kruse is an approximation, not first-principles physics
6. **No Molecular Absorption**: Only aerosol scattering is modeled

For production use, integrate validated atmospheric propagation tools (MODTRAN, etc.) and real-time weather data.

---

## References

- Kruse, P.W., et al. (1962): Visibility and atmospheric scattering empirical models
- Kim, McArthur, & Korevaar (2001): Free-space optical communications through atmospheric turbulence
- Weichel (1990): Laser Beam Propagation in the Atmosphere

---

## Future Work

1. Add support for molecular absorption (water vapor, O2, etc.)
2. Integrate vertical atmospheric profiles (exponential density decay)
3. Add temporal evolution of visibility (weather time series)
4. Provide hooks for MODTRAN or validated atmospheric models
5. Add atmospheric turbulence coupling (link to scintillation models)
