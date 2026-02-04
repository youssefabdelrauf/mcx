# PPG Simulation Analysis Report

**Generated:** 2026-01-30 15:02:14

## Summary

- **Wavelength Range:** 660 - 940 nm
- **Skin Types Analyzed:** 6 (Fitzpatrick I-VI)
- **Total Simulations:** 18

## Key Findings

### Optimal Wavelength for Skin-Tone Independence

The wavelength with **minimum variance** in penetration depth across skin types:

**lambda_optimal = 940 nm**

This wavelength shows the most consistent signal quality regardless of melanin content.

### Wavelength with Maximum Bias

The wavelength with **maximum variance** (most affected by melanin):

**lambda_worst = 660 nm**

Avoid this wavelength for skin-tone independent PPG sensors.

## Wavelength Performance Summary

| Wavelength | Mean Penetration (mm) | Variance | Range (mm) |
|------------|----------------------|----------|------------|
| 660 nm | 0.214 | 0.000907 | 0.086 |
| 880 nm | 0.312 | 0.000130 | 0.033 |
| 940 nm | 0.336 | 0.000107 | 0.030 |

## Recommendations

1. **For minimal skin-tone bias:** Use wavelengths > 800 nm (near-infrared)
2. **Avoid:** Wavelengths < 600 nm where melanin absorption is highest
3. **SpO2 optimization:** Consider non-standard wavelength pairs that minimize RoR variance

## Figures Generated

1. `fig1_intensity_vs_depth.png` - Intensity vs depth curves
2. `fig1b_skin_comparison.png` - Skin type comparison per wavelength
3. `fig2_signal_vs_skintone.png` - Signal strength degradation
4. `fig2b_signal_vs_melanin.png` - Signal vs melanin concentration
5. `fig3_bias_comparison.png` - Bias comparison heatmap

## Tables Generated

1. `penetration_depths.csv` - Penetration depth for each wavelength/skin type
2. `signal_attenuation.csv` - % signal reduction vs Type I
