# Wavelength and Sensor Geometry Optimization to Eliminate Skin-Tone Bias in Photoplethysmography

[![License](https://img.shields.io/badge/License-Academic-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)]()
[![MCX](https://img.shields.io/badge/MCX-Compatible-green.svg)]()

## Overview

This repository contains the code and simulation framework for our research on eliminating skin-tone bias in photoplethysmography (PPG) and pulse oximetry (SpO₂) through physics-driven hardware optimization. Unlike conventional approaches that rely on post-processing or algorithmic corrections, we optimize the optical sensor design itself—specifically wavelength selection and sensor geometry—to achieve intrinsic robustness against skin pigmentation variability.

**Key Finding**: We identified 880 nm as an intrinsically melanin-independent wavelength and a novel 530/590 nm wavelength pair that achieves a **91.9% reduction** in skin-tone-dependent SpO₂ bias compared to the standard 660/940 nm configuration.

## GUI

![GUI Demo](Screen%20Recording%202026-02-05%20220429.gif)

*The interactive PyQt5 GUI allows configuration of simulation parameters, running optimizations, and visualizing results.*

### Running the GUI

To launch the graphical interface:

```bash
cd mcx
python ppg_gui.py
```

**GUI Features:**
- Configure wavelength range, skin types, and photon counts
- Run wavelength optimization, SDD optimization, and SpO₂ pair analysis
- View real-time simulation progress
- Visualize results with interactive plots
- Export optimization results

## Team

**Authors:**  
Youssef Mahmoud Abdelrauf Mostafa, Rahma Ashraf Mohamed Khalifa, Rawan Abdallah Kotb, Habiba Abdelmneam Ramadan , Mahmoud Mohamed Mahmoud Ahmed

**Institution:** Cairo University  
**Contact:** ashrafrahma402@gmail.com  
**Mentor:** Dr. Sherif ElGohary

## The Problem

Photoplethysmography (PPG) and pulse oximetry exhibit systematic inaccuracies in individuals with darker skin tones, with studies showing:
- Black patients experience nearly **3× the rate of occult hypoxemia** undetected by standard pulse oximetry
- Measurement errors can reach **15%** in individuals with darker skin tones
- The bias arises from melanin absorption, which attenuates incident photons before they reach pulsatile dermal vasculature

Current mitigation strategies (signal post-processing, increased illumination) do not address the underlying physical interaction between light and tissue.

## Our Solution

We present a **physics-driven, multi-parameter optimization framework** that systematically explores:

1. **Wavelength selection** (500–1000 nm sweep)
2. **Sensor geometry** (source–detector distance: 2–10 mm)
3. **Skin phototype** (Fitzpatrick I–VI)

Using high-fidelity Monte Carlo tissue modeling, we identified hardware-level design parameters that minimize skin-tone bias at the source.

## Key Results



### 1. Optimal Single Wavelength: 880 nm
- **98.27% melanin independence score**
- Negligible perfusion index variance across all skin tones (σ² = 2.65 × 10⁻⁹)
- Intrinsically robust against pigmentation variability

### 2. Novel Wavelength Pair: 530 nm / 590 nm
- **91.9% reduction** in skin-tone-dependent SpO₂ bias
- Dramatically outperforms the standard 660/940 nm configuration
- Maintains accurate oxygenation measurement across Fitzpatrick types I–VI

### 3. Optimal Source–Detector Distance: 3 mm
- Balances signal quality and melanin-independent performance
- Suitable for wearable PPG configurations at 880 nm
- Maximizes penetration depth while minimizing skin-tone variance

## Methodology

### Monte Carlo Simulation Framework

**Software:** GPU-accelerated Monte Carlo eXtreme (MCX)  
**Hardware:** NVIDIA RTX 3080 GPU  
**Photon Count:** 10⁷ photons per configuration  
**Convergence:** Coefficient of variation < 2%

### Multi-Layer Skin Model

We implemented a physiologically accurate skin model consisting of:
- Stratum corneum
- Epidermis (with melanin variation)
- Papillary dermis
- Reticular dermis
- Subcutaneous tissue

**Melanin Modeling:**  
Fitzpatrick skin types I–VI were modeled by varying epidermal melanin volume fraction using the empirical Jacques model:

```
μₐ,melanin = 6.6 × 10¹¹ · λ⁻³·³³ · fmelanin · 0.1
```

**Blood Oxygenation:**  
SpO₂ levels of 70%, 80%, 90%, and 100% were simulated using linear mixtures of HbO₂ and Hb absorption coefficients.

### Optimization Metrics

- **Perfusion Index (PI):** AC/DC ratio of PPG signal
- **Ratio-of-Ratios (RoR):** (AC/DC)λ₁ / (AC/DC)λ₂
- **Skin-tone dependence:** Quantified by PI variance across Fitzpatrick types
- **Independence Score:** Scale-independent comparison metric

## Repository Structure

```
.
├── simulations/
│   ├── wavelength_sweep/      # 500-1000nm wavelength analysis
│   ├── geometry_optimization/  # Source-detector distance optimization
│   └── skin_models/           # Fitzpatrick I-VI tissue models
├── data/
│   ├── optical_properties/    # Wavelength-dependent μₐ, μₛ, g, n
│   ├── results/               # Simulation outputs and analysis
│   └── validation/            # Clinical validation datasets
├── scripts/
│   ├── run_mcx_simulations.py
│   ├── analyze_pi_variance.py
│   ├── calculate_ror_bias.py
│   └── generate_figures.py
├── images/                    # Figures and visualizations
│   └── pi_variance_vs_wavelength.png
├── figures/                   # Publication-quality figures
├── docs/                      # Additional documentation
└── README.md
```

## Installation & Requirements

### Prerequisites

```bash
# NVIDIA CUDA-capable GPU
# NVIDIA CUDA Toolkit 11.0+
# Python 3.7+
```

### Monte Carlo eXtreme (MCX)

1. Download MCX from [https://mcx.space](https://mcx.space)
2. Install NVIDIA GPU drivers
3. Verify installation:

```bash
mcx -L  # List available GPUs
```

### Python Dependencies

```bash
pip install numpy scipy matplotlib pandas
pip install pmcx  # Python MCX wrapper
```

## Usage

### Running Simulations

#### 1. Single Wavelength Optimization

```bash
python scripts/run_mcx_simulations.py \
    --wavelength-range 500 1000 20 \
    --skin-types I II III IV V VI \
    --photons 1e7 \
    --output results/wavelength_sweep/
```

#### 2. Dual-Wavelength SpO₂ Analysis

```bash
python scripts/calculate_ror_bias.py \
    --wavelength-pairs "660,940" "530,590" \
    --spo2-levels 70 80 90 100 \
    --skin-types I II III IV V VI \
    --output results/spo2_analysis/
```

#### 3. Geometry Optimization

```bash
python scripts/optimize_sdd.py \
    --wavelength 880 \
    --sdd-range 2 10 1 \
    --skin-types I II III IV V VI \
    --output results/geometry/
```

### Analyzing Results

```python
import numpy as np
import pandas as pd

# Load wavelength sweep results
data = pd.read_csv('results/wavelength_sweep/pi_variance.csv')

# Calculate independence score
data['independence_score'] = (1 - data['pi_variance'] / data['pi_variance'].max()) * 100

# Identify optimal wavelength
optimal_wl = data.loc[data['pi_variance'].idxmin(), 'wavelength']
print(f"Optimal wavelength: {optimal_wl} nm")
```

## Reproducing Key Figures

### Figure 1: Depth-Resolved Photon Intensity Profiles

```bash
python scripts/generate_figures.py --figure depth_profiles \
    --wavelengths 520 660 880 \
    --skin-types I III VI
```

### Figure 2: Wavelength-Dependent Bias Heatmap

```bash
python scripts/generate_figures.py --figure bias_heatmap \
    --wavelength-range 500 1000 \
    --skin-types I II III IV V VI
```

### Figure 3: Optimization Results Summary

```bash
python scripts/generate_figures.py --figure optimization_summary
```

## Key Findings in Detail

### Wavelength Optimization Results

The graph above shows the complete wavelength sweep from 500-1000 nm. Key wavelengths of interest:

| Wavelength (nm) | PI Variance (σ²) | Independence Score (%) |
|-----------------|------------------|------------------------|
| 500             | 8.54 × 10⁻⁹      | 94.42                  |
| 530 (proposed)  | 3.97 × 10⁻⁸      | 74.10                  |
| 590 (proposed)  | 1.39 × 10⁻⁷      | 9.25                   |
| 660 (standard)  | 1.75 × 10⁻⁸      | 88.59                  |
| **880 (optimal)** | **2.65 × 10⁻⁹** | **98.27**             |
| 940 (standard)  | 2.24 × 10⁻⁸      | 85.38                  |

*Note: The 530/590 nm pair achieves 91.9% bias reduction through complementary spectral properties, not individual wavelength independence.*

### Physical Interpretation

**Why 880 nm works:**
- Reduced melanin absorption (μₐ ∝ λ⁻³·³³)
- Sufficient hemoglobin contrast for cardiac pulsatility detection
- Optimal balance before water absorption increases (>950 nm)
- Deep penetration into dermal vasculature across all skin tones

**Why 530/590 nm outperforms 660/940 nm:**
- Better wavelength separation in the melanin absorption curve
- Reduced differential melanin interference between wavelengths
- Maintains adequate HbO₂/Hb discrimination

## Clinical Implications

1. **Immediate Impact:** Next-generation pulse oximeters using 880 nm or 530/590 nm wavelength pairs
2. **Wearable Devices:** 3 mm source–detector distance enables compact, equitable fitness trackers
3. **Regulatory Guidance:** Hardware-level solutions may satisfy emerging FDA requirements for bias mitigation
4. **Health Equity:** Reduced misdiagnosis and treatment delays in underserved populations

## Limitations & Future Work

### Current Limitations
- Simulations assume homogeneous blood oxygenation
- No modeling of motion artifacts or ambient light interference
- Limited to transmittance/reflectance geometries

### Ongoing Research
1. **Clinical Validation:** In-vivo testing across diverse populations
2. **Prototype Development:** Wearable sensors implementing optimized parameters
3. **Multi-Modal Extension:** Applying framework to other optical sensing modalities
4. **Real-Time Algorithms:** Combining hardware optimization with adaptive signal processing



## Related Publications

Our work builds upon and contributes to the following research areas:

**Bias in Pulse Oximetry:**
- Sjoding MW et al., "Racial bias in pulse oximetry measurement," NEJM, 2020
- Bent B et al., "Investigating sources of inaccuracy in wearable optical heart rate sensors," NPJ Digital Medicine, 2020

**Monte Carlo Photon Transport:**
- Fang Q, Boas DA, "Monte Carlo simulation of photon migration in 3D turbid media accelerated by GPUs," Optics Express, 2009
- Jacques SL, "Optical properties of biological tissues: a review," Physics in Medicine & Biology, 2013

## License

This project is released for academic and research purposes. Please contact the authors for commercial applications.

## Acknowledgments

We thank:
- **Dr. Sherif ElGohary** for mentorship and guidance
- **Cairo University** for computational resources
- **MCX Development Team** for the open-source Monte Carlo simulation platform
- The broader biomedical optics community for foundational research on light-tissue interactions

## Contact & Contributions

**Questions or Collaboration?**  
Email: ashrafrahma402@gmail.com

**Found a bug or have a suggestion?**  
Please open an issue or submit a pull request

---

**Keywords:** Photoplethysmography, Monte Carlo simulation, Pulse oximetry, Skin-tone bias, Biomedical optics, Health equity, Wearable sensors, Optical design optimization
