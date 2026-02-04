#!/usr/bin/env python3
"""
PPG Optimization Engine: Wavelength & SDD Optimization for Skin-Tone Independence

This module implements physics-based optimization to find:
1. Optimal Single Wavelength (λ_opt): Minimizes Perfusion Index variance across skin types
2. Melanin-Insensitive SpO2 Pair (λA, λB): Minimizes skin-tone deviation in RoR
3. Optimal Source-Detector Distance (SDD_opt): Maximizes signal while maintaining independence

Author: Generated for PPG optimization study
"""

import numpy as np
import json
import os
from datetime import datetime
from itertools import combinations

# Import simulation functions
from ppg_simulation import (
    FITZPATRICK_TYPES,
    get_wavelength_range,
    run_single_simulation,
    extract_detector_signal,
    get_optical_properties,
    create_skin_volume,
    hemoglobin_absorption
)

try:
    import pmcx
    HAS_PMCX = True
except ImportError:
    HAS_PMCX = False
    print("Warning: pmcx not available. Running in mock mode.")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_perfusion_index(dc_signal, ac_signal):
    """
    Calculate Perfusion Index.
    
    PI = (AC / DC) * 100%
    """
    if dc_signal > 0:
        return (ac_signal / dc_signal) * 100
    return 0.0


def calculate_ratio_of_ratios(ac_red, dc_red, ac_ir, dc_ir):
    """
    Calculate Ratio of Ratios (R) for SpO2 estimation.
    
    R = (AC_red / DC_red) / (AC_ir / DC_ir)
    
    This is the fundamental measurement used to estimate SpO2.
    The calibration curve SpO2 = f(R) is empirically determined.
    """
    if dc_red > 0 and dc_ir > 0 and ac_ir > 0:
        ratio_red = ac_red / dc_red
        ratio_ir = ac_ir / dc_ir
        if ratio_ir > 0:
            return ratio_red / ratio_ir
    return 0.0


def get_fine_wavelength_range(start=500, end=1000, step=10):
    """Get wavelengths for optimization with fine resolution."""
    return list(range(start, end + 1, step))


# =============================================================================
# PULSATILE SIMULATION WITH VARIABLE PARAMETERS
# =============================================================================

def run_pulsatile_simulation_advanced(wavelength_nm, skin_type='I', spo2=0.98, 
                                       sdd_mm=5.0, nphoton=1e6, test_mode=False):
    """
    Run pulsatile simulation with variable SpO2 and SDD.
    
    Parameters:
    -----------
    wavelength_nm : float
        Wavelength in nm
    skin_type : str
        Fitzpatrick skin type ('I' to 'VI')
    spo2 : float
        Oxygen saturation (0.70 to 1.00)
    sdd_mm : float
        Source-Detector Distance in mm
    nphoton : float
        Number of photons
    test_mode : bool
        Use reduced photons for testing
        
    Returns:
    --------
    result : dict
        Contains DC, AC, PI, and metadata
    """
    if test_mode:
        nphoton = 1e5
    
    melanin_fraction = FITZPATRICK_TYPES[skin_type]['melanin_fraction']
    
    # Volume dimensions
    nx, ny, nz = 100, 100, 200
    unit_um = 10
    unit_mm = unit_um / 1000.0
    
    # Create volume
    vol, layer_boundaries = create_skin_volume(nx, ny, nz, unit_um)
    
    # Diastolic state (baseline blood volume)
    blood_diastolic = 0.02
    prop_diastolic = get_optical_properties_with_spo2(
        wavelength_nm, melanin_fraction, blood_diastolic, spo2
    )
    
    # Systolic state (increased blood volume)
    blood_systolic = 0.022  # 10% increase
    prop_systolic = get_optical_properties_with_spo2(
        wavelength_nm, melanin_fraction, blood_systolic, spo2
    )
    
    # Run diastolic simulation
    cfg_base = {
        'nphoton': int(nphoton),
        'vol': vol,
        'tstart': 0,
        'tend': 5e-9,
        'tstep': 5e-9,
        'unitinmm': unit_mm,
        'srctype': 'disk',
        'srcpos': [nx/2, ny/2, 0],
        'srcdir': [0, 0, 1],
        'srcparam1': [5, 0, 0, 0],
        'isreflect': 1,
        'issrcfrom0': 1,
        'autopilot': 1,
        'gpuid': 1,
        'outputtype': 'fluence',
    }
    
    cfg_diastolic = cfg_base.copy()
    cfg_diastolic['prop'] = prop_diastolic
    
    cfg_systolic = cfg_base.copy()
    cfg_systolic['prop'] = prop_systolic
    
    try:
        res_diastolic = pmcx.run(cfg_diastolic)
        res_diastolic['unit_mm'] = unit_mm
        
        res_systolic = pmcx.run(cfg_systolic)
        res_systolic['unit_mm'] = unit_mm
        
        # Extract signals at specified SDD
        dc_signal = extract_detector_signal_at_sdd(res_diastolic, sdd_mm, unit_mm)
        sys_signal = extract_detector_signal_at_sdd(res_systolic, sdd_mm, unit_mm)
        
        ac_signal = abs(dc_signal - sys_signal)
        pi = calculate_perfusion_index(dc_signal, ac_signal)
        
        return {
            'wavelength_nm': wavelength_nm,
            'skin_type': skin_type,
            'skin_name': FITZPATRICK_TYPES[skin_type]['name'],
            'melanin_fraction': melanin_fraction,
            'spo2': spo2,
            'sdd_mm': sdd_mm,
            'dc_signal': dc_signal,
            'ac_signal': ac_signal,
            'perfusion_index': pi,
            'success': True
        }
        
    except Exception as e:
        print(f"Simulation error: {e}")
        return {
            'wavelength_nm': wavelength_nm,
            'skin_type': skin_type,
            'success': False,
            'error': str(e)
        }


def get_optical_properties_with_spo2(wavelength_nm, melanin_fraction, blood_fraction, spo2):
    """
    Get optical properties with explicit SpO2 control.
    """
    from ppg_simulation import tissue_scattering, melanin_absorption
    
    # Layer 0: Background
    background = [0.0, 0.0, 1.0, 1.0]
    
    # Layer 1: Stratum corneum
    sc_mua = 0.01 + 0.1 * ((wavelength_nm / 500) ** -1)
    sc_mus = tissue_scattering(wavelength_nm, 'stratum_corneum')
    stratum_corneum = [sc_mua, sc_mus, 0.9, 1.55]
    
    # Layer 2: Epidermis (melanin)
    epid_mua = 0.05 + melanin_absorption(wavelength_nm, melanin_fraction)
    epid_mus = tissue_scattering(wavelength_nm, 'epidermis')
    epidermis = [epid_mua, epid_mus, 0.85, 1.44]
    
    # Layer 3: Papillary dermis (blood with SpO2)
    derm_p_mua = 0.02 + hemoglobin_absorption(wavelength_nm, blood_fraction * 1.5, spo2)
    derm_p_mus = tissue_scattering(wavelength_nm, 'dermis')
    papillary_dermis = [derm_p_mua, derm_p_mus, 0.90, 1.40]
    
    # Layer 4: Reticular dermis
    derm_r_mua = 0.01 + hemoglobin_absorption(wavelength_nm, blood_fraction * 0.5, spo2)
    derm_r_mus = tissue_scattering(wavelength_nm, 'dermis') * 0.8
    reticular_dermis = [derm_r_mua, derm_r_mus, 0.90, 1.40]
    
    # Layer 5: Subcutis
    subcut_mua = 0.005 + hemoglobin_absorption(wavelength_nm, blood_fraction * 0.3, spo2)
    subcut_mus = tissue_scattering(wavelength_nm, 'subcutis')
    subcutis = [subcut_mua, subcut_mus, 0.80, 1.44]
    
    return np.array([
        background, stratum_corneum, epidermis,
        papillary_dermis, reticular_dermis, subcutis
    ], dtype=np.float32)


def extract_detector_signal_at_sdd(result, sdd_mm, unit_mm):
    """
    Extract detector signal at specified Source-Detector Distance.
    """
    flux = result['flux']
    
    if flux.ndim == 4:
        flux = np.squeeze(flux)
    
    nx, ny, nz = flux.shape
    cx, cy = nx // 2, ny // 2
    
    dist_voxels = int(sdd_mm / unit_mm)
    margin = 2
    y_detect = min(cy + dist_voxels, ny - 1)
    
    # Surface signal (top 0.5mm)
    z_depth = max(int(0.5 / unit_mm), 1)
    
    x_start = max(cx - margin, 0)
    x_end = min(cx + margin, nx)
    y_start = max(y_detect - margin, 0)
    y_end = min(y_detect + margin, ny)
    
    signal = np.sum(flux[x_start:x_end, y_start:y_end, 0:z_depth])
    
    return float(signal)


# =============================================================================
# OPTIMIZATION OBJECTIVE 1: OPTIMAL SINGLE WAVELENGTH
# =============================================================================

def find_optimal_wavelength(wavelengths=None, skin_types=None, sdd_mm=5.0, 
                            nphoton=1e6, test_mode=False):
    """
    Find the optimal wavelength that minimizes Perfusion Index variance
    across all Fitzpatrick skin types.
    
    Parameters:
    -----------
    wavelengths : list
        Wavelengths to test (default: 500-1000nm, 10nm steps)
    skin_types : list
        Skin types to test (default: all I-VI)
    sdd_mm : float
        Source-Detector Distance
    nphoton : float
        Photons per simulation
    test_mode : bool
        Use reduced photons
        
    Returns:
    --------
    result : dict
        Contains optimal wavelength and analysis data
    """
    if wavelengths is None:
        wavelengths = get_fine_wavelength_range(500, 1000, 1)  # Fine 1nm resolution
    
    if skin_types is None:
        skin_types = list(FITZPATRICK_TYPES.keys())
    
    print("=" * 70)
    print("OPTIMIZATION: Finding Optimal Wavelength for Skin-Tone Independence")
    print("=" * 70)
    print(f"Wavelengths: {len(wavelengths)} ({min(wavelengths)}-{max(wavelengths)} nm)")
    print(f"Skin Types: {skin_types}")
    print(f"SDD: {sdd_mm} mm")
    print(f"Test Mode: {test_mode}")
    print("=" * 70)
    
    # Store results
    pi_data = {}  # {wavelength: {skin_type: PI}}
    variance_by_wavelength = {}
    
    for wl in wavelengths:
        print(f"\n[λ = {wl} nm]")
        pi_data[wl] = {}
        pi_values = []
        
        for st in skin_types:
            result = run_pulsatile_simulation_advanced(
                wl, st, spo2=0.98, sdd_mm=sdd_mm,
                nphoton=nphoton, test_mode=test_mode
            )
            
            if result['success']:
                pi = result['perfusion_index']
                pi_data[wl][st] = pi
                pi_values.append(pi)
                print(f"  Type {st}: PI = {pi:.4f}")
            else:
                print(f"  Type {st}: FAILED")
        
        # Calculate variance across skin types
        if len(pi_values) >= 2:
            variance = np.var(pi_values)
            variance_by_wavelength[wl] = variance
            print(f"  → Variance: {variance:.6f}")
    
    # Find optimal wavelength (minimum variance)
    if variance_by_wavelength:
        optimal_wl = min(variance_by_wavelength, key=variance_by_wavelength.get)
        min_variance = variance_by_wavelength[optimal_wl]
    else:
        optimal_wl = wavelengths[0]
        min_variance = float('inf')
    
    # Calculate melanin independence score (0-100, higher is better)
    max_var = max(variance_by_wavelength.values()) if variance_by_wavelength else 1
    independence_scores = {
        wl: 100 * (1 - var / max_var) if max_var > 0 else 100
        for wl, var in variance_by_wavelength.items()
    }
    
    print("\n" + "=" * 70)
    print(f"RESULT: Optimal Wavelength = {optimal_wl} nm")
    print(f"        Minimum PI Variance = {min_variance:.6f}")
    print(f"        Independence Score = {independence_scores.get(optimal_wl, 0):.1f}/100")
    print("=" * 70)
    
    return {
        'optimal_wavelength': optimal_wl,
        'min_variance': min_variance,
        'variance_by_wavelength': variance_by_wavelength,
        'pi_by_wavelength_skintype': pi_data,
        'independence_scores': independence_scores,
        'parameters': {
            'wavelengths': wavelengths,
            'skin_types': skin_types,
            'sdd_mm': sdd_mm,
            'nphoton': nphoton
        }
    }


# =============================================================================
# OPTIMIZATION OBJECTIVE 2: OPTIMAL SpO2 WAVELENGTH PAIR
# =============================================================================

def find_optimal_spo2_pair(wavelengths=None, skin_types=None, spo2_levels=None,
                            sdd_mm=5.0, nphoton=1e6, test_mode=False):
    """
    Find the optimal wavelength pair (λA, λB) that minimizes skin-tone
    deviation in the Ratio-of-Ratios (RoR) calibration curve.
    
    The goal is to find a pair where the RoR at a given SpO2 is IDENTICAL
    across all skin types.
    
    Parameters:
    -----------
    wavelengths : list
        Candidate wavelengths (default: extended set)
    skin_types : list
        Skin types to test
    spo2_levels : list
        SpO2 levels for calibration curve
    sdd_mm : float
        Source-Detector Distance
        
    Returns:
    --------
    result : dict
        Contains optimal pair and analysis data
    """
    if wavelengths is None:
        # Use strategic wavelengths for SpO2 (spanning isosbestic and beyond)
        wavelengths = [530, 590, 630, 660, 700, 750, 810, 850, 880, 940, 970]
    
    if skin_types is None:
        skin_types = list(FITZPATRICK_TYPES.keys())
    
    if spo2_levels is None:
        spo2_levels = [0.70, 0.80, 0.90, 0.95, 0.98]
    
    print("=" * 70)
    print("OPTIMIZATION: Finding Optimal SpO2 Wavelength Pair")
    print("=" * 70)
    print(f"Candidate Wavelengths: {wavelengths}")
    print(f"SpO2 Levels: {spo2_levels}")
    print("=" * 70)
    
    # Step 1: Run simulations for all wavelength × skin_type × SpO2 combinations
    ac_dc_data = {}  # {wavelength: {skin_type: {spo2: (ac, dc)}}}
    
    for wl in wavelengths:
        print(f"\n[λ = {wl} nm]")
        ac_dc_data[wl] = {}
        
        for st in skin_types:
            ac_dc_data[wl][st] = {}
            
            for spo2 in spo2_levels:
                result = run_pulsatile_simulation_advanced(
                    wl, st, spo2=spo2, sdd_mm=sdd_mm,
                    nphoton=nphoton, test_mode=test_mode
                )
                
                if result['success']:
                    ac_dc_data[wl][st][spo2] = (result['ac_signal'], result['dc_signal'])
                else:
                    ac_dc_data[wl][st][spo2] = (0, 0)
            
            print(f"  Type {st}: done")
    
    # Step 2: Evaluate all wavelength pairs
    wavelength_pairs = list(combinations(wavelengths, 2))
    pair_scores = {}
    ror_data = {}
    
    print(f"\nEvaluating {len(wavelength_pairs)} wavelength pairs...")
    
    for wl_a, wl_b in wavelength_pairs:
        # For each pair, calculate RoR at each SpO2 for each skin type
        ror_by_skin_spo2 = {}  # {skin_type: {spo2: RoR}}
        
        for st in skin_types:
            ror_by_skin_spo2[st] = {}
            
            for spo2 in spo2_levels:
                ac_a, dc_a = ac_dc_data[wl_a][st][spo2]
                ac_b, dc_b = ac_dc_data[wl_b][st][spo2]
                
                ror = calculate_ratio_of_ratios(ac_a, dc_a, ac_b, dc_b)
                ror_by_skin_spo2[st][spo2] = ror
        
        # Calculate deviation: variance of RoR across skin types at each SpO2
        total_deviation = 0
        for spo2 in spo2_levels:
            ror_values = [ror_by_skin_spo2[st][spo2] for st in skin_types]
            if len(ror_values) >= 2 and all(v > 0 for v in ror_values):
                # Use coefficient of variation for scale-independence
                mean_ror = np.mean(ror_values)
                if mean_ror > 0:
                    cv = np.std(ror_values) / mean_ror
                    total_deviation += cv
        
        pair_scores[(wl_a, wl_b)] = total_deviation
        ror_data[(wl_a, wl_b)] = ror_by_skin_spo2
    
    # Find optimal pair (minimum total deviation)
    optimal_pair = min(pair_scores, key=pair_scores.get)
    min_deviation = pair_scores[optimal_pair]
    
    # Standard pair comparison
    standard_pair = (660, 940)
    if standard_pair in pair_scores:
        standard_deviation = pair_scores[standard_pair]
    else:
        standard_deviation = float('inf')
    
    improvement = ((standard_deviation - min_deviation) / standard_deviation * 100 
                   if standard_deviation > 0 else 0)
    
    print("\n" + "=" * 70)
    print(f"RESULT: Optimal Wavelength Pair = {optimal_pair[0]} nm, {optimal_pair[1]} nm")
    print(f"        Total RoR Deviation = {min_deviation:.6f}")
    print(f"        Standard (660/940) Deviation = {standard_deviation:.6f}")
    print(f"        Improvement = {improvement:.1f}%")
    print("=" * 70)
    
    return {
        'optimal_pair': optimal_pair,
        'optimal_deviation': min_deviation,
        'standard_pair': standard_pair,
        'standard_deviation': standard_deviation,
        'improvement_percent': improvement,
        'pair_scores': {f"{p[0]}/{p[1]}": s for p, s in pair_scores.items()},
        'ror_data': {
            f"{p[0]}/{p[1]}": {
                st: {str(spo2): ror for spo2, ror in spo2_data.items()}
                for st, spo2_data in ror_by_skin.items()
            }
            for p, ror_by_skin in ror_data.items()
        },
        'parameters': {
            'wavelengths': wavelengths,
            'skin_types': skin_types,
            'spo2_levels': spo2_levels,
            'sdd_mm': sdd_mm
        }
    }


# =============================================================================
# OPTIMIZATION OBJECTIVE 3: SDD OPTIMIZATION
# =============================================================================

def optimize_sdd(wavelength, skin_types=None, sdd_range=None,
                 nphoton=1e6, test_mode=False):
    """
    Find the optimal Source-Detector Distance that maximizes:
    1. Signal strength (DC amplitude)
    2. Skin-tone independence (low PI variance)
    
    Parameters:
    -----------
    wavelength : int
        Wavelength to optimize for
    skin_types : list
        Skin types to test
    sdd_range : list
        SDD values to test in mm
        
    Returns:
    --------
    result : dict
        Contains optimal SDD and analysis data
    """
    if skin_types is None:
        skin_types = list(FITZPATRICK_TYPES.keys())
    
    if sdd_range is None:
        sdd_range = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    
    print("=" * 70)
    print(f"OPTIMIZATION: Finding Optimal SDD for λ = {wavelength} nm")
    print("=" * 70)
    print(f"SDD Range: {sdd_range} mm")
    print("=" * 70)
    
    signal_by_sdd = {}  # {sdd: {skin_type: dc_signal}}
    pi_by_sdd = {}  # {sdd: {skin_type: pi}}
    
    for sdd in sdd_range:
        print(f"\n[SDD = {sdd} mm]")
        signal_by_sdd[sdd] = {}
        pi_by_sdd[sdd] = {}
        
        for st in skin_types:
            result = run_pulsatile_simulation_advanced(
                wavelength, st, spo2=0.98, sdd_mm=sdd,
                nphoton=nphoton, test_mode=test_mode
            )
            
            if result['success']:
                signal_by_sdd[sdd][st] = result['dc_signal']
                pi_by_sdd[sdd][st] = result['perfusion_index']
                print(f"  Type {st}: DC={result['dc_signal']:.4f}, PI={result['perfusion_index']:.4f}")
    
    # Calculate metrics for each SDD
    sdd_metrics = {}
    for sdd in sdd_range:
        if sdd in signal_by_sdd:
            signals = list(signal_by_sdd[sdd].values())
            pis = list(pi_by_sdd[sdd].values())
            
            if len(signals) >= 2:
                mean_signal = np.mean(signals)
                pi_variance = np.var(pis)
                
                # Combined score: high signal + low variance
                # Normalize both components
                sdd_metrics[sdd] = {
                    'mean_signal': mean_signal,
                    'pi_variance': pi_variance,
                    'signal_by_skin': signal_by_sdd[sdd],
                    'pi_by_skin': pi_by_sdd[sdd]
                }
    
    # Find optimal SDD using a combined objective
    # We want high signal but low variance
    if sdd_metrics:
        max_signal = max(m['mean_signal'] for m in sdd_metrics.values())
        max_variance = max(m['pi_variance'] for m in sdd_metrics.values())
        
        for sdd, metrics in sdd_metrics.items():
            # Normalized scores (0-1)
            signal_score = metrics['mean_signal'] / max_signal if max_signal > 0 else 0
            variance_score = 1 - (metrics['pi_variance'] / max_variance) if max_variance > 0 else 1
            
            # Combined score (equal weights)
            metrics['combined_score'] = 0.5 * signal_score + 0.5 * variance_score
        
        optimal_sdd = max(sdd_metrics, key=lambda x: sdd_metrics[x]['combined_score'])
    else:
        optimal_sdd = sdd_range[len(sdd_range) // 2]
    
    print("\n" + "=" * 70)
    print(f"RESULT: Optimal SDD = {optimal_sdd} mm")
    if optimal_sdd in sdd_metrics:
        print(f"        Mean Signal = {sdd_metrics[optimal_sdd]['mean_signal']:.4f}")
        print(f"        PI Variance = {sdd_metrics[optimal_sdd]['pi_variance']:.6f}")
        print(f"        Combined Score = {sdd_metrics[optimal_sdd]['combined_score']:.2f}")
    print("=" * 70)
    
    return {
        'wavelength': wavelength,
        'optimal_sdd': optimal_sdd,
        'sdd_metrics': sdd_metrics,
        'parameters': {
            'wavelength': wavelength,
            'skin_types': skin_types,
            'sdd_range': sdd_range
        }
    }


# =============================================================================
# MULTI-DIMENSIONAL CO-OPTIMIZATION
# =============================================================================

def co_optimize_wavelength_sdd(wavelengths=None, sdd_range=None, skin_types=None,
                                nphoton=1e6, test_mode=False):
    """
    Joint optimization of wavelength and SDD.
    
    Finds the (λ, SDD) pair that maximizes signal quality while
    minimizing skin-tone dependence.
    
    Returns:
    --------
    result : dict
        Contains optimal (λ, SDD) and 2D optimization surface
    """
    if wavelengths is None:
        wavelengths = [600, 660, 750, 810, 850, 880, 940]
    
    if sdd_range is None:
        sdd_range = [2, 3, 4, 5, 6, 8, 10]
    
    if skin_types is None:
        skin_types = list(FITZPATRICK_TYPES.keys())
    
    print("=" * 70)
    print("CO-OPTIMIZATION: Wavelength × SDD Joint Optimization")
    print("=" * 70)
    print(f"Wavelengths: {wavelengths}")
    print(f"SDD Range: {sdd_range}")
    print(f"Total combinations: {len(wavelengths) * len(sdd_range)}")
    print("=" * 70)
    
    # 2D grid of results
    optimization_surface = {}  # {(wl, sdd): metrics}
    
    for wl in wavelengths:
        for sdd in sdd_range:
            print(f"\n[λ = {wl} nm, SDD = {sdd} mm]")
            
            signals = []
            pis = []
            
            for st in skin_types:
                result = run_pulsatile_simulation_advanced(
                    wl, st, spo2=0.98, sdd_mm=sdd,
                    nphoton=nphoton, test_mode=test_mode
                )
                
                if result['success']:
                    signals.append(result['dc_signal'])
                    pis.append(result['perfusion_index'])
            
            if len(signals) >= 2:
                optimization_surface[(wl, sdd)] = {
                    'mean_signal': np.mean(signals),
                    'pi_variance': np.var(pis),
                    'min_signal': np.min(signals),
                    'signals': signals,
                    'pis': pis
                }
                print(f"  Mean Signal: {np.mean(signals):.4f}, PI Var: {np.var(pis):.6f}")
    
    # Calculate combined scores
    if optimization_surface:
        max_signal = max(m['mean_signal'] for m in optimization_surface.values())
        max_variance = max(m['pi_variance'] for m in optimization_surface.values())
        
        for key, metrics in optimization_surface.items():
            signal_score = metrics['mean_signal'] / max_signal if max_signal > 0 else 0
            variance_score = 1 - (metrics['pi_variance'] / max_variance) if max_variance > 0 else 1
            
            # Weight signal slightly more (we need usable signal)
            metrics['combined_score'] = 0.4 * signal_score + 0.6 * variance_score
        
        # Find optimal combination
        optimal_key = max(optimization_surface, key=lambda x: optimization_surface[x]['combined_score'])
        optimal_wl, optimal_sdd = optimal_key
    else:
        optimal_wl = wavelengths[len(wavelengths) // 2]
        optimal_sdd = sdd_range[len(sdd_range) // 2]
    
    print("\n" + "=" * 70)
    print(f"RESULT: Optimal Configuration")
    print(f"        Wavelength = {optimal_wl} nm")
    print(f"        SDD = {optimal_sdd} mm")
    if (optimal_wl, optimal_sdd) in optimization_surface:
        opt = optimization_surface[(optimal_wl, optimal_sdd)]
        print(f"        Combined Score = {opt['combined_score']:.3f}")
    print("=" * 70)
    
    # Format surface for JSON serialization
    surface_json = {
        f"{wl}/{sdd}": {
            'wavelength': wl,
            'sdd': sdd,
            **{k: v if not isinstance(v, list) else v for k, v in metrics.items()}
        }
        for (wl, sdd), metrics in optimization_surface.items()
    }
    
    return {
        'optimal_wavelength': optimal_wl,
        'optimal_sdd': optimal_sdd,
        'combined_score': optimization_surface.get((optimal_wl, optimal_sdd), {}).get('combined_score', 0),
        'optimization_surface': surface_json,
        'parameters': {
            'wavelengths': wavelengths,
            'sdd_range': sdd_range,
            'skin_types': skin_types
        }
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_optimization_results(results, output_dir='ppg_results/optimization'):
    """Save optimization results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = os.path.join(output_dir, f'optimization_results_{timestamp}.json')
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results_json = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to: {filename}")
    return filename


# =============================================================================
# MAIN - RUN FULL OPTIMIZATION
# =============================================================================

def run_full_optimization(test_mode=True, output_dir='ppg_results/optimization'):
    """
    Run complete optimization pipeline.
    
    Parameters:
    -----------
    test_mode : bool
        Use reduced photons for faster testing
    output_dir : str
        Output directory for results
    """
    print("\n" + "=" * 70)
    print("PPG SENSOR OPTIMIZATION: FULL PIPELINE")
    print("=" * 70)
    print(f"Test Mode: {test_mode}")
    print(f"Output: {output_dir}")
    print("=" * 70 + "\n")
    
    all_results = {}
    
    # 1. Find optimal single wavelength
    print("\n>>> PHASE 1: Optimal Wavelength <<<\n")
    wavelength_result = find_optimal_wavelength(
        wavelengths=get_fine_wavelength_range(500, 1000, 50) if test_mode else get_fine_wavelength_range(500, 1000, 20),
        test_mode=test_mode
    )
    all_results['optimal_wavelength'] = wavelength_result
    
    # 2. Find optimal SpO2 pair
    print("\n>>> PHASE 2: Optimal SpO2 Wavelength Pair <<<\n")
    spo2_result = find_optimal_spo2_pair(
        wavelengths=[530, 660, 810, 880, 940] if test_mode else None,
        spo2_levels=[0.80, 0.90, 0.98] if test_mode else None,
        test_mode=test_mode
    )
    all_results['optimal_spo2_pair'] = spo2_result
    
    # 3. Optimize SDD for optimal wavelength
    print("\n>>> PHASE 3: SDD Optimization <<<\n")
    sdd_result = optimize_sdd(
        wavelength=wavelength_result['optimal_wavelength'],
        sdd_range=[3, 5, 7, 10] if test_mode else None,
        test_mode=test_mode
    )
    all_results['sdd_optimization'] = sdd_result
    
    # 4. Co-optimization
    print("\n>>> PHASE 4: Joint Wavelength-SDD Optimization <<<\n")
    co_opt_result = co_optimize_wavelength_sdd(
        wavelengths=[660, 810, 880, 940] if test_mode else None,
        sdd_range=[3, 5, 7] if test_mode else None,
        test_mode=test_mode
    )
    all_results['co_optimization'] = co_opt_result
    
    # Save all results
    save_optimization_results(all_results, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"1. Optimal Single Wavelength: {wavelength_result['optimal_wavelength']} nm")
    print(f"   (PI Variance: {wavelength_result['min_variance']:.6f})")
    print(f"\n2. Optimal SpO2 Pair: {spo2_result['optimal_pair'][0]}/{spo2_result['optimal_pair'][1]} nm")
    print(f"   (Improvement over 660/940: {spo2_result['improvement_percent']:.1f}%)")
    print(f"\n3. Optimal SDD: {sdd_result['optimal_sdd']} mm")
    print(f"   (for λ = {sdd_result['wavelength']} nm)")
    print(f"\n4. Co-Optimized Config: λ = {co_opt_result['optimal_wavelength']} nm, SDD = {co_opt_result['optimal_sdd']} mm")
    print("=" * 70)
    
    return all_results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PPG Sensor Optimization for Skin-Tone Independence'
    )
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (reduced photons)')
    parser.add_argument('--full', action='store_true',
                        help='Run full optimization (all phases)')
    parser.add_argument('--wavelength', action='store_true',
                        help='Run only wavelength optimization')
    parser.add_argument('--spo2', action='store_true',
                        help='Run only SpO2 pair optimization')
    parser.add_argument('--sdd', action='store_true',
                        help='Run only SDD optimization')
    parser.add_argument('--co-opt', action='store_true',
                        help='Run only co-optimization')
    parser.add_argument('--output-dir', type=str, default='ppg_results/optimization',
                        help='Output directory')
    
    args = parser.parse_args()
    
    test_mode = args.test
    
    if args.full or not any([args.wavelength, args.spo2, args.sdd, args.co_opt]):
        run_full_optimization(test_mode=test_mode, output_dir=args.output_dir)
    else:
        if args.wavelength:
            result = find_optimal_wavelength(test_mode=test_mode)
            save_optimization_results({'optimal_wavelength': result}, args.output_dir)
        
        if args.spo2:
            result = find_optimal_spo2_pair(test_mode=test_mode)
            save_optimization_results({'optimal_spo2_pair': result}, args.output_dir)
        
        if args.sdd:
            result = optimize_sdd(wavelength=880, test_mode=test_mode)
            save_optimization_results({'sdd_optimization': result}, args.output_dir)
        
        if args.co_opt:
            result = co_optimize_wavelength_sdd(test_mode=test_mode)
            save_optimization_results({'co_optimization': result}, args.output_dir)
    
    print("\n[OK] Optimization complete!")
