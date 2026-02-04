#!/usr/bin/env python3
"""
PPG Simulation Engine: Melanin Bias Analysis for Pulse Oximetry

Monte Carlo simulation using MCX/pmcx to analyze how melanin (skin pigmentation)
affects PPG signal quality across different wavelengths and Fitzpatrick skin types.

This simulation computes:
- Intensity vs Depth profiles for each wavelength & skin type
- Signal attenuation due to melanin
- Penetration depth analysis

Author: Generated for PPG optimization study
"""

import numpy as np
import pmcx
import json
import os
from datetime import datetime

# =============================================================================
# FITZPATRICK SKIN TYPE DEFINITIONS
# =============================================================================
# Melanin volume fraction in epidermis for each skin type

FITZPATRICK_TYPES = {
    'I':   {'name': 'Very Fair',       'melanin_fraction': 0.010, 'melanin_label': 'Type I'},
    'II':  {'name': 'Fair',            'melanin_fraction': 0.020, 'melanin_label': 'Type II'},
    'III': {'name': 'Medium',          'melanin_fraction': 0.035, 'melanin_label': 'Type III'},
    'IV':  {'name': 'Olive',           'melanin_fraction': 0.060, 'melanin_label': 'Type IV'},
    'V':   {'name': 'Brown',           'melanin_fraction': 0.100, 'melanin_label': 'Type V'},
    'VI':  {'name': 'Dark Brown/Black','melanin_fraction': 0.150, 'melanin_label': 'Type VI'},
}

# =============================================================================
# WAVELENGTH RANGE FOR SPECTRAL SWEEP
# =============================================================================

def get_wavelength_range(mode='full'):
    """
    Get wavelengths for simulation.
    
    Parameters:
    -----------
    mode : str
        'full' - Full spectral sweep 500-1000 nm (every 50 nm)
        'standard' - Standard PPG wavelengths only (660, 880, 940 nm)
        'extended' - Common PPG wavelengths (530, 660, 880, 940, 1050 nm)
    
    Returns:
    --------
    wavelengths : list
        List of wavelengths in nm
    """
    if mode == 'full':
        return list(range(500, 1001, 1))  # 500, 501, 502, ..., 1000 nm
    elif mode == 'standard':
        return [660, 880, 940]
    elif mode == 'extended':
        return [530, 660, 810, 880, 940, 1050]
    elif mode == 'optimization':
        return list(range(500, 1001, 10))  # Fine resolution for optimization
    elif mode == 'optimization_fine':
        return list(range(500, 1001, 5))   # Very fine resolution (5nm)
    else:
        return [660, 880, 940]  # Default


# =============================================================================
# OPTICAL PROPERTIES (WAVELENGTH & MELANIN DEPENDENT)
# =============================================================================

def melanin_absorption(wavelength_nm, melanin_volume_fraction):
    """
    Calculate melanin absorption coefficient.
    
    Uses Jacques (1998) formula: μa_melanin = 6.6e11 * λ^(-3.33) [1/cm]
    Then scales by volume fraction.
    
    Parameters:
    -----------
    wavelength_nm : float
        Wavelength in nanometers
    melanin_volume_fraction : float
        Volume fraction of melanin in epidermis (0.01-0.15)
    
    Returns:
    --------
    mua : float
        Absorption coefficient in 1/mm
    """
    # Jacques melanin absorption formula (converted to 1/mm)
    # μa = 6.6e11 * λ^(-3.33) [1/cm] -> multiply by 0.1 for 1/mm
    mua_pure_melanin = 6.6e11 * (wavelength_nm ** -3.33) * 0.1  # 1/mm
    
    # Scale by volume fraction
    mua = melanin_volume_fraction * mua_pure_melanin
    return mua


def hemoglobin_absorption(wavelength_nm, blood_volume_fraction=0.02, oxygen_saturation=0.98):
    """
    Calculate blood/hemoglobin absorption coefficient.
    
    Uses simplified approximation based on published Hb/HbO2 spectra.
    
    Parameters:
    -----------
    wavelength_nm : float
        Wavelength in nanometers
    blood_volume_fraction : float
        Volume fraction of blood in tissue
    oxygen_saturation : float
        Oxygen saturation (0-1)
    
    Returns:
    --------
    mua : float
        Absorption coefficient in 1/mm
    """
    # Simplified hemoglobin absorption model (1/mm)
    # Based on characteristic spectra of HbO2 and Hb
    
    # Approximate molar extinction coefficients at key wavelengths
    # Interpolate for other wavelengths
    
    # Key wavelengths and approximate absorption (1/cm per mM)
    hbo2_data = {
        500: 2.4, 530: 4.2, 550: 5.3, 575: 4.8, 600: 0.3,
        630: 0.1, 660: 0.08, 700: 0.05, 750: 0.1, 800: 0.2,
        850: 0.25, 880: 0.3, 900: 0.35, 940: 0.4, 1000: 0.5
    }
    
    hb_data = {
        500: 1.8, 530: 2.5, 550: 5.4, 575: 3.5, 600: 1.2,
        630: 0.8, 660: 0.8, 700: 0.3, 750: 0.2, 800: 0.2,
        850: 0.25, 880: 0.3, 900: 0.35, 940: 0.4, 1000: 0.5
    }
    
    # Interpolate
    wavelengths = sorted(hbo2_data.keys())
    
    # Find bracketing wavelengths
    wl = wavelength_nm
    if wl <= wavelengths[0]:
        mua_hbo2 = hbo2_data[wavelengths[0]]
        mua_hb = hb_data[wavelengths[0]]
    elif wl >= wavelengths[-1]:
        mua_hbo2 = hbo2_data[wavelengths[-1]]
        mua_hb = hb_data[wavelengths[-1]]
    else:
        # Linear interpolation
        for i in range(len(wavelengths) - 1):
            if wavelengths[i] <= wl <= wavelengths[i+1]:
                w1, w2 = wavelengths[i], wavelengths[i+1]
                t = (wl - w1) / (w2 - w1)
                mua_hbo2 = hbo2_data[w1] + t * (hbo2_data[w2] - hbo2_data[w1])
                mua_hb = hb_data[w1] + t * (hb_data[w2] - hb_data[w1])
                break
    
    # Combine based on oxygen saturation
    mua_blood = oxygen_saturation * mua_hbo2 + (1 - oxygen_saturation) * mua_hb
    
    # Scale by blood volume fraction and convert to 1/mm
    mua = blood_volume_fraction * mua_blood * 0.1  # 1/mm
    
    return mua


def tissue_scattering(wavelength_nm, tissue_type='dermis'):
    """
    Calculate reduced scattering coefficient using Mie theory approximation.
    
    μs' = a * (λ/500)^(-b)  where a, b are tissue-specific
    
    Parameters:
    -----------
    wavelength_nm : float
        Wavelength in nanometers
    tissue_type : str
        Type of tissue
    
    Returns:
    --------
    mus : float
        Scattering coefficient in 1/mm
    """
    # Tissue-specific scattering parameters (a in 1/mm at 500nm, b is power)
    scattering_params = {
        'stratum_corneum': {'a': 100.0, 'b': 1.0},
        'epidermis':       {'a': 50.0,  'b': 1.2},
        'dermis':          {'a': 25.0,  'b': 1.5},
        'subcutis':        {'a': 15.0,  'b': 1.0},
    }
    
    params = scattering_params.get(tissue_type, {'a': 25.0, 'b': 1.0})
    mus = params['a'] * ((wavelength_nm / 500.0) ** -params['b'])
    
    return mus


def get_optical_properties(wavelength_nm, melanin_fraction=0.01, blood_fraction=0.02):
    """
    Calculate optical properties for 5-layer skin model.
    
    Parameters:
    -----------
    wavelength_nm : float
        Wavelength in nanometers
    melanin_fraction : float
        Melanin volume fraction in epidermis
    blood_fraction : float
        Blood volume fraction in dermis
    
    Returns:
    --------
    prop : numpy array
        Optical properties [mua, mus, g, n] for each tissue layer
    """
    
    # Layer 0: Background (air)
    background = [0.0, 0.0, 1.0, 1.0]
    
    # Layer 1: Stratum corneum (15 μm) - minimal absorption
    sc_mua = 0.01 + 0.1 * ((wavelength_nm / 500) ** -1)  # Slight baseline
    sc_mus = tissue_scattering(wavelength_nm, 'stratum_corneum')
    stratum_corneum = [sc_mua, sc_mus, 0.9, 1.55]
    
    # Layer 2: Epidermis (100 μm) - MELANIN IS HERE
    epid_mua_base = 0.05
    epid_mua_melanin = melanin_absorption(wavelength_nm, melanin_fraction)
    epid_mua = epid_mua_base + epid_mua_melanin
    epid_mus = tissue_scattering(wavelength_nm, 'epidermis')
    epidermis = [epid_mua, epid_mus, 0.85, 1.44]
    
    # Layer 3: Papillary dermis (200 μm) - blood vessels
    derm_p_mua = 0.02 + hemoglobin_absorption(wavelength_nm, blood_fraction * 1.5)
    derm_p_mus = tissue_scattering(wavelength_nm, 'dermis')
    papillary_dermis = [derm_p_mua, derm_p_mus, 0.90, 1.40]
    
    # Layer 4: Reticular dermis (1000 μm) - less blood
    derm_r_mua = 0.01 + hemoglobin_absorption(wavelength_nm, blood_fraction * 0.5)
    derm_r_mus = tissue_scattering(wavelength_nm, 'dermis') * 0.8
    reticular_dermis = [derm_r_mua, derm_r_mus, 0.90, 1.40]
    
    # Layer 5: Subcutaneous tissue - fat
    subcut_mua = 0.005 + hemoglobin_absorption(wavelength_nm, blood_fraction * 0.3)
    subcut_mus = tissue_scattering(wavelength_nm, 'subcutis')
    subcutis = [subcut_mua, subcut_mus, 0.80, 1.44]
    
    prop = np.array([
        background,
        stratum_corneum,
        epidermis,
        papillary_dermis,
        reticular_dermis,
        subcutis
    ], dtype=np.float32)
    
    return prop


def create_skin_volume(nx=100, ny=100, nz=200, unit_um=10):
    """
    Create 5-layer skin volume.
    
    Layer thicknesses (with given resolution):
    - Stratum corneum: 15 μm
    - Epidermis: 100 μm
    - Papillary dermis: 200 μm
    - Reticular dermis: 1000 μm
    - Subcutis: remaining
    
    Parameters:
    -----------
    nx, ny, nz : int
        Volume dimensions in voxels
    unit_um : float
        Voxel size in micrometers
    
    Returns:
    --------
    vol : numpy array (uint8)
        Volume with tissue labels 1-5
    layer_boundaries : dict
        Z-indices where each layer ends
    """
    vol = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    # Calculate layer boundaries (in voxels)
    z_sc = int(15 / unit_um)       # Stratum corneum end
    z_ep = z_sc + int(100 / unit_um)   # Epidermis end
    z_pd = z_ep + int(200 / unit_um)   # Papillary dermis end
    z_rd = z_pd + int(1000 / unit_um)  # Reticular dermis end
    
    # Cap at volume size
    z_sc = min(z_sc, nz)
    z_ep = min(z_ep, nz)
    z_pd = min(z_pd, nz)
    z_rd = min(z_rd, nz)
    
    # Assign tissue labels
    vol[:, :, 0:z_sc] = 1           # Stratum corneum
    vol[:, :, z_sc:z_ep] = 2        # Epidermis
    vol[:, :, z_ep:z_pd] = 3        # Papillary dermis
    vol[:, :, z_pd:z_rd] = 4        # Reticular dermis
    if z_rd < nz:
        vol[:, :, z_rd:nz] = 5      # Subcutis
    
    layer_boundaries = {
        'stratum_corneum_end': z_sc,
        'epidermis_end': z_ep,
        'papillary_dermis_end': z_pd,
        'reticular_dermis_end': z_rd,
        'unit_um': unit_um
    }
    
    return vol, layer_boundaries



def run_pulsatile_simulation(wavelength_nm, skin_type='I', nphoton=1e6, test_mode=False):
    """
    Run simulation for both Systolic (high blood) and Diastolic (low blood) states
    to calculate AC/DC Perfusion Index.
    
    Parameters:
    -----------
    wavelength_nm : float
        Wavelength in nm
    skin_type : str
        Fitzpatrick skin type
    nphoton : float
        Number of photons
        
    Returns:
    --------
    result : dict
        Contains DC (diastolic), AC (systolic - diastolic), and PI
    """
    if test_mode:
        nphoton = 1e5
        
    # Standard Diastolic State (Baseline)
    # Blood volume fraction: ~2%
    res_diastolic = run_single_simulation(
        wavelength_nm, skin_type, nphoton, test_mode, 
        blood_fraction=0.02
    )
    
    # Systolic State (Pulsatile Increase)
    # Blood volume increases by ~5-10% relative (e.g., 0.02 -> 0.022)
    # This simulates the arrival of the pulse wave
    res_systolic = run_single_simulation(
        wavelength_nm, skin_type, nphoton, test_mode, 
        blood_fraction=0.022
    )
    
    # Extract Intensities (Total detected photon weight)
    # We use total absorbed energy or fluence at specific detector as proxy
    # For simplicity in this engine, we use total energy absorbed in DERMIS (where pulsating blood is)
    # relative to input, or raw flux if detector modeling was explicit.
    # Here we use 'penetration_depth' as a proxy for signal quality, but for PI we need INTENSITY.
    # Let's use mean intensity at a standard detector distance (e.g., 5mm).
    
    # Extract signal from flux at r=5mm
    dc_signal = extract_detector_signal(res_diastolic)
    sys_signal = extract_detector_signal(res_systolic)
    
    # AC = Systolic - Diastolic (In reflectance, more blood = LESS light)
    # So actually: AC = Diastolic - Systolic (magnitude of modulation)
    # But usually PI = (I_max - I_min) / I_mean
    # Let's define AC amplitude as absolute difference
    ac_signal = abs(dc_signal - sys_signal)
    
    pi = ac_signal / dc_signal if dc_signal > 0 else 0
    
    result = {
        'wavelength_nm': wavelength_nm,
        'skin_type': skin_type,
        'dc_signal': dc_signal,
        'ac_signal': ac_signal,
        'perfusion_index': pi,
        'skin_name': FITZPATRICK_TYPES[skin_type]['name']
    }
    
    return result

def extract_detector_signal(result, sdd_mm=5.0):
    """
    Extract light intensity at a specific Source-Detector Distance (SDD).
    """
    flux = result['flux']
    unit_mm = result['unit_mm']
    
    if flux.ndim == 4:
        flux = np.squeeze(flux)
        
    # Source is at center (nx//2, ny//2)
    nx, ny, _ = flux.shape
    cx, cy = nx // 2, ny // 2
    
    # Calculate radius indices for SDD
    # sdd_voxels = sdd_mm / unit_mm
    # We sum ring of detectors at this distance on surface (z=0 to small depth)
    
    # Simplified: Just take total volume absorption as 'signal' inverse
    # For meaningful PI, we need Remittance.
    # MCX outputs Flux (Fluence Rate). Remittance ~ Flux at boundary.
    # We will sample flux at surface at distance SDD.
    
    dist_voxels = int(sdd_mm / unit_mm)
    # Sample a small region at y = cy + dist, x = cx (simplified point detector)
    
    margin = 2
    y_detect = cy + dist_voxels
    
    if y_detect >= ny:
        y_detect = ny - 1
        
    # Surface signal (integrate top 0.5mm)
    z_depth = int(0.5 / unit_mm)
    
    signal = np.sum(flux[cx-margin:cx+margin, y_detect-margin:y_detect+margin, 0:z_depth])
    
    return float(signal)

def run_single_simulation(wavelength_nm, skin_type='I', nphoton=1e6, test_mode=False, blood_fraction=0.02):
    """
    Run Monte Carlo simulation for a single wavelength and skin type.
    """
    
    if test_mode:
        nphoton = 1e5
    
    # Get melanin fraction for this skin type
    melanin_fraction = FITZPATRICK_TYPES[skin_type]['melanin_fraction']
    
    # Volume dimensions
    nx, ny, nz = 100, 100, 200
    unit_um = 10  # 10 μm voxel
    unit_mm = unit_um / 1000.0
    
    # Create volume
    vol, layer_boundaries = create_skin_volume(nx, ny, nz, unit_um)
    
    # Get optical properties (WITH VARIABLE BLOOD FRACTION)
    prop = get_optical_properties(wavelength_nm, melanin_fraction, blood_fraction)
    
    # Configure simulation
    cfg = {
        'nphoton': int(nphoton),
        'vol': vol,
        'prop': prop,
        'tstart': 0,
        'tend': 5e-9,
        'tstep': 5e-9,
        'unitinmm': unit_mm,
        
        # Source: small disk (LED)
        'srctype': 'disk',
        'srcpos': [nx/2, ny/2, 0],
        'srcdir': [0, 0, 1],
        'srcparam1': [5, 0, 0, 0],  # 5 voxel radius = 50 μm
        
        # Options
        'isreflect': 1,
        'issrcfrom0': 1,
        'autopilot': 1,
        'gpuid': 1,
        'outputtype': 'fluence',
    }
    
    # Run simulation
    # print(f"  WL={wavelength_nm}nm, Type {skin_type}, Blood={blood_fraction:.3f}...", end=' ')
    
    try:
        result = pmcx.run(cfg)
        
        # Add metadata
        result['wavelength_nm'] = wavelength_nm
        result['skin_type'] = skin_type
        result['melanin_fraction'] = melanin_fraction
        result['skin_name'] = FITZPATRICK_TYPES[skin_type]['name']
        result['prop'] = prop
        result['layer_boundaries'] = layer_boundaries
        result['unit_mm'] = unit_mm
        result['nphoton'] = nphoton
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        raise



def extract_depth_profile(result):
    """
    Extract normalized intensity vs depth profile from simulation result.
    
    Parameters:
    -----------
    result : dict
        Simulation result
    
    Returns:
    --------
    depth_mm : numpy array
        Depth values in mm
    intensity : numpy array
        Normalized intensity values
    """
    flux = result['flux']
    unit_mm = result['unit_mm']
    
    # Handle 4D array
    if flux.ndim == 4:
        flux = np.squeeze(flux)
    
    nx, ny, nz = flux.shape
    
    # Average over central region
    margin = 20
    cx, cy = nx // 2, ny // 2
    central_flux = flux[cx-margin:cx+margin, cy-margin:cy+margin, :]
    depth_profile = np.mean(central_flux, axis=(0, 1))
    
    # Normalize to maximum
    max_val = np.max(depth_profile)
    if max_val > 0:
        intensity = depth_profile / max_val
    else:
        intensity = depth_profile
    
    # Create depth array
    depth_mm = np.arange(nz) * unit_mm
    
    return depth_mm, intensity


def calculate_penetration_depth(depth_mm, intensity):
    """
    Calculate 1/e penetration depth.
    
    Returns:
    --------
    pen_depth : float
        Penetration depth in mm (where intensity drops to 1/e)
    """
    threshold = 1.0 / np.e
    
    # Find where intensity first drops below threshold
    below_threshold = np.where(intensity < threshold)[0]
    
    if len(below_threshold) > 0:
        idx = below_threshold[0]
        # Interpolate for more accuracy
        if idx > 0:
            i1, i2 = intensity[idx-1], intensity[idx]
            d1, d2 = depth_mm[idx-1], depth_mm[idx]
            if i1 != i2:
                pen_depth = d1 + (threshold - i1) * (d2 - d1) / (i2 - i1)
            else:
                pen_depth = d1
        else:
            pen_depth = depth_mm[idx]
    else:
        pen_depth = depth_mm[-1]  # Didn't decay below threshold
    
    return pen_depth


def run_full_analysis(wavelengths=None, skin_types=None, nphoton=1e6, test_mode=False):
    """
    Run full simulation sweep across wavelengths and skin types.
    
    Parameters:
    -----------
    wavelengths : list
        List of wavelengths to simulate (default: full sweep)
    skin_types : list
        List of skin types (default: all I-VI)
    nphoton : float
        Number of photons per simulation
    test_mode : bool
        Use reduced photon count for testing
    
    Returns:
    --------
    results : dict
        Nested dict with results[wavelength][skin_type]
    """
    
    if wavelengths is None:
        wavelengths = get_wavelength_range('full')
    
    if skin_types is None:
        skin_types = list(FITZPATRICK_TYPES.keys())
    
    print("=" * 70)
    print("PPG SIMULATION: MELANIN BIAS ANALYSIS")
    print("=" * 70)
    print(f"Wavelengths: {wavelengths}")
    print(f"Skin Types: {skin_types}")
    print(f"Photons per simulation: {nphoton:.0e}")
    print(f"Total simulations: {len(wavelengths) * len(skin_types)}")
    print("=" * 70)
    
    results = {}
    
    for wl in wavelengths:
        print(f"\n[Wavelength: {wl} nm]")
        results[wl] = {}
        
        for st in skin_types:
            print(f"  Analysing Type {st}...", end=' ')
            
            # Run pulsatile simulation (AC/DC)
            sim_result = run_pulsatile_simulation(wl, st, nphoton, test_mode)
            print(f"PI={sim_result['perfusion_index']:.4f}")
            
            # Use the diastolic (baseline) state for depth profile analysis
            # We need to re-run or cache the full volume if we want depth profiles.
            # To save time, let's just use the PI data primarily.
            # But the GUI expects depth profiles.
            # Let's run a single simulation for depth profile (standard blood fraction)
            # This is slightly redundant but cleaner than managing huge result objects in memory.
            
            # Store PI results
            results[wl][st] = {
                'wavelength_nm': wl,
                'skin_type': st,
                'skin_name': FITZPATRICK_TYPES[st]['name'],
                'melanin_fraction': FITZPATRICK_TYPES[st]['melanin_fraction'],
                'perfusion_index': sim_result['perfusion_index'],
                'ac_signal': sim_result['ac_signal'],
                'dc_signal': sim_result['dc_signal'],
                'penetration_depth_mm': 0, # Placeholder, will fill if we do depth profile
            }
            
            # Optional: Get depth profile for legacy GUI support (Tab 1)
            # Just use the DC signal run (extracting depth info isn't part of run_pulsatile yet)
            # Ideally we refactor run_pulsatile to return the full volume object too.
            # For now, let's skip depth profile to be fast, OR re-enable it.
            # Let's do a quick approximation for depth:
            # We don't have the full flux volume here because run_pulsatile doesn't return it to save RAM.
            # Let's Modify run_pulsatile to return it? No, meaningful depth is DC state.
            
            # Simple solution: Minimal run for depth
            depth_res = run_single_simulation(wl, st, nphoton/5, test_mode) # Faster run for just depth shape
            depth_mm, intensity = extract_depth_profile(depth_res)
            pen_depth = calculate_penetration_depth(depth_mm, intensity)
            
            results[wl][st]['depth_mm'] = depth_mm.tolist()
            results[wl][st]['intensity'] = intensity.tolist()
            results[wl][st]['penetration_depth_mm'] = pen_depth
    
    return results


def save_results(results, output_dir='ppg_results'):
    """
    Save simulation results to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full results
    results_file = os.path.join(output_dir, 'data', f'simulation_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Create summary table
    summary = []
    for wl in results:
        for st in results[wl]:
            r = results[wl][st]
            summary.append({
                'wavelength_nm': r['wavelength_nm'],
                'skin_type': r['skin_type'],
                'skin_name': r['skin_name'],
                'melanin_fraction': r['melanin_fraction'],
                'penetration_depth_mm': r['penetration_depth_mm'],
            })
    
    summary_file = os.path.join(output_dir, 'data', f'summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    return results_file, summary_file


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PPG Simulation for Melanin Bias Analysis'
    )
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with reduced photons')
    parser.add_argument('--nphoton', type=float, default=1e6,
                        help='Number of photons per simulation')
    parser.add_argument('--mode', choices=['full', 'standard', 'extended'],
                        default='full', help='Wavelength selection mode')
    parser.add_argument('--output-dir', type=str, default='ppg_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Get wavelengths
    wavelengths = get_wavelength_range(args.mode)
    
    # Run analysis
    results = run_full_analysis(
        wavelengths=wavelengths,
        nphoton=args.nphoton,
        test_mode=args.test
    )
    
    # Save results
    save_results(results, args.output_dir)
    
    print("\n[OK] PPG simulation complete!")
