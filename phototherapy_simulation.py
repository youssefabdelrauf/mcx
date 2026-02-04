#!/usr/bin/env python3
"""
Phototherapy Light Penetration Simulation: Jaundiced vs Healthy Neonatal Skin

This script uses Monte Carlo eXtreme (MCX) via pmcx to simulate light penetration
through neonatal skin at phototherapy wavelength (460nm blue light).

Compares healthy skin vs jaundiced skin at various bilirubin concentrations
to optimize blue-light phototherapy for neonatal jaundice treatment.

Author: Generated for MCX phototherapy optimization study
"""

import numpy as np
import pmcx
import json
import os
import argparse
from datetime import datetime

# =============================================================================
# OPTICAL PROPERTIES AT 460nm (BLUE LIGHT PHOTOTHERAPY)
# =============================================================================
# Format: [mua (1/mm), mus (1/mm), g (anisotropy), n (refractive index)]
# Based on published tissue optics data for neonatal skin

def get_optical_properties(bilirubin_mg_dl=0.0):
    """
    Calculate optical properties for 4-layer neonatal skin model at 460nm.
    
    Parameters:
    -----------
    bilirubin_mg_dl : float
        Bilirubin concentration in mg/dL (typical range: 0-25 for neonates)
        Normal: < 1 mg/dL
        Moderate jaundice: 5-15 mg/dL
        Severe jaundice: 15-25+ mg/dL
    
    Returns:
    --------
    prop : numpy array
        Optical properties array [mua, mus, g, n] for each tissue type
    """
    
    # Bilirubin absorption coefficient at 460nm
    # Based on: molar extinction coeff ~55,000 L/(mol·cm), MW=584.66 g/mol
    # Converting to mm^-1 per mg/dL: approximately 0.025-0.03 mm^-1 per mg/dL
    BILIRUBIN_MUA_PER_MGDL = 0.025  # mm^-1 per mg/dL
    
    # Additional absorption from bilirubin
    bilirubin_absorption = BILIRUBIN_MUA_PER_MGDL * bilirubin_mg_dl
    
    # Baseline optical properties at 460nm (blue light)
    # Layer 0: Background/air (required by MCX)
    background = [0.0, 0.0, 1.0, 1.0]
    
    # Layer 1: Stratum corneum (10 μm thick)
    # Low absorption, moderate scattering
    stratum_corneum = [0.1, 50.0, 0.9, 1.5]
    
    # Layer 2: Epidermis (50-100 μm thick)
    # Contains melanin - absorption varies with skin type
    # Using moderate melanin for neonatal Caucasian skin
    epidermis = [0.5, 45.0, 0.8, 1.4]
    
    # Layer 3: Dermis (1-2 mm thick) - PRIMARY BILIRUBIN LOCATION
    # Contains blood vessels (hemoglobin absorption) + bilirubin in jaundice
    # Hemoglobin at 460nm has significant absorption
    dermis_base_mua = 0.35  # baseline from blood/hemoglobin
    dermis = [dermis_base_mua + bilirubin_absorption, 20.0, 0.9, 1.4]
    
    # Layer 4: Subcutaneous tissue (fat layer, 2+ mm)
    # Lower scattering, bilirubin also accumulates here
    subcutis_base_mua = 0.05
    subcutis = [subcutis_base_mua + bilirubin_absorption * 0.7, 12.0, 0.8, 1.4]
    
    # Combine all layers
    prop = np.array([
        background,        # Type 0: background/air
        stratum_corneum,   # Type 1
        epidermis,         # Type 2
        dermis,            # Type 3
        subcutis           # Type 4
    ], dtype=np.float32)
    
    return prop


def create_skin_volume(nx=100, ny=100, nz=150):
    """
    Create 4-layer neonatal skin volume.
    
    Layer thicknesses (in voxels, with 5 μm resolution):
    - Stratum corneum: 2 voxels (10 μm)
    - Epidermis: 16 voxels (80 μm)  
    - Dermis: 300 voxels (1.5 mm)
    - Subcutis: remaining depth
    
    Returns:
    --------
    vol : numpy array (uint8)
        Volume with tissue labels 1-4
    """
    vol = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    # Layer boundaries (z-direction, 0 is top surface)
    z_stratum_end = 2      # 10 μm
    z_epidermis_end = 18   # 90 μm total
    z_dermis_end = 318     # 1.59 mm total - but we cap at nz
    
    # Ensure boundaries don't exceed volume
    z_stratum_end = min(z_stratum_end, nz)
    z_epidermis_end = min(z_epidermis_end, nz)
    z_dermis_end = min(z_dermis_end, nz)
    
    # Assign tissue types
    vol[:, :, 0:z_stratum_end] = 1          # Stratum corneum
    vol[:, :, z_stratum_end:z_epidermis_end] = 2   # Epidermis
    vol[:, :, z_epidermis_end:z_dermis_end] = 3    # Dermis
    if z_dermis_end < nz:
        vol[:, :, z_dermis_end:nz] = 4      # Subcutis
    
    return vol


def run_simulation(bilirubin_mg_dl=0.0, nphoton=1e7, test_mode=False):
    """
    Run Monte Carlo simulation for a given bilirubin level.
    
    Parameters:
    -----------
    bilirubin_mg_dl : float
        Bilirubin concentration in mg/dL
    nphoton : float
        Number of photons to simulate
    test_mode : bool
        If True, use reduced photon count for quick testing
        
    Returns:
    --------
    result : dict
        Dictionary containing simulation results
    """
    
    if test_mode:
        nphoton = 1e5  # Reduced for quick testing
        print(f"[TEST MODE] Using {nphoton:.0e} photons")
    
    # Volume dimensions
    nx, ny, nz = 100, 100, 150
    unit_mm = 0.005  # 5 μm voxel size
    
    # Create the skin volume
    vol = create_skin_volume(nx, ny, nz)
    
    # Get optical properties for this bilirubin level
    prop = get_optical_properties(bilirubin_mg_dl)
    
    # Configure the simulation
    cfg = {
        'nphoton': int(nphoton),
        'vol': vol,
        'prop': prop,
        'tstart': 0,
        'tend': 5e-9,
        'tstep': 5e-9,
        'unitinmm': unit_mm,
        
        # Source configuration: disk illumination (phototherapy lamp)
        'srctype': 'disk',
        'srcpos': [nx/2, ny/2, 0],  # Center of top surface
        'srcdir': [0, 0, 1],        # Pointing into the tissue
        'srcparam1': [20, 0, 0, 0], # Disk radius = 20 voxels = 0.1 mm
        
        # Simulation options
        'isreflect': 1,     # Enable Fresnel reflection at boundaries
        'issrcfrom0': 1,    # Source starts at position
        'autopilot': 1,     # Auto-optimize GPU parameters
        'gpuid': 1,         # Use first GPU
        
        # Output energy deposition (absorbed energy)
        'outputtype': 'energy',
    }
    
    print(f"\n{'='*60}")
    print(f"Running simulation: Bilirubin = {bilirubin_mg_dl:.1f} mg/dL")
    print(f"Photons: {nphoton:.2e}")
    print(f"Volume: {nx}x{ny}x{nz} voxels ({nx*unit_mm:.2f}x{ny*unit_mm:.2f}x{nz*unit_mm:.2f} mm)")
    print(f"{'='*60}")
    
    # Print optical properties
    print("\nOptical Properties (mua, mus, g, n):")
    layer_names = ['Background', 'Stratum Corneum', 'Epidermis', 'Dermis', 'Subcutis']
    for i, name in enumerate(layer_names):
        print(f"  {name}: mua={prop[i,0]:.4f}, mus={prop[i,1]:.2f}, g={prop[i,2]:.2f}, n={prop[i,3]:.2f}")
    
    # Run simulation
    print("\nRunning Monte Carlo simulation...")
    try:
        result = pmcx.run(cfg)
        print("Simulation completed successfully!")
        
        # Add metadata to result
        result['bilirubin_mg_dl'] = bilirubin_mg_dl
        result['nphoton'] = nphoton
        result['prop'] = prop
        result['vol'] = vol
        result['unitinmm'] = unit_mm
        result['layer_thicknesses'] = {
            'stratum_corneum_um': 10,
            'epidermis_um': 80,
            'dermis_um': 1500,
            'total_depth_mm': nz * unit_mm
        }
        
        return result
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        raise


def analyze_results(result):
    """
    Analyze simulation results and extract key metrics.
    
    Parameters:
    -----------
    result : dict
        Simulation result from run_simulation()
        
    Returns:
    --------
    metrics : dict
        Dictionary of analysis metrics
    """
    flux = result['flux']
    vol = result['vol']
    unit_mm = result['unitinmm']
    
    # Handle 4D array (x, y, z, time) - squeeze time dimension
    if flux.ndim == 4:
        flux = np.squeeze(flux)  # Remove time dimension if single timestep
    
    nx, ny, nz = flux.shape
    
    # Get central slice for analysis
    center_x = nx // 2
    center_slice = flux[center_x, :, :]
    
    # Calculate depth profile (average over central region)
    margin = 10
    central_flux = flux[center_x-margin:center_x+margin, 
                        ny//2-margin:ny//2+margin, :]
    depth_profile = np.mean(central_flux, axis=(0, 1))
    
    # Calculate 1/e penetration depth
    max_flux = np.max(depth_profile)
    threshold = max_flux / np.e
    penetration_indices = np.where(depth_profile >= threshold)[0]
    if len(penetration_indices) > 0:
        penetration_depth_voxels = penetration_indices[-1]
        penetration_depth_mm = penetration_depth_voxels * unit_mm
    else:
        penetration_depth_mm = 0
    
    # Calculate energy deposited in each layer
    layer_absorption = {}
    layer_names = {1: 'stratum_corneum', 2: 'epidermis', 3: 'dermis', 4: 'subcutis'}
    total_energy = np.sum(flux)
    
    for label, name in layer_names.items():
        mask = (vol == label)
        layer_energy = np.sum(flux[mask])
        layer_absorption[name] = {
            'total_energy': float(layer_energy),
            'fraction': float(layer_energy / total_energy) if total_energy > 0 else 0
        }
    
    metrics = {
        'bilirubin_mg_dl': result['bilirubin_mg_dl'],
        'penetration_depth_mm': float(penetration_depth_mm),
        'total_absorbed_energy': float(total_energy),
        'layer_absorption': layer_absorption,
        'depth_profile': depth_profile.tolist(),
        'center_slice': center_slice,
    }
    
    return metrics


def run_comparison_study(bilirubin_levels=None, nphoton=1e7, test_mode=False):
    """
    Run simulations for multiple bilirubin levels and compare results.
    
    Parameters:
    -----------
    bilirubin_levels : list
        List of bilirubin concentrations to simulate
    nphoton : float
        Number of photons per simulation
    test_mode : bool
        If True, use reduced photon count
        
    Returns:
    --------
    all_results : list
        List of result dictionaries for each bilirubin level
    """
    
    if bilirubin_levels is None:
        # Default: healthy (0) and jaundiced at various levels
        bilirubin_levels = [0, 5, 10, 15, 20, 25]
    
    all_results = []
    
    for bili in bilirubin_levels:
        result = run_simulation(bili, nphoton=nphoton, test_mode=test_mode)
        metrics = analyze_results(result)
        all_results.append(metrics)
    
    return all_results


def save_results(results, output_dir='phototherapy_results'):
    """Save simulation results to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary metrics (without large arrays)
    summary = []
    for r in results:
        summary_entry = {
            'bilirubin_mg_dl': r['bilirubin_mg_dl'],
            'penetration_depth_mm': r['penetration_depth_mm'],
            'total_absorbed_energy': r['total_absorbed_energy'],
            'layer_absorption': r['layer_absorption']
        }
        summary.append(summary_entry)
    
    summary_file = os.path.join(output_dir, f'summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    # Save depth profiles
    depth_data = {str(r['bilirubin_mg_dl']): r['depth_profile'] for r in results}
    depth_file = os.path.join(output_dir, f'depth_profiles_{timestamp}.json')
    with open(depth_file, 'w') as f:
        json.dump(depth_data, f)
    print(f"Depth profiles saved to: {depth_file}")
    
    return summary_file, depth_file


def main():
    parser = argparse.ArgumentParser(
        description='Phototherapy Light Penetration Simulation for Neonatal Jaundice'
    )
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with reduced photon count')
    parser.add_argument('--nphoton', type=float, default=1e7,
                        help='Number of photons to simulate (default: 1e7)')
    parser.add_argument('--bilirubin', type=float, nargs='+',
                        default=[0, 5, 10, 15, 20, 25],
                        help='Bilirubin levels to simulate in mg/dL')
    parser.add_argument('--output-dir', type=str, default='phototherapy_results',
                        help='Output directory for results')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Check GPU availability
    print("Checking GPU availability...")
    try:
        gpus = pmcx.gpuinfo()
        if gpus:
            print(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
        else:
            print("Warning: No GPU detected!")
    except Exception as e:
        print(f"GPU check failed: {e}")
    
    # Run comparison study
    print(f"\nRunning comparison study for bilirubin levels: {args.bilirubin}")
    results = run_comparison_study(
        bilirubin_levels=args.bilirubin,
        nphoton=args.nphoton,
        test_mode=args.test
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SIMULATION RESULTS SUMMARY")
    print("="*70)
    print(f"{'Bilirubin (mg/dL)':<20} {'Penetration (mm)':<20} {'Dermis Absorption %':<20}")
    print("-"*70)
    
    for r in results:
        dermis_pct = r['layer_absorption']['dermis']['fraction'] * 100
        print(f"{r['bilirubin_mg_dl']:<20.1f} {r['penetration_depth_mm']:<20.3f} {dermis_pct:<20.1f}")
    
    # Save results
    save_results(results, args.output_dir)
    
    # Generate plots if matplotlib available and not disabled
    if not args.no_plot:
        try:
            from analyze_phototherapy import generate_comparison_plots
            generate_comparison_plots(results, args.output_dir)
        except ImportError:
            print("\nNote: Run analyze_phototherapy.py separately for visualization")
    
    print("\n[OK] Simulation complete!")
    return results


if __name__ == '__main__':
    main()
