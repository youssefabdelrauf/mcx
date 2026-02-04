#!/usr/bin/env python3
"""
PPG Complete Analysis Runner

Runs the full PPG simulation and analysis pipeline:
1. Monte Carlo simulations for all wavelengths × skin types
2. Extract intensity vs depth profiles
3. Generate figures and tables
4. Create summary report

Usage:
    python run_ppg_analysis.py --test    # Quick test run
    python run_ppg_analysis.py           # Full analysis

Author: Generated for PPG optimization study
"""

import os
import sys
import argparse
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppg_simulation import (
    run_full_analysis, 
    save_results, 
    get_wavelength_range,
    FITZPATRICK_TYPES
)
from ppg_analysis import run_analysis


def main():
    parser = argparse.ArgumentParser(
        description='PPG Complete Analysis: Melanin Bias Study'
    )
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode (reduced photons)')
    parser.add_argument('--nphoton', type=float, default=1e6,
                        help='Number of photons per simulation (default: 1e6)')
    parser.add_argument('--mode', choices=['full', 'standard', 'extended'],
                        default='full',
                        help='Wavelength mode: full (500-1000nm), standard (660/880/940), extended')
    parser.add_argument('--output-dir', type=str, default='ppg_results',
                        help='Output directory for all results')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, only run analysis (requires existing results)')
    parser.add_argument('--results-file', type=str,
                        help='Path to existing results file (for --skip-sim)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PPG COMPLETE ANALYSIS: MELANIN BIAS STUDY")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.skip_sim:
        # Load existing results
        if not args.results_file:
            # Find most recent results file
            data_dir = os.path.join(args.output_dir, 'data')
            if not os.path.exists(data_dir):
                print("Error: No existing results found. Run simulation first.")
                sys.exit(1)
            
            files = [f for f in os.listdir(data_dir) if f.startswith('simulation_results')]
            if not files:
                print("Error: No simulation results files found.")
                sys.exit(1)
            
            args.results_file = os.path.join(data_dir, sorted(files)[-1])
        
        print(f"\n[SKIP SIMULATION] Loading results from: {args.results_file}")
        from ppg_analysis import load_results
        results = load_results(args.results_file)
        
    else:
        # Run simulations
        wavelengths = get_wavelength_range(args.mode)
        
        print(f"\n[PHASE 1: SIMULATION]")
        print(f"Wavelengths: {wavelengths}")
        print(f"Skin Types: {list(FITZPATRICK_TYPES.keys())}")
        print(f"Photons: {args.nphoton:.0e}")
        print(f"Test Mode: {args.test}")
        
        # Calculate total simulations
        n_sims = len(wavelengths) * len(FITZPATRICK_TYPES)
        print(f"Total Simulations: {n_sims}")
        
        # Run simulation
        results = run_full_analysis(
            wavelengths=wavelengths,
            nphoton=args.nphoton,
            test_mode=args.test
        )
        
        # Save results
        results_file, summary_file = save_results(results, args.output_dir)
        print(f"\nResults saved to: {results_file}")
    
    # Run analysis
    print(f"\n[PHASE 2: ANALYSIS]")
    analysis_results = run_analysis(results=results, output_dir=args.output_dir)
    
    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"Optimal Wavelength (minimum skin-tone bias): {analysis_results['optimal_wavelength']} nm")
    print("\nAll figures and tables saved to:", args.output_dir)
    print("=" * 70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("[OK] COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
