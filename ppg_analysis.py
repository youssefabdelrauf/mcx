#!/usr/bin/env python3
"""
PPG Analysis and Visualization

Generates figures and tables from PPG simulation results:
- Figure 1: Intensity vs Depth (per wavelength, per skin type)
- Figure 2: Signal Strength vs Skin Tone
- Figure 3: Bias Comparison (heatmap)
- Tables: Penetration depths, signal attenuation, recommendations

Author: Generated for PPG optimization study
"""

import numpy as np
import json
import os
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available for CSV export")


# Color schemes
SKIN_TYPE_COLORS = {
    'I': '#FFE4C4',    # Bisque (very fair)
    'II': '#FFDAB9',   # Peach puff
    'III': '#DEB887',  # Burlywood
    'IV': '#D2691E',   # Chocolate
    'V': '#8B4513',    # Saddle brown
    'VI': '#4A2C2A',   # Dark brown
}

WAVELENGTH_COLORS = plt.cm.viridis if HAS_MATPLOTLIB else None


def load_results(results_file):
    """Load simulation results from JSON file."""
    with open(results_file, 'r') as f:
        # Handle string keys (JSON converts int keys to strings)
        data = json.load(f)
    
    # Convert wavelength keys back to integers
    results = {}
    for wl_str, skin_data in data.items():
        wl = int(wl_str)
        results[wl] = skin_data
    
    return results


def generate_figure1_intensity_vs_depth(results, output_dir='ppg_results'):
    """
    Generate Figure 1: Intensity vs Depth.
    
    Creates a figure with subplots for each skin type,
    showing intensity vs depth curves for all wavelengths.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    wavelengths = sorted(results.keys())
    skin_types = sorted(results[wavelengths[0]].keys())
    
    # Create figure with 2x3 subplots for 6 skin types
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Figure 1: Intensity vs Depth by Skin Type\n(Each curve = different wavelength)', 
                 fontsize=14, fontweight='bold')
    
    # Color map for wavelengths
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(wavelengths)))
    
    for idx, st in enumerate(skin_types):
        ax = axes.flat[idx]
        
        for wl_idx, wl in enumerate(wavelengths):
            data = results[wl][st]
            depth = np.array(data['depth_mm'])
            intensity = np.array(data['intensity'])
            
            ax.plot(depth, intensity, color=colors[wl_idx], 
                   label=f'{wl} nm', linewidth=1.5, alpha=0.8)
        
        ax.set_title(f"Skin Type {st}: {data['skin_name']}", fontweight='bold')
        ax.set_xlabel('Depth (mm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_xlim([0, 2.0])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1/np.e, color='red', linestyle='--', alpha=0.5, label='1/e threshold')
    
    # Add colorbar for wavelengths
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                                norm=plt.Normalize(vmin=min(wavelengths), vmax=max(wavelengths)))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label('Wavelength (nm)', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    
    # Save figure
    fig_path = os.path.join(output_dir, 'figures', 'fig1_intensity_vs_depth.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure 1 saved to: {fig_path}")
    plt.close()
    
    # Also create version showing wavelength comparison
    generate_figure1_wavelength_comparison(results, output_dir)


def generate_figure1_wavelength_comparison(results, output_dir='ppg_results'):
    """
    Alternative Figure 1: Compare skin types for selected wavelengths.
    """
    if not HAS_MATPLOTLIB:
        return
    
    wavelengths = sorted(results.keys())
    skin_types = sorted(results[wavelengths[0]].keys())
    
    # Select representative wavelengths
    key_wavelengths = [wl for wl in [550, 660, 800, 940] if wl in wavelengths]
    if not key_wavelengths:
        key_wavelengths = wavelengths[::len(wavelengths)//4]  # Sample 4 wavelengths
    
    fig, axes = plt.subplots(1, len(key_wavelengths), figsize=(4*len(key_wavelengths), 4))
    if len(key_wavelengths) == 1:
        axes = [axes]
    
    fig.suptitle('Figure 1b: Intensity vs Depth - Skin Type Comparison', 
                 fontsize=12, fontweight='bold')
    
    for ax_idx, wl in enumerate(key_wavelengths):
        ax = axes[ax_idx]
        
        for st in skin_types:
            data = results[wl][st]
            depth = np.array(data['depth_mm'])
            intensity = np.array(data['intensity'])
            
            ax.plot(depth, intensity, color=SKIN_TYPE_COLORS.get(st, 'gray'),
                   label=f'Type {st}', linewidth=2)
        
        ax.set_title(f'λ = {wl} nm', fontweight='bold')
        ax.set_xlabel('Depth (mm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_xlim([0, 2.0])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig_path = os.path.join(output_dir, 'figures', 'fig1b_skin_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure 1b saved to: {fig_path}")
    plt.close()


def generate_figure2_signal_vs_skintone(results, output_dir='ppg_results'):
    """
    Generate Figure 2: Signal Strength vs Skin Tone.
    
    Shows how signal strength degrades with increasing melanin across wavelengths.
    Signal strength is computed as integrated intensity in detection zone.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    wavelengths = sorted(results.keys())
    skin_types = sorted(results[wavelengths[0]].keys())
    
    # Calculate signal strength for each combination
    # Using penetration depth as proxy for signal quality
    signal_data = {wl: [] for wl in wavelengths}
    
    for wl in wavelengths:
        baseline = results[wl]['I']['penetration_depth_mm']  # Type I as reference
        for st in skin_types:
            pen_depth = results[wl][st]['penetration_depth_mm']
            # Relative signal strength (normalized to Type I)
            rel_signal = pen_depth / baseline if baseline > 0 else 1.0
            signal_data[wl].append(rel_signal)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(skin_types))
    width = 0.8 / len(wavelengths)
    
    # Color map for wavelengths
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(wavelengths)))
    
    for i, wl in enumerate(wavelengths):
        offset = (i - len(wavelengths)/2 + 0.5) * width
        bars = ax.bar(x + offset, signal_data[wl], width, 
                     label=f'{wl} nm', color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=12)
    ax.set_ylabel('Relative Signal Strength\n(normalized to Type I)', fontsize=12)
    ax.set_title('Figure 2: Signal Strength vs Skin Tone\n(Higher = better signal penetration)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Type {st}\n{results[wavelengths[0]][st]["skin_name"]}' 
                        for st in skin_types], fontsize=9)
    ax.legend(title='Wavelength', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Type I baseline')
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'figures', 'fig2_signal_vs_skintone.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure 2 saved to: {fig_path}")
    plt.close()
    
    # Also create line plot version
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_vals = [results[wavelengths[0]][st]['melanin_fraction'] * 100 for st in skin_types]
    
    for i, wl in enumerate(wavelengths):
        ax.plot(x_vals, signal_data[wl], 'o-', color=colors[i], 
               label=f'{wl} nm', linewidth=2, markersize=8)
    
    ax.set_xlabel('Melanin Concentration (%)', fontsize=12)
    ax.set_ylabel('Relative Signal Strength', fontsize=12)
    ax.set_title('Figure 2b: Signal Strength vs Melanin Concentration', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Wavelength', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'figures', 'fig2b_signal_vs_melanin.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure 2b saved to: {fig_path}")
    plt.close()


def generate_figure3_bias_comparison(results, output_dir='ppg_results'):
    """
    Generate Figure 3: Bias Comparison Heatmap.
    
    Shows signal attenuation (% reduction vs Type I) as a heatmap
    with wavelengths on Y-axis and skin types on X-axis.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    wavelengths = sorted(results.keys())
    skin_types = sorted(results[wavelengths[0]].keys())
    
    # Calculate bias matrix (% signal reduction vs Type I)
    bias_matrix = np.zeros((len(wavelengths), len(skin_types)))
    
    for i, wl in enumerate(wavelengths):
        baseline = results[wl]['I']['penetration_depth_mm']
        for j, st in enumerate(skin_types):
            pen_depth = results[wl][st]['penetration_depth_mm']
            # Bias = % reduction from Type I
            bias = (1 - pen_depth / baseline) * 100 if baseline > 0 else 0
            bias_matrix[i, j] = bias
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(bias_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
    
    # Labels
    ax.set_xticks(np.arange(len(skin_types)))
    ax.set_xticklabels([f'Type {st}' for st in skin_types])
    ax.set_yticks(np.arange(len(wavelengths)))
    ax.set_yticklabels([f'{wl} nm' for wl in wavelengths])
    
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=12)
    ax.set_ylabel('Wavelength', fontsize=12)
    ax.set_title('Figure 3: Bias Comparison\n(% Signal Reduction vs Type I - Lower is better)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Signal Reduction (%)', fontsize=10)
    
    # Add text annotations
    for i in range(len(wavelengths)):
        for j in range(len(skin_types)):
            val = bias_matrix[i, j]
            color = 'white' if val > 25 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                   color=color, fontsize=8)
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'figures', 'fig3_bias_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure 3 saved to: {fig_path}")
    plt.close()
    
    # Find optimal wavelength (minimum variance across skin types)
    variance_per_wavelength = np.var(bias_matrix, axis=1)
    optimal_idx = np.argmin(variance_per_wavelength)
    optimal_wavelength = wavelengths[optimal_idx]
    
    print(f"\n*** OPTIMAL WAVELENGTH (minimum skin-tone variance): {optimal_wavelength} nm ***")
    
    return bias_matrix, optimal_wavelength


def generate_tables(results, output_dir='ppg_results'):
    """
    Generate comparison tables.
    """
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    
    wavelengths = sorted(results.keys())
    skin_types = sorted(results[wavelengths[0]].keys())
    
    # Table 1: Penetration Depths
    pen_depths = {}
    for wl in wavelengths:
        pen_depths[f'{wl} nm'] = {}
        for st in skin_types:
            pen_depths[f'{wl} nm'][f'Type {st}'] = results[wl][st]['penetration_depth_mm']
    
    # Table 2: Signal Attenuation (% reduction vs Type I)
    attenuation = {}
    for wl in wavelengths:
        attenuation[f'{wl} nm'] = {}
        baseline = results[wl]['I']['penetration_depth_mm']
        for st in skin_types:
            pen = results[wl][st]['penetration_depth_mm']
            att = (1 - pen / baseline) * 100 if baseline > 0 else 0
            attenuation[f'{wl} nm'][f'Type {st}'] = f'{att:.1f}%'
    
    # Save as JSON
    tables_data = {
        'penetration_depths_mm': pen_depths,
        'signal_attenuation_percent': attenuation
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    json_path = os.path.join(output_dir, 'tables', f'comparison_tables_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(tables_data, f, indent=2)
    print(f"Tables saved to: {json_path}")
    
    # Save as CSV if pandas available
    if HAS_PANDAS:
        # Penetration depths
        df_pen = pd.DataFrame(pen_depths).T
        csv_path = os.path.join(output_dir, 'tables', f'penetration_depths_{timestamp}.csv')
        df_pen.to_csv(csv_path)
        print(f"Penetration depths CSV: {csv_path}")
        
        # Attenuation
        df_att = pd.DataFrame(attenuation).T
        csv_path = os.path.join(output_dir, 'tables', f'signal_attenuation_{timestamp}.csv')
        df_att.to_csv(csv_path)
        print(f"Signal attenuation CSV: {csv_path}")
    
    return tables_data


def generate_summary_report(results, output_dir='ppg_results'):
    """
    Generate a markdown summary report.
    """
    wavelengths = sorted(results.keys())
    skin_types = sorted(results[wavelengths[0]].keys())
    
    # Calculate key statistics
    max_variance_wl = None
    min_variance_wl = None
    max_var = 0
    min_var = float('inf')
    
    wl_stats = {}
    for wl in wavelengths:
        pen_depths = [results[wl][st]['penetration_depth_mm'] for st in skin_types]
        variance = np.var(pen_depths)
        mean_pen = np.mean(pen_depths)
        
        wl_stats[wl] = {
            'mean_penetration': mean_pen,
            'variance': variance,
            'range': max(pen_depths) - min(pen_depths)
        }
        
        if variance > max_var:
            max_var = variance
            max_variance_wl = wl
        if variance < min_var:
            min_var = variance
            min_variance_wl = wl
    
    report = f"""# PPG Simulation Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Wavelength Range:** {min(wavelengths)} - {max(wavelengths)} nm
- **Skin Types Analyzed:** {len(skin_types)} (Fitzpatrick I-VI)
- **Total Simulations:** {len(wavelengths) * len(skin_types)}

## Key Findings

### Optimal Wavelength for Skin-Tone Independence

The wavelength with **minimum variance** in penetration depth across skin types:

**lambda_optimal = {min_variance_wl} nm**

This wavelength shows the most consistent signal quality regardless of melanin content.

### Wavelength with Maximum Bias

The wavelength with **maximum variance** (most affected by melanin):

**lambda_worst = {max_variance_wl} nm**

Avoid this wavelength for skin-tone independent PPG sensors.

## Wavelength Performance Summary

| Wavelength | Mean Penetration (mm) | Variance | Range (mm) |
|------------|----------------------|----------|------------|
"""
    
    for wl in wavelengths:
        stats = wl_stats[wl]
        report += f"| {wl} nm | {stats['mean_penetration']:.3f} | {stats['variance']:.6f} | {stats['range']:.3f} |\n"
    
    report += """
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
"""
    
    report_path = os.path.join(output_dir, 'summary_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSummary report saved to: {report_path}")
    
    return report


def run_analysis(results_file=None, results=None, output_dir='ppg_results'):
    """
    Run full analysis pipeline.
    
    Parameters:
    -----------
    results_file : str
        Path to simulation results JSON file
    results : dict
        Alternatively, pass results dict directly
    output_dir : str
        Output directory for figures and tables
    """
    print("=" * 70)
    print("PPG ANALYSIS AND VISUALIZATION")
    print("=" * 70)
    
    # Load results
    if results is None:
        if results_file is None:
            raise ValueError("Must provide either results_file or results")
        results = load_results(results_file)
    
    wavelengths = sorted(results.keys())
    skin_types = sorted(results[wavelengths[0]].keys())
    
    print(f"Loaded results for {len(wavelengths)} wavelengths × {len(skin_types)} skin types")
    
    # Generate figures
    print("\n[Generating Figures]")
    generate_figure1_intensity_vs_depth(results, output_dir)
    generate_figure2_signal_vs_skintone(results, output_dir)
    bias_matrix, optimal_wl = generate_figure3_bias_comparison(results, output_dir)
    
    # Generate tables
    print("\n[Generating Tables]")
    tables = generate_tables(results, output_dir)
    
    # Generate summary report
    print("\n[Generating Summary Report]")
    report = generate_summary_report(results, output_dir)
    
    print("\n" + "=" * 70)
    print("[OK] ANALYSIS COMPLETE!")
    print("=" * 70)
    
    return {
        'bias_matrix': bias_matrix,
        'optimal_wavelength': optimal_wl,
        'tables': tables
    }


# =============================================================================
# OPTIMIZATION VISUALIZATION FUNCTIONS
# =============================================================================

def generate_pi_variance_heatmap(optimization_results, output_dir='ppg_results/optimization'):
    """
    Generate heatmap of Perfusion Index across wavelengths and skin types.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return None
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    pi_data = optimization_results.get('pi_by_wavelength_skintype', {})
    variance_data = optimization_results.get('variance_by_wavelength', {})
    optimal_wl = optimization_results.get('optimal_wavelength', 0)
    
    if not pi_data:
        print("No PI data available")
        return None
    
    wavelengths = sorted([int(wl) for wl in pi_data.keys()])
    skin_types = sorted(pi_data[wavelengths[0]].keys())
    
    # Create PI matrix
    pi_matrix = np.zeros((len(wavelengths), len(skin_types)))
    for i, wl in enumerate(wavelengths):
        for j, st in enumerate(skin_types):
            pi_matrix[i, j] = pi_data[wl].get(st, 0)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: PI Heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(pi_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(np.arange(len(skin_types)))
    ax1.set_xticklabels([f'Type {st}' for st in skin_types])
    ax1.set_yticks(np.arange(len(wavelengths)))
    ax1.set_yticklabels([f'{wl} nm' for wl in wavelengths])
    ax1.set_xlabel('Fitzpatrick Skin Type', fontsize=11)
    ax1.set_ylabel('Wavelength', fontsize=11)
    ax1.set_title('Perfusion Index (PI) by Wavelength & Skin Type', fontsize=12, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Perfusion Index (%)', fontsize=10)
    
    # Highlight optimal wavelength
    opt_idx = wavelengths.index(optimal_wl) if optimal_wl in wavelengths else -1
    if opt_idx >= 0:
        ax1.axhline(y=opt_idx, color='green', linewidth=3, alpha=0.7)
    
    # Subplot 2: PI Variance by Wavelength
    ax2 = axes[1]
    variances = [variance_data.get(wl, 0) for wl in wavelengths]
    colors = ['green' if wl == optimal_wl else 'steelblue' for wl in wavelengths]
    ax2.barh([f'{wl} nm' for wl in wavelengths], variances, color=colors, alpha=0.8)
    ax2.set_xlabel('PI Variance (lower = more consistent)', fontsize=11)
    ax2.set_ylabel('Wavelength', fontsize=11)
    ax2.set_title('Variance of PI Across Skin Types', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'figures', 'pi_variance_heatmap.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"PI Variance Heatmap saved to: {fig_path}")
    plt.close()
    
    return fig_path


def generate_pi_variance_vs_wavelength_figure(optimization_results, output_dir='ppg_results/optimization'):
    """
    Generate publication-ready PI Variance vs Wavelength figure.
    
    Creates a line plot showing:
    - PI variance (Y-axis) vs Wavelength (X-axis)
    - Clear minimum at optimal wavelength indicating melanin resistance
    - Suitable for manuscript Figure showing skin-tone independence
    
    Parameters:
    -----------
    optimization_results : dict
        Results from optimal wavelength optimization containing variance_by_wavelength
    output_dir : str
        Output directory for figures
        
    Returns:
    --------
    str : Path to saved figure
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return None
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    variance_data = optimization_results.get('variance_by_wavelength', {})
    optimal_wl = optimization_results.get('optimal_wavelength', 0)
    min_variance = optimization_results.get('min_variance', 0)
    pi_data = optimization_results.get('pi_by_wavelength_skintype', {})
    
    if not variance_data:
        print("No variance data available")
        return None
    
    # Sort wavelengths and extract variance values
    wavelengths = sorted([int(wl) for wl in variance_data.keys()])
    variances = [variance_data.get(str(wl), variance_data.get(wl, 0)) for wl in wavelengths]
    
    # Convert variance to a more readable scale (multiply by 1e6 for display)
    variances_scaled = [v * 1e6 for v in variances]
    min_variance_scaled = min_variance * 1e6
    
    # Create publication-quality figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Set style for publication
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })
    
    # Main line plot with markers
    ax.plot(wavelengths, variances_scaled, 'o-', 
            color='#2C3E50', linewidth=2.5, markersize=10,
            markerfacecolor='#3498DB', markeredgecolor='#2C3E50', 
            markeredgewidth=1.5, label='PI Variance', zorder=3)
    
    # Highlight optimal wavelength with special marker
    opt_idx = wavelengths.index(optimal_wl) if optimal_wl in wavelengths else -1
    if opt_idx >= 0:
        ax.scatter([optimal_wl], [variances_scaled[opt_idx]], 
                   s=250, marker='*', color='#E74C3C', edgecolor='white',
                   linewidth=2, zorder=5, label=f'Optimal: {optimal_wl} nm')
        
        # Add annotation for minimum
        ax.annotate(f'Minimum at {optimal_wl} nm\n(Variance = {min_variance_scaled:.3f}×10⁻⁶)',
                    xy=(optimal_wl, variances_scaled[opt_idx]),
                    xytext=(optimal_wl + 50, variances_scaled[opt_idx] + max(variances_scaled)*0.15),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FADBD8', 
                              edgecolor='#E74C3C', alpha=0.9))
    
    # Add shaded region for NIR/melanin-resistant zone
    ax.axvspan(800, max(wavelengths) + 50, alpha=0.15, color='#27AE60', 
               label='Near-IR Region (Melanin Resistant)')
    
    # Add reference line at mean variance
    mean_var = np.mean(variances_scaled)
    ax.axhline(y=mean_var, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(min(wavelengths) + 10, mean_var * 1.05, 'Mean Variance', 
            fontsize=10, color='gray', style='italic')
    
    # Labels and title
    ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('PI Variance (×10⁻⁶)', fontsize=14, fontweight='bold')
    ax.set_title('Perfusion Index Variance Across Fitzpatrick Skin Types I–VI\n'
                 '(Lower variance = higher melanin resistance)',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim([min(wavelengths) - 30, max(wavelengths) + 30])
    ax.set_ylim([0, max(variances_scaled) * 1.2])
    
    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=6, width=1)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure in multiple formats
    fig_path_png = os.path.join(output_dir, 'figures', 'pi_variance_vs_wavelength.png')
    fig_path_pdf = os.path.join(output_dir, 'figures', 'pi_variance_vs_wavelength.pdf')
    fig_path_svg = os.path.join(output_dir, 'figures', 'pi_variance_vs_wavelength.svg')
    
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(fig_path_pdf, bbox_inches='tight', facecolor='white')
    plt.savefig(fig_path_svg, bbox_inches='tight', facecolor='white')
    
    print(f"PI Variance vs Wavelength figure saved to:")
    print(f"  - PNG: {fig_path_png}")
    print(f"  - PDF: {fig_path_pdf}")
    print(f"  - SVG: {fig_path_svg}")
    plt.close()
    
    return fig_path_png


def generate_ror_calibration_curves(spo2_results, output_dir='ppg_results/optimization'):
    """Generate RoR calibration curves for different wavelength pairs."""
    if not HAS_MATPLOTLIB:
        return None
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    optimal_pair = spo2_results.get('optimal_pair', (0, 0))
    standard_pair = spo2_results.get('standard_pair', (660, 940))
    ror_data = spo2_results.get('ror_data', {})
    
    if not ror_data:
        return None
    
    std_key = f"{standard_pair[0]}/{standard_pair[1]}"
    opt_key = f"{optimal_pair[0]}/{optimal_pair[1]}"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    skin_types = ['I', 'II', 'III', 'IV', 'V', 'VI']
    
    for ax, (pair_key, title) in zip(axes, [(std_key, f'Standard ({std_key} nm)'), 
                                              (opt_key, f'Optimal ({opt_key} nm)')]):
        if pair_key not in ror_data:
            continue
        
        for st in skin_types:
            if st not in ror_data[pair_key]:
                continue
            st_data = ror_data[pair_key][st]
            spo2_vals = sorted([float(s) for s in st_data.keys()])
            ror_vals = [st_data[str(s)] for s in spo2_vals]
            
            ax.plot([s*100 for s in spo2_vals], ror_vals, 'o-', 
                   color=SKIN_TYPE_COLORS.get(st, 'gray'), label=f'Type {st}', linewidth=2)
        
        ax.set_xlabel('True SpO2 (%)', fontsize=11)
        ax.set_ylabel('Ratio of Ratios (R)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('RoR Calibration: Standard vs Optimal Pairs', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    fig_path = os.path.join(output_dir, 'figures', 'ror_calibration_curves.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"RoR Calibration Curves saved to: {fig_path}")
    plt.close()
    
    return fig_path


def generate_sdd_optimization_plot(sdd_results, output_dir='ppg_results/optimization'):
    """Generate SDD optimization plot."""
    if not HAS_MATPLOTLIB:
        return None
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    wavelength = sdd_results.get('wavelength', 0)
    optimal_sdd = sdd_results.get('optimal_sdd', 0)
    sdd_metrics = sdd_results.get('sdd_metrics', {})
    
    if not sdd_metrics:
        return None
    
    sdd_values = sorted([float(s) for s in sdd_metrics.keys()])
    mean_signals = [sdd_metrics[s]['mean_signal'] for s in sdd_values]
    pi_variances = [sdd_metrics[s]['pi_variance'] for s in sdd_values]
    combined_scores = [sdd_metrics[s].get('combined_score', 0) for s in sdd_values]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Signal vs SDD
    axes[0].plot(sdd_values, mean_signals, 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[0].axvline(x=optimal_sdd, color='green', linestyle='--', linewidth=2)
    axes[0].set_xlabel('SDD (mm)')
    axes[0].set_ylabel('Signal Strength')
    axes[0].set_title('Signal vs SDD')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Variance vs SDD
    axes[1].plot(sdd_values, pi_variances, 'o-', color='coral', linewidth=2, markersize=8)
    axes[1].axvline(x=optimal_sdd, color='green', linestyle='--', linewidth=2)
    axes[1].set_xlabel('SDD (mm)')
    axes[1].set_ylabel('PI Variance')
    axes[1].set_title('Skin-Tone Independence vs SDD')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Combined Score
    colors = ['green' if s == optimal_sdd else 'gray' for s in sdd_values]
    axes[2].bar([str(int(s)) for s in sdd_values], combined_scores, color=colors, alpha=0.8)
    axes[2].set_xlabel('SDD (mm)')
    axes[2].set_ylabel('Combined Score')
    axes[2].set_title(f'Optimal SDD = {optimal_sdd} mm')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'SDD Optimization for λ = {wavelength} nm', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    fig_path = os.path.join(output_dir, 'figures', 'sdd_optimization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"SDD Optimization plot saved to: {fig_path}")
    plt.close()
    
    return fig_path


def generate_cooptimization_surface(co_opt_results, output_dir='ppg_results/optimization'):
    """Generate 2D heatmap of co-optimization results."""
    if not HAS_MATPLOTLIB:
        return None
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    optimal_wl = co_opt_results.get('optimal_wavelength', 0)
    optimal_sdd = co_opt_results.get('optimal_sdd', 0)
    surface = co_opt_results.get('optimization_surface', {})
    
    if not surface:
        return None
    
    wl_set, sdd_set = set(), set()
    for key in surface.keys():
        wl, sdd = key.split('/')
        wl_set.add(int(wl))
        sdd_set.add(int(sdd))
    
    wavelengths = sorted(wl_set)
    sdds = sorted(sdd_set)
    
    score_matrix = np.zeros((len(wavelengths), len(sdds)))
    for i, wl in enumerate(wavelengths):
        for j, sdd in enumerate(sdds):
            key = f"{wl}/{sdd}"
            if key in surface:
                score_matrix[i, j] = surface[key].get('combined_score', 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(score_matrix, cmap='viridis', aspect='auto', origin='lower')
    
    ax.set_xticks(np.arange(len(sdds)))
    ax.set_xticklabels([f'{s}' for s in sdds])
    ax.set_yticks(np.arange(len(wavelengths)))
    ax.set_yticklabels([f'{wl}' for wl in wavelengths])
    ax.set_xlabel('SDD (mm)', fontsize=12)
    ax.set_ylabel('Wavelength (nm)', fontsize=12)
    ax.set_title(f'Co-Optimization: Optimal = {optimal_wl} nm, {optimal_sdd} mm', 
                fontsize=13, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Score')
    
    # Mark optimal
    opt_wl_idx = wavelengths.index(optimal_wl) if optimal_wl in wavelengths else -1
    opt_sdd_idx = sdds.index(optimal_sdd) if optimal_sdd in sdds else -1
    if opt_wl_idx >= 0 and opt_sdd_idx >= 0:
        ax.scatter([opt_sdd_idx], [opt_wl_idx], marker='*', s=300, color='red', edgecolor='white')
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'figures', 'cooptimization_surface.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Co-optimization Surface saved to: {fig_path}")
    plt.close()
    
    return fig_path


def generate_optimization_summary_figures(all_results, output_dir='ppg_results/optimization'):
    """Generate all optimization figures from combined results."""
    print("\n[Generating Optimization Figures]")
    
    figures = {}
    
    if 'optimal_wavelength' in all_results:
        fig = generate_pi_variance_heatmap(all_results['optimal_wavelength'], output_dir)
        if fig:
            figures['pi_variance_heatmap'] = fig
        
        # Also generate the publication-ready PI Variance vs Wavelength line plot
        fig2 = generate_pi_variance_vs_wavelength_figure(all_results['optimal_wavelength'], output_dir)
        if fig2:
            figures['pi_variance_vs_wavelength'] = fig2
    
    if 'optimal_spo2_pair' in all_results:
        fig = generate_ror_calibration_curves(all_results['optimal_spo2_pair'], output_dir)
        if fig:
            figures['ror_calibration'] = fig
    
    if 'sdd_optimization' in all_results:
        fig = generate_sdd_optimization_plot(all_results['sdd_optimization'], output_dir)
        if fig:
            figures['sdd_optimization'] = fig
    
    if 'co_optimization' in all_results:
        fig = generate_cooptimization_surface(all_results['co_optimization'], output_dir)
        if fig:
            figures['cooptimization'] = fig
    
    print(f"\n[OK] Generated {len(figures)} optimization figures")
    return figures


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PPG Analysis and Visualization'
    )
    parser.add_argument('--results', type=str, required=True,
                        help='Path to simulation results JSON file')
    parser.add_argument('--output-dir', type=str, default='ppg_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_analysis(results_file=args.results, output_dir=args.output_dir)
