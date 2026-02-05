"""
Plot PI Variance vs Wavelength from Monte Carlo Simulation Results
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from optimization_results_20260203_223012.json
wavelengths = [500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 
               760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000]

variance = [8.538415455536322e-09, 3.9651426177318253e-08, 9.393657257032761e-08,
            1.5309076987509717e-07, 1.3893063976422735e-07, 3.0505787644969436e-08,
            1.1988673288915645e-08, 3.5484541942910023e-09, 1.7469750068260278e-08,
            8.416123642457337e-09, 9.565459372999702e-09, 1.1910617090043975e-08,
            1.4919778674626944e-08, 3.865066527229119e-09, 1.0835519045653098e-08,
            1.5058507760496367e-08, 9.912795665168382e-09, 3.4426262035570828e-09,
            1.844138406018445e-08, 2.6464604058775754e-09, 2.9317202724661136e-08,
            1.276501280804633e-08, 2.238720084426909e-08, 1.0141187545070282e-08,
            1.1666473040705297e-08, 1.2405689290482264e-08]

# Independence scores (for secondary y-axis)
independence = [94.42, 74.10, 38.64, 0.00, 9.25, 80.07, 92.17, 97.68, 88.59, 
                94.50, 93.75, 92.22, 90.25, 97.48, 92.92, 90.16, 93.52, 97.75, 
                87.95, 98.27, 80.85, 91.66, 85.38, 93.38, 92.38, 91.90]

# Find optimal wavelength (minimum variance)
min_idx = np.argmin(variance)
optimal_wl = wavelengths[min_idx]
min_var = variance[min_idx]

# Create figure with dark theme
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(14, 8))

# Primary axis - PI Variance
color1 = '#00D4FF'  # Cyan
ax1.semilogy(wavelengths, variance, 'o-', color=color1, linewidth=2.5, 
             markersize=8, markeredgecolor='white', markeredgewidth=1, label='PI Variance')
ax1.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold', color='white')
ax1.set_ylabel('PI Variance (σ²)', fontsize=14, fontweight='bold', color=color1)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
ax1.tick_params(axis='x', labelsize=11)

# Highlight optimal wavelength
ax1.axvline(x=optimal_wl, color='#FFD700', linestyle='--', linewidth=2, alpha=0.8)
ax1.scatter([optimal_wl], [min_var], s=300, color='#FFD700', zorder=5, 
            edgecolors='white', linewidths=2, marker='*')
ax1.annotate(f'Optimal: {optimal_wl} nm\nσ² = {min_var:.2e}', 
             xy=(optimal_wl, min_var), xytext=(optimal_wl + 30, min_var * 5),
             fontsize=12, fontweight='bold', color='#FFD700',
             arrowprops=dict(arrowstyle='->', color='#FFD700', lw=2),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#333333', edgecolor='#FFD700'))

# Secondary axis - Independence Score
ax2 = ax1.twinx()
color2 = '#FF6B6B'  # Coral
ax2.plot(wavelengths, independence, 's--', color=color2, linewidth=2, 
         markersize=6, alpha=0.8, label='Independence Score')
ax2.set_ylabel('Independence Score (%)', fontsize=14, fontweight='bold', color=color2)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
ax2.set_ylim(0, 105)

# Title
plt.title('Perfusion Index Variance vs Wavelength\n(Monte Carlo Simulation: 10⁶ photons, Fitzpatrick I-VI)', 
          fontsize=16, fontweight='bold', color='white', pad=20)

# Grid
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add spectral regions
ax1.axvspan(500, 600, alpha=0.1, color='green', label='Visible (Green-Yellow)')
ax1.axvspan(600, 700, alpha=0.1, color='red', label='Visible (Red)')
ax1.axvspan(700, 1000, alpha=0.1, color='purple', label='Near-Infrared')

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1[:1] + lines2, labels1[:1] + labels2, loc='upper right', 
           fontsize=11, facecolor='#222222', edgecolor='white')

# Add text annotation for simulation parameters
textstr = 'Simulation Parameters:\n• 1×10⁶ photons\n• SDD = 5.0 mm\n• Skin Types I-VI'
props = dict(boxstyle='round,pad=0.5', facecolor='#333333', edgecolor='#888888', alpha=0.9)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props, color='white')

plt.tight_layout()

# Save figure
output_path = r'c:\Users\Youssef\Desktop\MCX\mcx\ppg_results\optimization\figures\pi_variance_wavelength.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
print(f"Graph saved to: {output_path}")

plt.show()
