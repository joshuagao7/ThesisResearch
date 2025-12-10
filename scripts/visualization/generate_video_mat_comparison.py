"""Generate video vs mat time comparison plot with wider aspect ratio.

This script creates a scatter plot comparing video-based flight time measurements
(heel-to-heel and toe-to-toe) versus mat sensor measurements.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set up paths
project_root = Path(__file__).parent.parent.parent
output_path = project_root / "writeup" / "video_mat_comparison.png"

# Set consistent font styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# Regression statistics from results.tex
# Heel-to-heel: slope=0.724, intercept=0.157, R²=0.495
# Toe-to-toe: slope=0.687, intercept=0.019, R²=0.623

# NOTE: You'll need to replace this with your actual data
# This is a template that generates synthetic data matching the statistics
# Replace the data arrays below with your actual measurements

# Example: Generate synthetic data that matches the reported statistics
# Replace these with your actual data arrays
np.random.seed(42)  # For reproducibility of example

# Generate synthetic mat times (x-axis) in range 0.3 to 0.7 seconds
n_points = 31
mat_times = np.linspace(0.3, 0.7, n_points) + np.random.normal(0, 0.02, n_points)

# Generate heel-to-heel video times based on regression: y = 0.724*x + 0.157 + noise
heel_slope = 0.724
heel_intercept = 0.157
heel_r_squared = 0.495
heel_correlation = np.sqrt(heel_r_squared)  # R = 0.704
heel_video_times = heel_slope * mat_times + heel_intercept
# Add noise to match R²
heel_noise_std = np.std(heel_video_times) * np.sqrt((1 - heel_r_squared) / heel_r_squared)
heel_video_times += np.random.normal(0, heel_noise_std, n_points)

# Generate toe-to-toe video times based on regression: y = 0.687*x + 0.019 + noise
toe_slope = 0.687
toe_intercept = 0.019
toe_r_squared = 0.623
toe_correlation = np.sqrt(toe_r_squared)  # R = 0.789
toe_video_times = toe_slope * mat_times + toe_intercept
# Add noise to match R²
toe_noise_std = np.std(toe_video_times) * np.sqrt((1 - toe_r_squared) / toe_r_squared)
toe_video_times += np.random.normal(0, toe_noise_std, n_points)

# TODO: Replace the synthetic data above with your actual data:
# mat_times = your_mat_measurements
# heel_video_times = your_heel_to_heel_measurements  
# toe_video_times = your_toe_to_toe_measurements

# Create figure with wider aspect ratio (16:9 or wider)
fig, ax = plt.subplots(figsize=(12, 6.75))  # Wider aspect ratio

# Plot heel-to-heel measurements (blue circles)
ax.scatter(mat_times, heel_video_times, 
           color='#0066CC', marker='o', s=80, alpha=0.7, 
           label='Heel-to-heel', edgecolors='darkblue', linewidths=1.5)

# Plot toe-to-toe measurements (pink triangles)
ax.scatter(mat_times, toe_video_times, 
           color='#FF69B4', marker='^', s=80, alpha=0.7, 
           label='Toe-to-toe', edgecolors='darkmagenta', linewidths=1.5)

# Calculate and plot regression lines
# Heel-to-heel regression
heel_slope_calc, heel_intercept_calc, heel_r_value, heel_p_value, heel_std_err = stats.linregress(mat_times, heel_video_times)
heel_r_squared_calc = heel_r_value ** 2
heel_line_x = np.linspace(mat_times.min(), mat_times.max(), 100)
heel_line_y = heel_slope_calc * heel_line_x + heel_intercept_calc
ax.plot(heel_line_x, heel_line_y, '--', color='#0066CC', linewidth=2, alpha=0.8,
        label=f'Heel-to-heel fit (R² = {heel_r_squared_calc:.3f})')

# Toe-to-toe regression
toe_slope_calc, toe_intercept_calc, toe_r_value, toe_p_value, toe_std_err = stats.linregress(mat_times, toe_video_times)
toe_r_squared_calc = toe_r_value ** 2
toe_line_x = np.linspace(mat_times.min(), mat_times.max(), 100)
toe_line_y = toe_slope_calc * toe_line_x + toe_intercept_calc
ax.plot(toe_line_x, toe_line_y, '--', color='#FF69B4', linewidth=2, alpha=0.8,
        label=f'Toe-to-toe fit (R² = {toe_r_squared_calc:.3f})')

# Add 1:1 reference line (dashed gray)
max_val = max(heel_video_times.max(), toe_video_times.max(), mat_times.max())
min_val = min(heel_video_times.min(), toe_video_times.min(), mat_times.min())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.3, label='1:1 line')

# Labels and formatting
ax.set_xlabel('Mat Sensor Flight Time (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Video Flight Time (s)', fontsize=14, fontweight='bold')
ax.set_title('Video vs Mat Flight Time Comparison', fontsize=16, fontweight='bold', pad=15)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

# Set equal aspect ratio for better comparison
ax.set_aspect('equal', adjustable='box')

# Tight layout
plt.tight_layout()

# Save figure
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved video-mat comparison plot to: {output_path}")
print(f"  Heel-to-heel: slope={heel_slope_calc:.3f}, intercept={heel_intercept_calc:.3f}, R²={heel_r_squared_calc:.3f}")
print(f"  Toe-to-toe: slope={toe_slope_calc:.3f}, intercept={toe_intercept_calc:.3f}, R²={toe_r_squared_calc:.3f}")

plt.show()
