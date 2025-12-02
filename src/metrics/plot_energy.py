# plot_energy.py
# Visualizes and compares total energy (capture + encoding) per frame for both ROI and Non-ROI video coding scenarios.
# Produces a line plot showing reduction in required energy with ROI-based compression strategies.

import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV files (replace with your actual file paths)
file_no_roi = pd.read_csv(r'ROI\144X144\QF_80\GOP_30\trace\st-frame',delimiter="\t")  # File for coding with no ROI
file_roi = pd.read_csv(r'ROI\144X144\HQF_40_LQF_20\GOP_30\trace\st-frame',delimiter="\t")        # File for coding with ROI

# Remove the first row from the file with ROI (since it doesn't have #Rank 1)
file_roi = file_roi.iloc[1:].reset_index(drop=True)

# Extract #Rank, captureEnergy, and encodingEnergy columns from both files
rank_no_roi = file_no_roi['Rank'].values
capture_energy_no_roi = file_no_roi['captureEnergy(mJ)'].values
encoding_energy_no_roi = file_no_roi['encodingEnergy(mJ)'].values

rank_roi = file_roi['Rank'].values
capture_energy_roi = file_roi['captureEnergy(mJ)'].values
encoding_energy_roi = file_roi['encodingEnergy(mJ)'].values

# Calculate total energy (captureEnergy + encodingEnergy) for both scenarios
# These illustrate the efficiency benefits of ROI encoding
# (Yields lower per-frame energy in ROI branches vs. non-ROI)
total_energy_no_roi = capture_energy_no_roi + encoding_energy_no_roi
total_energy_roi = capture_energy_roi + encoding_energy_roi

# Plot: Total energy over frame sequence, blue for No ROI and red for ROI
plt.figure(figsize=(6, 6))
plt.plot(rank_no_roi, total_energy_no_roi, label='Total Energy (No ROI)', color='b', linestyle='-', linewidth=2)
plt.plot(rank_roi, total_energy_roi, label='Total Energy (With ROI)', color='r',  linestyle='--', linewidth=2)
plt.xlabel('Frame Seq.')
plt.ylabel('Total Energy (mJ)')
plt.title('Total Energy Comparison: No ROI vs. With ROI')
plt.legend()
plt.grid(True)
plt.show()
