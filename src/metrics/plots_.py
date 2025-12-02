# plots_.py
# Standalone script for side-by-side comparison plots for ROI and Non-ROI video encoding metrics.
# Visualizes frame-level PSNR, SSIM, BRISQUE metrics from both pipeline strategies for quality analysis.

import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
metrics_roi_data = pd.read_csv(r"ROI\144X144\HQF_40_LQF_20\GOP_30\quality_metrics_roi", delimiter="\t")
metrics_data = pd.read_csv(r"ROI\144X144\QF_80\GOP_30\quality_metrics", delimiter="\t")

# Extract the relevant columns for plotting
frame_numbers = metrics_data['#frameNb']
psnr_global = metrics_data['Psnr']
psnr_roi = metrics_roi_data['Psnr']
brisque_global = metrics_data['Brisque_current']
brisque_roi = metrics_roi_data['Brisque_current']
ssim_global = metrics_data['SSIM']
ssim_roi = metrics_roi_data['SSIM']

# --- Plot 1: Comparison of PSNR (Global vs ROI) ---
plt.figure(figsize=(6, 6))
plt.plot(frame_numbers, psnr_global, label="PSNR (No ROI)", color="blue", marker="o")
plt.plot(frame_numbers, psnr_roi, label="PSNR (ROI)", color="orange", marker="x")
plt.xlabel("Frame seq.")
plt.ylabel("PSNR (db)")
plt.legend()
plt.grid()
plt.show()

# --- Plot 2: Comparison of SSIM (Global vs ROI) ---
plt.figure(figsize=(6, 6))
plt.plot(frame_numbers, ssim_global, label="SSIM (No ROI)", color="brown", marker="o")
plt.plot(frame_numbers, ssim_roi, label="SSIM (ROI)", color="magenta", marker="x")
plt.xlabel("Frame seq.")
plt.ylabel("SSIM")
plt.legend()
plt.grid()
plt.show()

# --- Plot 3: BRISQUE Metrics (Global vs ROI) ---
plt.figure(figsize=(6, 6))
plt.plot(frame_numbers, brisque_global, label="BRISQUE (No ROI)", color="green", marker="o")
plt.plot(frame_numbers, brisque_roi, label="BRISQUE (ROI)", color="red", marker="x")
plt.xlabel("Frame Seq.")
plt.ylabel("BRISQUE Score")
plt.legend()
plt.grid()
plt.show()



plt.tight_layout()
plt.show()
