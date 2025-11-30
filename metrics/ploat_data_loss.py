# ploat_data_loss.py
# Provides visual comparison of network loss-related metrics (Signal Loss, SNR, BER) across frames/time.
# Reads packet trace data and produces labeled line plots for each loss/signal channel, supporting analysis of network conditions on video transmission quality.

import pandas as pd
import matplotlib.pyplot as plt

# Load the data (adjust the file path and delimiter if necessary)
file_path = r"VIRAT_S_010204_05_000856_000890\144X144\QF_80\GOP_30\trace\st-packet"
data = pd.read_csv(file_path, delimiter="\t")

time = data["#time"]
signal_loss = data["signal_lost(db)"]
snr = data["SNR"]
ber = data["BER"]

# --- Plot 1: Signal Loss vs. Time ---
plt.figure(figsize=(6, 6))
plt.plot(time, signal_loss, color="red", label="Signal Loss (dB)")
plt.xlabel("Time")
plt.ylabel("Signal Loss (dB)")
plt.title("Signal Loss vs. Time")
plt.grid(True)
plt.legend()
plt.show()

# --- Plot 2: SNR vs. Time ---
plt.figure(figsize=(6, 6))
plt.plot(time, snr, color="blue", label="SNR")
plt.xlabel("Time")
plt.ylabel("SNR (dB)")
plt.title("SNR vs. Time")
plt.grid(True)
plt.legend()
plt.show()

# --- Plot 3: BER vs. Time ---
plt.figure(figsize=(6, 6))
plt.plot(time, ber, color="green", label="BER")
plt.xlabel("Time")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs. Time")
plt.grid(True)
plt.legend()
plt.show()
