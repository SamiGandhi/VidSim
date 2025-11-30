# ploting.py
# Provides reusable plotting functions for visualizing performance metrics, quality statistics, bitrate/energy, and network loss over frame or time sequences.
# Each function produces a pre-labelled matplotlib chart for the indicated measure.

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

# -- PSNR Frame Plot --
def plot_psnr(ax, data):
    """
    Plot PSNR vs Frame Sequence. Args: ax - axis, data - pandas DataFrame with columns 'Rank', 'PSNR'.
    """
    ax.clear()
    ax.plot(data["Rank"], data["PSNR"], 'g-', label='PSNR')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('PSNR', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()

def plot_ref_psnr(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["refPSNR"], 'g-', label='PSNR')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('PSNR', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()

def plot_psnr_bpp(ax, rt_frame, st_frame):
    ax.clear()
    zipped_lists = list(zip(st_frame['bpp'], rt_frame['PSNR']))
    x1,y1 = zip(*sorted(zipped_lists))
    ax.scatter(x1, y1, color='red', s = 5, label='PSNR Rate Distortion')
    ax.set_xlabel('Bits per Pixel (bpp).', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('PSNR', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.figure.canvas.draw()

# Function to plot SSIM data
def plot_ssim(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["SSIM"], 'b-', label='SSIM')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('SSIM', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()


def plot_ssim_bpp(ax, rt_frame, st_frame):
    ax.clear()
    zipped_lists = list(zip(st_frame['bpp'], rt_frame['SSIM']))
    x1,y1 = zip(*sorted(zipped_lists))
    ax.scatter(x1, y1, color='red', s = 5, label='SSIM Rate Distortion')
    ax.set_xlabel('Bits per Pixel (bpp).', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('SSIM', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.figure.canvas.draw()


def plot_ref_ssim(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["refSSIM"], 'b-', label='SSIM')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('SSIM', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()


def plot_brisque(ax, st_frame, rt_frame):
    ax.clear()
    ax.plot(st_frame["Rank"], st_frame["refBRISQUE"], 'g-', label='Original')
    ax.plot(rt_frame["Rank"], rt_frame["BRISQUE"], 'b-', label='Decoded')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('BRISQUE', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()


def plot_brisque_bpp(ax, rt_frame, st_frame):
    ax.clear()
    zipped_lists = list(zip(st_frame['bpp'], rt_frame['BRISQUE']))
    x1,y1 = zip(*sorted(zipped_lists))
    ax.scatter(x1, y1, color='red', s = 5, label='BRISQUE Rate Distortion')
    ax.set_xlabel('Bits per Pixel (bpp).', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('BRISQUE', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.figure.canvas.draw()


# Function to plot Bitrate data
def plot_bitrate(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["bit rate (kbps)"], 'b-', label='Bitrate (kbps)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Bitrate (kbps)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()


def plot_bpp(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["bpp"], 'purple', label='Bit Per Pixel')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Bits Per Pixel', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()


def plot_energy(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["encodingEnergy(mJ)"], 'orange', label='Encoding Energy (mJ)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Encoding Energy (mJ)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()

def plot_captured_energy(ax, data):
    ax.clear()
    data["totalEnergy(mJ)"] = data["captureEnergy(mJ)"] + data["encodingEnergy(mJ)"]
    ax.plot(data["Rank"], data["totalEnergy(mJ)"], 'orange', label='Total Energy (mJ)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Total Energy (mJ)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()


def plot_data_size(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["Size(Bytes)"], 'purple', label='Size (Bytes)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Size(Bytes)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()


def plot_packet_size_over_time(ax, data):
    ax.clear()
    ax.plot(data["time"], data["pktSize"], 'green', label='Pacekt Size (Bits)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Pacekt Size (Bits)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()


def plot_ber(ax, data):
    ax.clear()
    ax.plot(data["time"], data["BER"], 'red', label='BER')
    ax.set_xlabel('Time.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('BER', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()

    

def plot_snr(ax, data):
    ax.clear()
    ax.plot(data["time"], data["SNR"], 'red', label='SNR')
    ax.set_xlabel('Time.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('SNR', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()

def plot_signal_lost(ax, data):
    ax.clear()
    ax.plot(data["time"], data["signal_lost(db)"], 'red', label='Signal Lost (db)')
    ax.set_xlabel('Time', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Signal Lost (db)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=1)
    ax.figure.canvas.draw()






