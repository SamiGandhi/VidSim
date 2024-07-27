import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

def plot_psnr(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["PSNR"], 'g-', label='PSNR')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('PSNR', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()

def plot_ref_psnr(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["refPSNR"], 'g-', label='PSNR')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('PSNR', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()

# Function to plot SSIM data
def plot_ssim(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["SSIM"], 'b-', label='SSIM')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('SSIM', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()

def plot_ref_ssim(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["refSSIM"], 'b-', label='SSIM')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('SSIM', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()


# Function to plot Bitrate data
def plot_bitrate(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["bit rate (kbps)"], 'b-', label='Bitrate (kbps)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Bitrate (kbps)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()


def plot_energy(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["encodingEnergy(mJ)"], 'orange', label='Encoding Energy (mJ)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Encoding Energy (mJ)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()

def plot_captured_energy(ax, data):
    ax.clear()
    data["totalEnergy(mJ)"] = data["captureEnergy(mJ)"] + data["encodingEnergy(mJ)"]
    ax.plot(data["Rank"], data["totalEnergy(mJ)"], 'orange', label='Total Energy (mJ)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Total Energy (mJ)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()


def plot_data_size(ax, data):
    ax.clear()
    ax.plot(data["Rank"], data["Size(Bytes)"], 'purple', label='Size (Bytes)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Size(Bytes)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()


def plot_packet_size_over_time(ax, data):
    ax.clear()
    ax.plot(data["time"], data["pktSize"], 'green', label='Pacekt Size (Bits)')
    ax.set_xlabel('Frame Seq.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Pacekt Size (Bits)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()


def plot_ber(ax, data):
    ax.clear()
    ax.plot(data["time"], data["BER"], 'red', label='BER')
    ax.set_xlabel('Time.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('BER', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()

def plot_snr(ax, data):
    ax.clear()
    ax.plot(data["time"], data["SNR"], 'red', label='SNR')
    ax.set_xlabel('Time.', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('SNR', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()

def plot_signal_lost(ax, data):
    ax.clear()
    ax.plot(data["time"], data["signal_lost(db)"], 'red', label='Signal Lost (db)')
    ax.set_xlabel('Time', fontsize= 'small', fontweight = 'bold')
    ax.set_ylabel('Signal Lost (db)', fontsize= 'small', fontweight = 'bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0), prop={'weight': 'bold', 'size': 'xx-small'}, edgecolor='black')
    plt.grid(True)
    ax.figure.canvas.draw()



