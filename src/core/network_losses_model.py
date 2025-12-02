import numpy as np
from core.parameters import Parameters as para
import math
# Speed of light in m/s
c = 3 * 10**8

def free_space_path_loss(d, f, G=1, d0=1):
    """
    Calculate the free-space path loss.
    :param d: Distance between transmitter and receiver in meters.
    :param f: Frequency in Hz.
    :param G: Antenna gain.
    :param d0: Reference distance, usually 1 meter.
    :return: Path loss in dB.
    """
    PL = 20 * np.log10(d/d0) + 20 * np.log10(f) + 20 * np.log10(4 * np.pi * d0 / c) - G
    return PL

def rayleigh_fading(scale=1):
    """
    Simulate Rayleigh fading.
    :param scale: Scale parameter (sigma) for the distribution.
    :return: Fading coefficient.
    """
    return np.random.rayleigh(scale)

def rician_fading(s, sigma):
    """
    Simulate Rician fading.
    :param s: Line-of-sight component.
    :param sigma: Scale parameter for the non-line-of-sight components.
    :return: Fading coefficient.
    """
    real_part = s + np.random.normal(0, sigma)
    imag_part = np.random.normal(0, sigma)
    return np.sqrt(real_part**2 + imag_part**2)

def calculate_noise_power(bandwidth_hz, noise_figure_db):
    return -174 + 10 * np.log10(bandwidth_hz) + noise_figure_db



def calculate_ber(snr_db):
    snr_linear = 10 ** (snr_db / 10)
    ber = 0.5 * math.erfc(np.sqrt(snr_linear))
    return ber

def calculate_vegetation_loss(f, vegetation_density):
    """
    Estimate signal loss due to vegetation.
    :param f: Frequency in Hz.
    :param vegetation_density: Density of vegetation (arbitrary units).
    :return: Loss in dB.
    """
    return vegetation_density * 0.2 * np.log10(f)

def calculate_humidity_loss(f, humidity):
    """
    Estimate signal loss due to humidity.
    :param f: Frequency in Hz.
    :param humidity: Relative humidity (percentage).
    :return: Loss in dB.
    """
    return humidity * 0.1 * np.log10(f)



def combined_loss_model(d, f, environment, humidity=0, vegetation_density=0, G=1, d0=1, s=1, sigma=1):
    """
    Combined signal loss model.
    :param d: Distance in meters.
    :param f: Frequency in Hz.
    :param environment: Type of environment ('urban', 'suburban', 'wet_zone', 'wildlife_zone').
    :param humidity: Humidity percentage.
    :param vegetation_density: Vegetation density.
    :param G: Antenna gain.
    :param d0: Reference distance.
    :param s: Rician fading LOS component.
    :param sigma: Rician/ Rayleigh fading scale parameter.
    :return: Total signal loss in dB.
    """
    if environment != 'None':
        total_loss_db = free_space_path_loss(d, f, G, d0)
        if environment in ["wetland", "wildlife_zone"]:
            total_loss_db += calculate_humidity_loss(f/1e6, humidity)
            total_loss_db += calculate_vegetation_loss(f/1e6, vegetation_density)
            noise_power_dbm = calculate_noise_power(f, noise_figure_db)
            if environment in ["urban", "suburban", "wetland"]:
                fading_loss = rayleigh_fading(scale=sigma)
            elif environment == "line_of_sight":
                fading_loss = rician_fading(s, sigma)
            else:
                fading_loss = 0
            total_loss_db -= fading_loss
            snr_db = transmitted_signal_power_dbm - total_loss_db - noise_power_dbm
              # No additional fading considered
            return snr_db, total_loss_db
    else:
        return 0


transmitted_signal_power_dbm = 22.5  # Transmitted Signal Power in dBm
bandwidth_hz = 125000  # Bandwidth in Hz
noise_figure_db = 5  # Noise Figure in dB





















'''
# Example usage of the model
distance = 1  # meters
frequency = 2.4e9  # Hz (2.4 GHz, common for Wi-Fi)
environment = "wildlife_zone"
humidity_level = 70  # percent
vegetation_density_level = 5  # arbitrary unit

loss = combined_loss_model(distance, frequency, environment, humidity_level, vegetation_density_level)
#print(f"Total Signal Loss: {loss} dB")
'''



def apply_loss_to_image(img, signal_loss_db):
    # Calculate the BER
    ber = calculate_ber(signal_loss_db)

    # Determine the number of pixels to affect
    total_pixels = img.shape[0] * img.shape[1]
    num_affected_pixels = int(total_pixels * ber)

    # Randomly choose pixels to turn black
    for _ in range(num_affected_pixels):
        x = np.random.randint(0, img.shape[1])  # Random x-coordinate
        y = np.random.randint(0, img.shape[0])  # Random y-coordinate
        img[y, x] = 0  # Set pixel to black

    return img



def apply_ber_to_block(block, ber):
    # Convert block to binary array
    block_binary = np.unpackbits(block)

    # Apply BER
    for i in range(len(block_binary)):
        if random.random() < ber:
            block_binary[i] = 1 - block_binary[i]  # Flipping the bit

    # Reconstruct block from binary array
    return np.packbits(block_binary).reshape(block.shape)

def process_image_with_ber(image, ber, block_size=(8, 8)):
    height, width = image.shape[:2]
    noisy_image = np.zeros_like(image)

    # Iterate over the image in block_size steps
    for y in range(0, height, block_size[1]):
        for x in range(0, width, block_size[0]):
            block = image[y:y+block_size[1], x:x+block_size[0]]

            noisy_block = apply_ber_to_block(block, ber)
            noisy_image[y:y+block_size[1], x:x+block_size[0]] = noisy_block

    return noisy_image

