# quality_metrics.py
# Provides core image/video quality metric calculations (PSNR, SSIM, BRISQUE) for pipeline evaluation.
# Includes functions to extract blocks and compute detailed per-block and global visual quality statistics.

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import pytest
from metrics.brisque import BRISQUE
import cv2
import os
import re
import core.util
import core.parameters as parameters

# NO NEED TO EXTRACT THE BLOCKS THEN CALCULATE THE QUALITY METRICS ONLY IF WANT TO COMPARE ROI WITH NO ROI

# Calculate PSNR

def calculate_psnr(ref_image, img):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between reference and processed image.
    Args:
        ref_image: Reference image (ground truth)
        img:       Processed or decoded image
    Returns:
        float: PSNR value
    """
    psnr = peak_signal_noise_ratio(ref_image, img)
    return psnr

# Calculate SSIM

def calculate_ssim(ref_img, img):
    """
    Compute Structural Similarity Index (SSIM) for perceptual image quality between reference and processed frame.
    Args:
        ref_img: Reference image (ground truth)
        img:     Processed or decoded image
    Returns:
        float: SSIM value
    """
    ssim = structural_similarity(ref_img, img, window_size=8)
    return ssim

def calculate_brisque(img):
    """
    Compute the BRISQUE (perceptual no-reference image quality) score for RGB images.
    Args:
        img: RGB (3-channel) image
    Returns:
        float: BRISQUE quality score
    """
    obj = BRISQUE(url=False)
    return obj.score(img)

def grayscale_to_rgb(image):
    """
    Converts a grayscale image (2D array) to a 3-channel RGB image.
    Args:
        image: A 2D NumPy array representing the grayscale image.
    Returns:
        A 3-channel NumPy array representing the RGB image.
    """
    if len(image.shape) == 2:  # Check if grayscale
        rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    else:
        rgb_image = image
    return rgb_image



def extract_blocks_from_string(string, roi_blocks):
    # Define a regular expression pattern to match numbers
    pattern = r'(\d+)->(\d+)([A-Za-z])'

    # Search for the pattern in the input string
    match = re.search(pattern, string)

    if match:
        first_digit = int(match.group(1))  # First set of digits
        second_digit = int(match.group(2)) # Second set of digits
        letter = match.group(3)       # Letter
        
    smaller = min(first_digit, second_digit)
    larger = max(first_digit, second_digit)
  
    # Generate the list of numbers between start and end (inclusive)
    numbers_between = list(range(smaller, larger + 1))
    list_ = []
    if letter == 'N':
        for block in numbers_between:
            list_.append([block,'N'])
        roi_blocks.extend(list_)
    elif letter == 'H':
        for block in numbers_between:
            list_.append([block,'H'])
        roi_blocks.extend(list_)
    elif letter == 'L':
        for block in numbers_between:
            list_.append([block,'L'])
        roi_blocks.extend(list_)
    return roi_blocks




##Calculate the quality metrics for roi method

## st packet


def calculate_quality_metrics_roi(st_frame, st_packet, frame_nb, line_index=1):
    # Opening and read  trace file
    with open(st_packet_filepath, 'r') as trace_file:
        trace_content = trace_file.readlines()
    #skip the headers line by setting the index = 1 -> 2nd line
    #now skip the lines of already calculated frames
    while line_index < len(trace_content):
        line = trace_content[line_index]
        values = line.split('\t')
        frame_number = values[3]
        if int(frame_number) == frame_nb:
            break
        line_index += 1
    decoded_frames = parameters.decoded_frames_dir
    for filename in os.listdir(decoded_frames):
        match = re.match(pattern, filename)
        if match:
            # Extract the sequence number as an integer
            sequence_number = int(match.group(1))
            #without the frame number 1
            if sequence_number == 1:
                continue
    # Load the image and get all of its roi blocks
    captured_image_filename = os.path.join(parameters.captured_frame_dir,f'frame{frame_nb}.png')
    image_filename = os.path.join(parameters.decoded_frames_dir,f'frame{frame_nb}.png')
    captured_image_ = cv2.imread(captured_image_filename)
    image_ = cv2.imread(image_filename)
    image = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
    captured_image = cv2.cvtColor(captured_image_, cv2.COLOR_BGR2GRAY)
    while line_index < len(trace_content):
        line = trace_content[line_index]
        values = line.split('\t')
        frame_number = values[3]
        line_blocks_str = values[6].split(' ')
        for block_str in line_blocks_str:
            roi_blocks.append(extract_blocks_from_string(block_str, roi_blocks)) 
        if(frame_number != frame_nb):
            break
        line_index += 1
    
    blocks_decoded = []
    blocks_black = []
    captured_blocks = []
    three_channels_blocks = []
    black_image = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

    # Divide the image into 8x8 blocks
    for row in range(0, image.shape[0], 8):
        for col in range(0, image.shape[1], 8):
            block = image[row:row+8, col:col+8]
            blocks_decoded.append(block)

    for row in range(0, black_image.shape[0], 8):
        for col in range(0, black_image.shape[1], 8):
            block = black_image[row:row+8, col:col+8]
            blocks_black.append(block)

    for row in range(0, captured_image.shape[0], 8):
        for col in range(0, captured_image.shape[1], 8):
            block = captured_image[row:row+8, col:col+8]
            captured_blocks.append(block)
    
    for row in range(0, image_.shape[0], 8):
        for col in range(0, image.shape[1], 8):
            block = image_[row:row+8, col:col+8]
            three_channels_blocks.append(block)

    ## now iterate over all the roi blocks and replace them from the image to the black image
    for roi_block,Level in roi_blocks:
        try:
                #extract the roi_block from the image 
            block = blocks_decoded[roi_block]
            blocks_black[roi_block] = block
        except TypeError:
            if roi_block[1] == 'N' :
                red_block = np.array([255, 0, 0], dtype=np.uint8) * np.ones((8, 8, 3), dtype=np.uint8)
                blocks_black[roi_block[0]] = red_block
            elif roi_block[1] == 'H':
                red_block = np.array([0, 255, 0], dtype=np.uint8) * np.ones((8, 8, 3), dtype=np.uint8)
                blocks_black[roi_block[0]] = red_block
            elif roi_block[1] == 'L':
                red_block = np.array([0, 0, 255], dtype=np.uint8) * np.ones((8, 8, 3), dtype=np.uint8)
                blocks_black[roi_block[0]] = red_block

    image_height = black_image.shape[0]
    image_width = black_image.shape[1]

    for i, block in enumerate(blocks_black):
        if np.any(block>0):
            continue
        else:
            blocks_black[i] = captured_blocks[i]

        
    '''
    '''
    #rebuild the black frame and show it to confirm
    # 
    # Create an empty image to restore the blocks

    

    reconstructed_image  = np.zeros((image_height, image_width), dtype=np.uint8)
    # Iterate over the blocks and place them in the restored image
    
    
    for i, block in enumerate(blocks_black):
        # Calculate the position of the current block
        row = i // 36
        col = i % 36
        
        start_row = row * 8
        start_col = col * 8
        block_2d = np.squeeze(block)
        # Place the block in the corresponding position
        reconstructed_image[start_row:start_row+8, start_col:start_col+8] = block_2d

    ## calculate the psrn ssim and brisque

    ssim = calculate_ssim(captured_image,reconstructed_image)
    psnr = calculate_psnr(captured_image,reconstructed_image)
    reconstructed_image_3ch = cv2.cvtColor(reconstructed_image, cv2.COLOR_GRAY2BGR)
    brisque_current = calculate_brisque(reconstructed_image_3ch)
    brisque_original = calculate_brisque(captured_image_)



    
    