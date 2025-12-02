# decoder.py
# Implements the decoding side of the video compression pipeline.
# Handles reconstruction of ROI and standard main/secondary frames from logs and bitstreams.
# Supports error handling, BER noise application, and reconstruction metrics recording.

import numpy as np
import core.util as util
import cv2
import os
from core.parameters import Parameters as para
import core.trace as trace
import core.main_frame as main_frame
import core.network_losses_model as loss_model
import random
import metrics.quality_metrics as metrics


EXIT_FAILURE = 0


def apply_mask(image, mask):
    """
    Applies a binary mask to the input image, zeroing out non-masked pixels.
    Args:
        image (ndarray): Input image.
        mask (ndarray): Binary mask.
    Returns:
        ndarray: Masked image result.
    """
    mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    result = np.where(mask_binary == 255, image, 0)
    return result


def build_received_video_roi(frame_number):
    """
    Decodes the ROI-based video, reconstructs decoded frames, fills in missing/partial data blocks as needed.
    Updates PSNR, SSIM, and BRISQUE, and saves results. Applies BER noise if necessary for simulation.
    """
    decoded_frame_vector = [trace.frame_decoded() for _ in range(frame_number)]
    zero_frame = np.zeros((para.default_height, para.default_width), dtype=np.uint8)
    ones_frame = np.ones((para.default_height, para.default_width), dtype=np.uint8)
    ones_block = np.ones((8, 8), dtype=np.uint8)
    image_name, mask_name, enhanced_image_name = "", "", ""
    packet_blocks = []
    with open(os.path.join(para.trace_file_path, 'st-packet'), 'r') as trace_file:
        for line in trace_file:
            if line and line[0] != '#' and line[0] != ' ':
                frame_nb, frame_type, layer_nb, packet_blocks, signal_loss,snr, ber = util.parse_rcv_file_roi(line)
                if frame_type == "M":
                    if decoded_frame_vector[frame_nb - 2].frame_type == "N":
                        empty_layers = util.get_empty_layers(frame_nb)
                        all_blocks = [0, para.default_width * para.default_height // 64 - 1]
                        for empty_layer in empty_layers:
                            fill_in_main_frame_roi(decoded_frame_vector[frame_nb - 2], all_blocks, empty_layer, para.level_numbers,ber)

                    if para.interleaving == "torus":
                        fill_in_main_frame_torus(decoded_frame_vector[frame_nb - 2], packet_blocks, layer_nb, para.level_numbers, para.default_height // 8, para.default_width // 8, para.trace_file_path,signal_loss)
                    else:
                        fill_in_main_frame_roi(decoded_frame_vector[frame_nb - 2], packet_blocks, layer_nb, para.level_numbers,ber)

                if frame_type == 'S':
                    decode_second_frame(decoded_frame_vector[frame_nb - 2], packet_blocks, layer_nb)
                    packet_blocks = []
                    decoded_frame_vector[frame_nb - 2].frame_type = frame_type

                decoded_frame_vector[frame_nb - 2].frame_type = frame_type

    if not decoded_frame_vector:
        #print("No frame received !!", image_name)
        exit(EXIT_FAILURE)
    #print(decoded_frame_vector[frame_nb - 2])
    for ind, decoded_frame_struct in enumerate(decoded_frame_vector):
        image_name = os.path.join(para.captured_frame_dir, f"frame{ind + 2}.png")
        orig_frame = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        reference_frame = cv2.imread(os.path.join(para.reference_frames_dir, f"frame{ind + 2}.png"), cv2.IMREAD_GRAYSCALE)
        if reference_frame is None:
            continue
        if decoded_frame_struct.frame_type == "N":
            #Dropped not sent
            continue
            last_received = get_last_received_frame(ind, decoded_frame_vector)
            if last_received >= 0:
                decoded_frame_struct.decoded_frame = decoded_frame_vector[last_received].decoded_frame
        
        if decoded_frame_struct.frame_type == "M":            
            decoded_frame_struct.mask_frame =  cv2.bitwise_or(decoded_frame_struct.sad_map, cv2.bitwise_or(decoded_frame_struct.r_map, decoded_frame_struct.g_map))
            '''
            #print('--------------------------------------------------------------------------------------------------------')
            #print('\n')
            #print('\n')
            #print(f' frame number {ind+1} \n')
            #print('\n')
            #print('\n')
            #print('sad map')
            for row in decoded_frame_struct.sad_map:
                row_str = ' '.join(map(str, row))
                #print(row_str)
            #print('R map')
            for row in decoded_frame_struct.r_map:
                row_str = ' '.join(map(str, row))
                #print(row_str)
            #print('G map')
            for row in decoded_frame_struct.g_map:
                row_str = ' '.join(map(str, row))
                #print(row_str)  
            #print('--------------------------------------------------------------------------------------------------------')
            #print('\n')
            #print('\n')
            #print('\n')
            '''
            captured_frame = get_image(os.path.join(para.roi_frame_dir, f"roi_frame{ind + 2}.png"))
            quantif_frame,blocks_compression_vector = main_frame.get_roi_quantif_frame(captured_frame,decoded_frame_struct.sad_map,decoded_frame_struct.r_map,decoded_frame_struct.g_map,ind + 1)
            decoded_frame_s = decoded_frame_struct.decoded_frame.astype(np.int16)
            decoded_frame_s = np.multiply(decoded_frame_s, quantif_frame)
            decoded_frame_struct.decoded_frame = main_frame.get_roi_reference_frame(decoded_frame_s,decoded_frame_struct.sad_map,decoded_frame_struct.r_map,decoded_frame_struct.g_map)
            if para.enhance_method != "":
                enhancedMat = cv2.inpaint(decoded_frame_struct.decoded_frame, decoded_frame_struct.mask_frame,3, cv2.INPAINT_TELEA)
                decoded_frame_struct.decodedFrame = enhancedMat 
            noisy_image = process_image_with_ber(decoded_frame_struct.decoded_frame,decoded_frame_struct.ber_matrix)  
            decoded_frame_struct.decoded_frame = noisy_image 
            decoded_frame_struct.PSNR = metrics.calculate_psnr(decoded_frame_struct.decoded_frame, orig_frame)
            decoded_frame_struct.SSIM = metrics.calculate_ssim(decoded_frame_struct.decoded_frame, orig_frame)
            decoded_frame_struct.BRISQUE = metrics.calculate_brisque(cv2.cvtColor(decoded_frame_struct.decoded_frame, cv2.COLOR_GRAY2BGR))
            util.write_decoded_frame_record(ind + 1, decoded_frame_struct)
            image_name = os.path.join(para.decoded_frames_dir, f"frame{ind + 2}.png")
            cv2.imwrite(image_name, decoded_frame_struct.decoded_frame)





def build_received_video(frame_number):
    """
    Decodes the standard video, reconstructs decoded frames, fills in missing/partial data blocks as needed.
    Updates PSNR, SSIM, and BRISQUE, and saves results. Applies BER noise if necessary for simulation.
    """
    decoded_frame_vector = [trace.frame_decoded() for _ in range(frame_number)]
    zero_frame = np.zeros((para.default_height, para.default_width), dtype=np.uint8)
    ones_frame = np.ones((para.default_height, para.default_width), dtype=np.uint8)
    ones_block = np.ones((8, 8), dtype=np.uint8)

    image_name, mask_name, enhanced_image_name = "", "", ""
    packet_blocks = []

    with open(os.path.join(para.trace_file_path, 'st-packet'), 'r') as trace_file:
        for line in trace_file:
            if line and line[0] != '#' and line[0] != ' ':
                frame_nb, frame_type, layer_nb, packet_blocks, signal_loss,snr,ber = util.parse_rcv_file(line)
                if frame_type == "M":
                    if decoded_frame_vector[frame_nb - 1].frame_type == "N":
                        empty_layers = util.get_empty_layers(frame_nb)
                        all_blocks = [0, para.default_width * para.default_height // 64 - 1]
                        for empty_layer in empty_layers:
                            fill_in_main_frame(decoded_frame_vector[frame_nb - 1], all_blocks, empty_layer, para.level_numbers,ber)

                    if para.interleaving == "torus":
                        fill_in_main_frame_torus(decoded_frame_vector[frame_nb - 1], packet_blocks, layer_nb, para.level_numbers, util.videoP.frame_height // 8, util.videoP.frame_width // 8, util.simP.trace_path,signal_loss)
                    else:
                        fill_in_main_frame(decoded_frame_vector[frame_nb - 1], packet_blocks, layer_nb, para.level_numbers,ber)

                if 'S' in frame_type:
                    decode_second_frame(decoded_frame_vector[frame_nb - 1], packet_blocks, layer_nb,ber)
                    packet_blocks = []
                    decoded_frame_vector[frame_nb - 1].frame_type = frame_type

                decoded_frame_vector[frame_nb - 1].frame_type = frame_type

    if not decoded_frame_vector:
        #print("No frame received !!", image_name)
        exit(EXIT_FAILURE)

    for ind, decoded_frame_struct in enumerate(decoded_frame_vector):
        image_name = os.path.join(para.captured_frame_dir, f"frame{ind + 1}.png")
        orig_frame = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        reference_frame = get_image(os.path.join(para.reference_frames_dir, f"frame{ind + 1}.png"))

        if decoded_frame_struct.frame_type == "N":
            last_received = get_last_received_frame(ind, decoded_frame_vector)
            if last_received >= 0:
                decoded_frame_struct.decoded_frame = decoded_frame_vector[last_received].decoded_frame

        if decoded_frame_struct.frame_type == "M":
            #print("mframe")
            for i in range(0, para.default_height, 8):
                for j in range(0, para.default_width, 8):
                    tbloc = decoded_frame_struct.mask_frame[i:i + 8, j:j + 8]
                    if np.any(tbloc != 0):  # Check if any element in tbloc is not equal to 0
                        tbloc[:] = 255  # Set all elements in tbloc to 255
                    if np.sum(tbloc) < 255 * 64:
                        tbloc[:] = 0
            captured_frame = get_image(os.path.join(para.captured_frame_dir, f"frame{ind + 1}.png"))
            quantif_frame = main_frame.getQuantifFrame(captured_frame)
            decoded_frame_s = decoded_frame_struct.decoded_frame.astype(np.int16)
            decoded_frame_s = np.multiply(decoded_frame_s, quantif_frame)
            decoded_frame_struct.decoded_frame = main_frame.getReferenceFrame(decoded_frame_s)
            if para.enhance_method != "":
                enhancedMat = cv2.inpaint(decoded_frame_struct.decoded_frame, decoded_frame_struct.mask_frame,3, cv2.INPAINT_TELEA)
                decoded_frame_struct.decodedFrame = enhancedMat

        if decoded_frame_struct.frame_type[0] == 'S':
            #print("sframe")
            main_frame_nb = int(decoded_frame_struct.frame_type[1:])
            decoded_frame_struct.decoded_frame = np.multiply(decoded_frame_struct.decoded_frame, reference_frame)
            main_frame_nb = int(decoded_frame_struct.frame_type[1:])
            if decoded_frame_vector[main_frame_nb - 1].frame_type == "N" or main_frame_nb == -1:
                decoded_frame_struct.frame_type = "N"
                last_received = get_last_received_frame(ind, decoded_frame_vector)
                if last_received >= 0:
                    decoded_frame_struct.decoded_frame = decoded_frame_vector[last_received].decoded_frame
                continue

            decoded_frame_struct.frame_type = "S"
            image_name = os.path.join(para.decoded_frames_dir, f"frame{main_frame_nb}.png")
            decoded_main_frame = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

            for i in range(0, decoded_main_frame.shape[0], 8):
                for j in range(0, decoded_main_frame.shape[1], 8):
                    tempBlock = decoded_frame_struct.decoded_frame[i:i + 8, j:j + 8]
                    if np.count_nonzero(tempBlock) == 0:
                        tempBlock += decoded_main_frame[i:i + 8, j:j + 8]
                        decoded_frame_struct.decoded_frame[i:i + 8, j:j + 8] = tempBlock
                        #else decode the received blocks
                        ## First load the ref frame then we process it and make the other non received blocks to 0 
                        ## then when we get to this point after decoding the m frame like the s frame we replace the non sent blocks with there symetric from the m frame
        noisy_image = process_image_with_ber(decoded_frame_struct.decoded_frame,decoded_frame_struct.ber_matrix)  
        decoded_frame_struct.decoded_frame = noisy_image   
        decoded_frame_struct.PSNR = metrics.calculate_psnr(decoded_frame_struct.decoded_frame, orig_frame)
        decoded_frame_struct.SSIM = metrics.calculate_ssim(decoded_frame_struct.decoded_frame, orig_frame)
        decoded_frame_struct.BRISQUE = metrics.calculate_brisque(cv2.cvtColor(decoded_frame_struct.decoded_frame, cv2.COLOR_GRAY2BGR))
        util.write_decoded_frame_record(ind + 1, decoded_frame_struct)
        image_name = os.path.join(para.decoded_frames_dir, f"frame{ind + 1}.png")
        cv2.imwrite(image_name, decoded_frame_struct.decoded_frame)





def fill_in_main_frame_roi(frame_record, packet_blocks, layer_nb, layers_nb,  signal_loss):
    """
    Fills in the main frame ROI with decoded data, updates BER matrix, and applies BER noise.
    Args:
        frame_record (trace.frame_decoded): The frame record to update.
        packet_blocks (list): List of block indices and compression levels.
        layer_nb (int): The current layer number.
        layers_nb (int): Total number of layers.
        signal_loss (float): Signal loss value for the current block.
    """
    for block in packet_blocks:
        assert len(block) >= 2
    for block in packet_blocks:
        start, end, compression_level = block
        for ind in range(start, end + 1):
            i, j = util.get_row_col(ind, frame_record.decoded_frame.shape[1])
            block = frame_record.decoded_frame[i:i + 8, j:j + 8]
            if compression_level == 'H':
                frame_record.sad_map[i:i + 8, j:j + 8] = 255
            elif compression_level == 'L':
                frame_record.r_map[i:i + 8, j:j + 8] = 255
            elif compression_level == 'N':
                frame_record.g_map[i:i + 8, j:j + 8] = 255
            height, width = frame_record.ber_matrix.shape[:2]
            x_ber,y_ber = util.get_row_col_ber_matrix(ind,width)
            frame_record.ber_matrix[x_ber,y_ber] = float(signal_loss)
            fill_layer(block, layer_nb, layers_nb)

def fill_in_main_frame(frame_record, packet_blocks, layer_nb, layers_nb,signal_lost):
    """
    Fills in the main frame with decoded data, updates BER matrix, and applies BER noise.
    Args:
        frame_record (trace.frame_decoded): The frame record to update.
        packet_blocks (list): List of block indices and compression levels.
        layer_nb (int): The current layer number.
        layers_nb (int): Total number of layers.
        signal_lost (float): Signal loss value for the current block.
    """
    assert len(packet_blocks) >= 2
    
    for ind in range(packet_blocks[0], packet_blocks[1] + 1):
        i, j = util.get_row_col(ind, frame_record.decoded_frame.shape[1])
        block = frame_record.decoded_frame[i:i + 8, j:j + 8]
        height, width = frame_record.ber_matrix.shape[:2]
        x_ber,y_ber = util.get_row_col_ber_matrix(ind,width)
        frame_record.ber_matrix[x_ber,y_ber] = float(signal_lost)
        fill_layer(block, layer_nb, layers_nb)


def fill_in_main_frame_torus(frame_record, packet_blocks, layer_nb, layers_nb, blocks_in_row, blocks_in_col, trace_path,signal_lost):
    """
    Fills in the main frame using torus interleaving, updates BER matrix, and applies BER noise.
    Args:
        frame_record (trace.frame_decoded): The frame record to update.
        packet_blocks (list): List of block indices.
        layer_nb (int): The current layer number.
        layers_nb (int): Total number of layers.
        blocks_in_row (int): Number of blocks in a row for torus interleaving.
        blocks_in_col (int): Number of blocks in a column for torus interleaving.
        trace_path (str): Path to the trace file.
        signal_lost (float): Signal loss value for the current block.
    """
    assert len(packet_blocks) >= 2

    for block_nb in range(packet_blocks[0], packet_blocks[1] + 1):
        i, j = get_row_col_torus(trace_path, block_nb)
        block = frame_record.decoded_frame[i:i+8, j:j+8]
        height, width = frame_record.ber_matrix.shape[:2]
        x_ber,y_ber = util.get_row_col_ber_matrix(ind,width)
        frame_record.ber_matrix[x_ber,y_ber] = float(signal_lost)
        fill_layer(block, layer_nb, layers_nb)


def decode_second_frame(frame_record, packet_blocks, layer_nb,signal_lost):
    """
    Decodes the second frame, which contains the main frame's reference frame.
    Args:
        frame_record (trace.frame_decoded): The frame record to update.
        packet_blocks (list): List of block indices.
        layer_nb (int): The current layer number.
        signal_lost (float): Signal loss value for the current block.
    """
    assert len(packet_blocks) >= 1
    if(len(packet_blocks)>1):
        for block_nb in range(packet_blocks[0], packet_blocks[1] + 1):
            i, j = util.get_row_col(block_nb, frame_record.decoded_frame.shape[1])
            block = frame_record.decoded_frame[i:i+8, j:j+8]
            height, width = frame_record.ber_matrix.shape[:2]
            x_ber,y_ber = util.get_row_col_ber_matrix(block_nb,width)
            frame_record.ber_matrix[x_ber,y_ber] = float(signal_lost)
            block[:, :] = 1
            frame_record.decoded_frame[i:i+8, j:j+8] = block
        else:
            block_nb = packet_blocks[0]
            i, j = util.get_row_col(block_nb, frame_record.decoded_frame.shape[1])
            block = frame_record.decoded_frame[i:i+8, j:j+8]
            height, width = frame_record.ber_matrix.shape[:2]
            x_ber,y_ber = util.get_row_col_ber_matrix(block_nb,width)
            frame_record.ber_matrix[x_ber,y_ber] = float(signal_lost)
            block[:, :] = 1
            frame_record.decoded_frame[i:i+8, j:j+8] = block



def get_last_received_frame(ind, decoded_frame_vector):
    """
    Finds the index of the last received frame in the vector.
    Args:
        ind (int): The current frame index.
        decoded_frame_vector (list): List of frame records.
    Returns:
        int: Index of the last received frame, or -1 if no frame received.
    """
    if decoded_frame_vector and ind > 0:
        for i in range(ind - 1, -1, -1):
            return i
    return -1


def fill_layer(block, layer_nb, layers_nb):
    """
    Fills a single layer of the main frame.
    Args:
        block (ndarray): The block to fill.
        layer_nb (int): The current layer number.
        layers_nb (int): Total number of layers.
    """

    if layer_nb == 0:
        block[0, 0] = 1
    if layer_nb == 12:
        block[7, 7] = 1

    x, y = 0, 0

    if layer_nb <= 6:
        x = 0
        y = layer_nb + 1
        while y >= 0:
            block[x, y] = 1
            x += 1
            y -= 1
    else:
        x = 7
        y = layer_nb - 6
        while y <= 7:
            block[x, y] = 1
            x -= 1
            y += 1

    if layer_nb == layers_nb - 1:
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                if i + j >= layers_nb:
                    block[i, j] = 1

def get_image(image_name):
    """
    Reads an image from the given file path.
    Args:
        image_name (str): Path to the image file.
    Returns:
        ndarray: The loaded image.
    """
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    if image is None:
        #print(f"Could not open or find the image {image_name}")
        exit(EXIT_FAILURE)
    return image




def apply_ber_to_block(block, ber):
    """
    Applies BER noise to a single block.
    Args:
        block (ndarray): The block to apply BER noise to.
        ber (float): Bit error rate for the block.
    Returns:
        ndarray: Block with BER noise applied.
    """
    block_flat = block.flatten()
    total_bits = block_flat.size * 8  # Total bits in the block
    num_errors = int(total_bits * ber)  # Number of bits to flip

    for _ in range(num_errors):
        bit_to_flip = random.randint(0, total_bits - 1)
        byte_index = bit_to_flip // 8
        bit_index = bit_to_flip % 8
        block_flat[byte_index] ^= 1 << bit_index

    return block_flat.reshape(block.shape)

def process_image_with_ber(image, ber_matrix, block_size=(8, 8)):
    """
    Applies BER noise to the entire image based on a BER matrix.
    Args:
        image (ndarray): The input image.
        ber_matrix (ndarray): A 2D array of BER values corresponding to the image blocks.
        block_size (tuple): Size of blocks to process (default: (8, 8)).
    Returns:
        ndarray: Image with BER noise applied.
    """
    height, width = image.shape[:2]
    noisy_image = np.zeros_like(image)

    for y in range(0, height, block_size[1]):
        for x in range(0, width, block_size[0]):
            block = image[y:y+block_size[1], x:x+block_size[0]]
            ber_value = ber_matrix[y // block_size[1]][x // block_size[0]]
            noisy_block = apply_ber_to_block(block, ber_value)
            noisy_image[y:y+block_size[1], x:x+block_size[0]] = noisy_block

    return noisy_image