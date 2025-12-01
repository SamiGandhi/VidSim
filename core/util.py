# util.py
# Utility functions for directory management, file tracing, metric calculations, and mathematical/image helpers.
# Used throughout the video coding/decoding pipeline for shared operations.

import os
import cv2
import numpy as np
import errno
import math
import re
from core.parameters import Parameters as para


def make_dir(dirName):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(dirName, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            #print(f"Error while creating directory {dirName}")
            exit(EXIT_FAILURE)


def deleteFolderContent(dirName):
    """Recursively remove all files from a directory."""
    try:
        for filename in os.listdir(dirName):
            filepath = os.path.join(dirName, filename)
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.unlink(filepath)
    except Exception as e:
        raise RuntimeError(f"Error while deleting folder content in {dirName}: {e}")
        #print(f"Error while deleting folder content in {dirName}: {e}")


def create_trace_files(tracePath):
    """Creates and initializes trace files for frame/packet logging."""
    frameTraceName = os.path.join(tracePath, "st-frame")
    with open(frameTraceName, "w") as traceFile:
        traceFile.write("#Rank\tType\tSize(Bytes)\trefPSNR\trefSSIM\tBRISQUE\tbpp\tlayers Size (bits)\tcaptureEnergy(mJ)\tencodingEnergy(mJ)\tbit rate (kbps)\n")

    packetTraceName = os.path.join(tracePath, "st-packet")
    with open(packetTraceName, "w") as traceFile:
        traceFile.write("#time\tseqNb\tpktSize\tframeNb\tframeType\tlayerNb\tblocksList\tsignal_lost(db)\tSNR\tBER\n")

    packetTraceName = os.path.join(tracePath, "rt-frame")
    with open(packetTraceName, "w") as traceFile:
        traceFile.write("#Rank\tframeType\tPSNR\tSSIM\tBRISQUE\n")


def write_decoded_frame_record(frame_nb, frame_record):
    """Appends a record of the decoded frame's metrics to the trace file."""
    with open(para.trace_file_path + "/rt-frame", "a") as trace_file:
        if not trace_file:
            #print(f"Problem opening file ... {trace_path}/rt-frame")
            return
        trace_file.write(f"{frame_nb}\t{frame_record.frame_type}\t{frame_record.PSNR:.12f}\t{frame_record.SSIM:.12f}\t{frame_record.BRISQUE:.12f}\n")


def writePacketRecordS(packetRecord, tracePath,signal_loss,snr,ber_value):
    """Appends a packet record (S-frame) to the packet trace file. Handles block vector formatting."""
    packetTraceName = tracePath + "/st-packet"
    with open(packetTraceName, "a") as traceFile:
        traceFile.write(f"{packetRecord.send_time}\t{packetRecord.seq_nb}\t{int(np.ceil(packetRecord.packet_size/8.0))}\t{packetRecord.frame_nb}\t{packetRecord.frame_type}\t{packetRecord.layer_nb}\t")
        if packetRecord.frame_type == "S":
            suite = False
            for ind in range(len(packetRecord.blockSeqVector)):
                if ind == 0:
                    traceFile.write(str(packetRecord.block_seq_vector[ind]))
                elif packetRecord.block_seq_vector[ind] == packetRecord.bloblock_seq_vectorckSeqVector[ind-1] + 1:
                    if ind == len(packetRecord.block_seq_vector) - 1:
                        traceFile.write(f"-{packetRecord.block_seq_vector[ind]}")
                    else:
                        suite = True
                elif suite:
                    traceFile.write(f"-{packetRecord.block_seq_vector[ind-1]} {packetRecord.block_seq_vector[ind]}")
                    suite = False
                else:
                    traceFile.write(f" {packetRecord.block_seq_vector[ind]}")
            traceFile.write(f"\t{signal_loss}\t{snr}\t{ber_value}\n")
        else:
            for ind in range(len(packetRecord.block_seq_vector)):
                traceFile.write(f"{packetRecord.block_seq_vector[ind]} ")
            traceFile.write(f"\t{signal_loss}\t{snr}\t{ber_value}\n")


def compress_vectors_to_string(block_numbers, compressions):
    """Compress block indices and their compression types to a compact string for trace logs."""
    compressed_string = ""
    current_range = [block_numbers[0]]
    #here corrected must take in consederation the sequences
    for i in range(1, len(block_numbers)):
        if compressions[i] != compressions[i - 1] or block_numbers[i]-1 != block_numbers[i - 1]:
            if len(current_range) == 1:
                compressed_string += str(current_range[0]) + "->" + str(current_range[0]) + compressions[i - 1] + " "
            else:
                compressed_string += str(current_range[0]) + "->" + str(current_range[-1]) + compressions[i - 1] + " "
            current_range = [block_numbers[i]]
        else:
            current_range.append(block_numbers[i])
    # Handle the last range
    if len(current_range) == 1:
        compressed_string += str(current_range[0]) + "->" + str(current_range[0]) + compressions[-1]
    else:
        compressed_string += str(current_range[0]) + "->" + str(current_range[-1]) + compressions[-1]
    return compressed_string


def write_packet_record_roi_m_frame(packet_record, trace_path, signal_loss,snr,ber_value):
    """Write a ROI M-frame packet record with compressed block string format."""
    packet_trace_name = f"{trace_path}/st-packet"
    with open(packet_trace_name, "a") as trace_file:
        blocks = compress_vectors_to_string(packet_record.block_seq_vector, packet_record.blocks_compression_vector)
        trace_file.write(f"{packet_record.send_time}\t{packet_record.seq_nb}\t{math.ceil(packet_record.packet_size / 8.0)}\t{packet_record.frame_nb}\t{packet_record.frame_type}\t{packet_record.layer_nb}\t{blocks}\t{signal_loss}\t{snr}\t{ber_value}\n")


def write_packet_record_m_frame(packet_record, trace_path,signal_loss,snr,ber_value):
    """Write a main-frame packet record to trace log."""
    packet_trace_name = f"{trace_path}/st-packet"
    with open(packet_trace_name, "a") as trace_file:
        first_block = packet_record.block_seq_vector[0]
        last_block = packet_record.block_seq_vector[len(packet_record.block_seq_vector) - 1]
        trace_file.write(f"{packet_record.send_time}\t{packet_record.seq_nb}\t{math.ceil(packet_record.packet_size / 8.0)}\t{packet_record.frame_nb}\t{packet_record.frame_type}\t{packet_record.layer_nb}\t{first_block} {last_block}\t{signal_loss}\t{snr}\t{ber_value}\n")


def write_frame_record(frame_record):
    """Log metrics and compression details for a frame to the st-frame trace file."""
    frame_trace_name = para.trace_file_path + "/st-frame"
    with open(frame_trace_name, "a") as trace_file:
        trace_file.write(f"{frame_record.frameNb}\t"
                         f"{frame_record.frameType}\t"
                         f"{int(frame_record.frameSize/8.0 + 0.5)}\t"
                         f"{frame_record.PSNR:.12f}\t"
                         f"{frame_record.SSIM:.12f}\t"
                         f"{frame_record.BRISQUE:.12f}\t"
                         f"{frame_record.bpp:.12f}\t")
        if frame_record.frameType == "M":
            trace_file.write(" ".join(map(str, frame_record.layersSizeVector)))
        else:
            trace_file.write("-")
        trace_file.write(f"\t{frame_record.captureEnergy:.12f}\t"
                         f"{frame_record.encodingEnergy:.12f}\t"
                         f"{frame_record.bitRate:.12f}\n")


def getSSIM(capturedFrame, decodedFrame):
    """Calculate SSIM (Structural Similarity Index) between two images."""
    C1 = 6.5025
    C2 = 58.5225
    I1 = capturedFrame.astype(np.float64)
    I2 = decodedFrame.astype(np.float64)
    I1_2 = I1 * I1
    I2_2 = I2 * I2
    I1_I2 = I1 * I2
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5) - mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5) - mu1_mu2
    t1 = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    t2 = (mu1_2 + mu2_2 + C1) * (sigma1_2 + sigma2_2 + C2)
    ssim_map = t1 / t2
    mssim = np.mean(ssim_map)
    ssim = mssim.item()
    return ssim


def getPSNR(block1, block2):
    """Compute Peak Signal-to-Noise Ratio between two blocks/images."""
    mse = getMSE(block1,block2)
    if mse >= 1e-10:
        return 10.0 * np.log10(255 ** 2 / mse)
    else:
        return 600.0  # Arbitrarily large value when MSE is very small


def delete_folder_content(folder_path):        
    """Remove all contents from a directory, or create the directory if it doesn't exist."""
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # The folder exists, so delete its contents
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                raise RuntimeError(f"Error deleting {file_path}: {str(e)}")
                #print(f"Error deleting {file_path}: {str(e)}")
    else:
        # The folder doesn't exist, so create an empty folder
        try:
            os.makedirs(folder_path)
            #print(f"Created folder: {folder_path}")
        except Exception as e:
            raise RuntimeError(f"Error creating folder: {str(e)}")
            #print(f"Error creating folder: {str(e)}")


def getMS(block):
    """Calculate the Mean Square value of the input block."""
    largeBlock = block.astype(np.uint16)
    largeBlock = np.multiply(largeBlock, largeBlock)
    error2 = np.sum(largeBlock)
    return error2 / (block.shape[0] * block.shape[1])


def msToPsnr(mse):
    """Convert Mean Squared Error to PSNR."""
    if mse >= 1e-10:
        return 10.0 * np.log10((255**2) / mse)
    else:
        return 600.0


def getMSE(block1, block2):
    """Calculate Mean Squared Error between two blocks/images (uint16 version)."""
    diffBlock = cv2.absdiff(block1, block2)
    diffB = cv2.convertScaleAbs(diffBlock, cv2.CV_16U)
    diffB = np.square(diffB)
    error2 = np.sum(diffB)
    return error2 / (block1.shape[0] * block1.shape[1])


def getMSE_2(block1,block2):
    """Alternate Mean Squared Error calculation, returns mean of squared difference."""
    diffImage = cv2.absdiff(block1, block2)
    diffImage = np.float32(diffImage)
    squared_diff = np.square(diffImage)
    mse = np.mean(squared_diff)
    return mse


def get_row_col(block_nb, frame_width):
    """Convert block number to (row, column) indices given frame width."""
    assert frame_width > 0 and frame_width % 8 == 0
    blocks_per_row = frame_width // 8
    row = 8 * (block_nb // blocks_per_row)
    col = 8 * (block_nb % blocks_per_row)
    return row, col


def count_captured_frames(frame_dir):
    """Return the number of captured frame files present in a directory."""
    if not os.path.isdir(frame_dir):
        return 0
    try:
        return len([name for name in os.listdir(frame_dir) if name.startswith("frame") and name.endswith(".png")])
    except OSError:
        return 0


def parse_rcv_file(line):
    """Parse a trace file line for normal packet/frame (non-ROI) information."""
    packet_blocks = []
    tokens = line.strip().split('\t')
    frame_nb = int(tokens[3])
    frame_type = tokens[4]
    layer_nb = int(tokens[5])
    blocks = tokens[6].split()
    signal_loss = float(tokens[7])
    snr = float(tokens[8])
    ber = float(tokens[9])
    for block in blocks:
        if '-' in block:
            start_block, end_block = map(int, block.split('-'))
            packet_blocks.extend(range(start_block, end_block + 1))
        else:
            packet_blocks.append(int(block))
    return frame_nb, frame_type, layer_nb, packet_blocks, signal_loss,snr,ber


def parse_rcv_file_noise(line):
    """Parse a noisy trace file line for S frame recovery blocks and metadata."""
    packet_blocks = []
    tokens = line.strip().split('\t')
    time = tokens[0]
    seq_nb = tokens[1]
    pktSize= tokens[2]
    frame_nb = int(tokens[3])
    frame_type = tokens[4]
    layer_nb = int(tokens[5])
    blocks = tokens[6].split()
    for block in blocks:
        if '-' in block:
            start_block, end_block = map(int, block.split('-'))
            packet_blocks.extend(range(start_block, end_block + 1))
        else:
            packet_blocks.append(int(block))
    return time, seq_nb, pktSize ,frame_nb, frame_type, layer_nb, packet_blocks


def parse_rcv_file_roi(line):
    """Parse a ROI trace file line, extracting block ranges with compression level and signal metrics."""
    packet_blocks = []
    tokens = line.strip().split('\t')
    frame_nb = int(tokens[3])
    frame_type = tokens[4]
    layer_nb = int(tokens[5])
    blocks = tokens[6]
    signal_loss = float(tokens[7])
    snr = float(tokens[8])
    ber = float(tokens[9])
    pattern = r'(\d+)->(\d+)([A-Za-z])'
    matches = re.findall(pattern, blocks)
    levels = []
    # Extract the start, end, and level for each match
    for match in matches:
        start, end, level = match
        levels.append([int(start),int(end),level])
    return frame_nb, frame_type, layer_nb, levels,signal_loss,snr,ber


def get_empty_layers(frame_nb):
    """Find all zero-sized layers for a given frame number (for error concealment)."""
    frame_trace_name = para.trace_file_path + "/st-frame"
    empty_blocks = []
    with open(frame_trace_name, 'r') as trace_file:
        for line in trace_file:
            if not line.strip() or line.startswith('#') or line.startswith(' '):
                continue
            tokens = line.strip().split('\t')
            if frame_nb == int(tokens[0]):
                blocks = list(map(int, tokens[7].split()))
                empty_blocks.extend([i for i, block in enumerate(blocks) if block == 0])
    return empty_blocks


def get_row_col_ber_matrix(block_nb, frame_width):
    """Map a block number to its row/column in the BER matrix."""
    assert frame_width > 0 
    row =(block_nb // frame_width)
    col =(block_nb % frame_width)
    return row, col