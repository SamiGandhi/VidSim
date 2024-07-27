import os
import cv2
import numpy as np
import errno
import math
import re
from parameters import Parameters as para

def make_dir(dir_name):
        os.makedirs(dir_name, exist_ok=True)

def delete_folder_content(dir_name):
    try:
        for filename in os.listdir(dir_name):
            filepath = os.path.join(dir_name, filename)
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.unlink(filepath)
    except Exception as e:
        print(f"Error while deleting folder content in {dir_name}: {e}")

def create_trace_files(trace_path):
    frame_trace_name = os.path.join(trace_path, "st-frame")
    with open(frame_trace_name, "w") as trace_file:
        trace_file.write("#Rank\tType\tSize(Bytes)\trefPSNR\trefSSIM\tbpp\tlayers Size (bits)\tcaptureEnergy(mJ)\tencodingEnergy(mJ)\tbit rate (kbps)\n")

    packet_trace_name = os.path.join(trace_path, "st-packet")
    with open(packet_trace_name, "w") as trace_file:
        trace_file.write("#time\tseqNb\tpktSize\tframeNb\tframeType\tlayerNb\tblocksList\tsignal_lost(db)\tSNR\tBER\n")

    rt_frame_trace_name = os.path.join(trace_path, "rt-frame")
    with open(rt_frame_trace_name, "w") as trace_file:
        trace_file.write("#Rank\tframeType\tPSNR\tSSIM\n")

def write_decoded_frame_record(frame_nb, frame_record):
    rt_frame_trace_name = os.path.join(para.trace_file_path, "rt-frame")
    with open(rt_frame_trace_name, "a") as trace_file:
        if not trace_file:
            print(f"Problem opening file {rt_frame_trace_name}")
            return
        trace_file.write(f"{frame_nb}\t{frame_record.frame_type}\t{frame_record.PSNR:.12f}\t{frame_record.SSIM:.12f}\n")

def write_packet_record_s_frame(packet_record, trace_path, signal_loss, snr, ber_value):
    packet_trace_name = trace_path + "/st-packet"
    with open(packet_trace_name, "a") as trace_file:
        trace_file.write(f"{packet_record.send_time}\t{packet_record.seq_nb}\t{int(np.ceil(packet_record.packet_size/8.0))}\t{packet_record.frame_nb}\t{packet_record.frame_type}\t{packet_record.layer_nb}\t")

        if packet_record.frame_type == "S":
            suite = False
            for ind in range(len(packet_record.blockSeqVector)):
                if ind == 0:
                    trace_file.write(str(packet_record.block_seq_vector[ind]))
                elif packet_record.block_seq_vector[ind] == packet_record.bloblock_seq_vectorckSeqVector[ind-1] + 1:
                    if ind == len(packet_record.block_seq_vector) - 1:
                        trace_file.write(f"-{packet_record.block_seq_vector[ind]}")
                    else:
                        suite = True
                elif suite:
                    trace_file.write(f"-{packet_record.block_seq_vector[ind-1]} {packet_record.block_seq_vector[ind]}")
                    suite = False
                else:
                    trace_file.write(f" {packet_record.block_seq_vector[ind]}")

            trace_file.write(f"\t{signal_loss}\t{snr}\t{ber_value}\n")
        else:
            for ind in range(len(packet_record.block_seq_vector)):
                trace_file.write(f"{packet_record.block_seq_vector[ind]} ")
            trace_file.write(f"\t{signal_loss}\t{snr}\t{ber_value}\n")



def write_packet_record_roi_m_frame(packet_record, trace_path, signal_loss,snr,ber_value):
    packet_trace_name = f"{trace_path}/st-packet"
    
    with open(packet_trace_name, "a") as trace_file:
        blocks = compress_vectors_to_string(packet_record.block_seq_vector, packet_record.blocks_compression_vector)
        trace_file.write(f"{packet_record.send_time}\t{packet_record.seq_nb}\t{math.ceil(packet_record.packet_size / 8.0)}\t{packet_record.frame_nb}\t{packet_record.frame_type}\t{packet_record.layer_nb}\t{blocks}\t{signal_loss}\t{snr}\t{ber_value}\n")

def write_packet_record_m_frame(packet_record, trace_path,signal_loss,snr,ber_value):
    packet_trace_name = f"{trace_path}/st-packet"
    
    with open(packet_trace_name, "a") as trace_file:
        first_block = packet_record.block_seq_vector[0]
        last_block = packet_record.block_seq_vector[len(packet_record.block_seq_vector) - 1]
        trace_file.write(f"{packet_record.send_time}\t{packet_record.seq_nb}\t{math.ceil(packet_record.packet_size / 8.0)}\t{packet_record.frame_nb}\t{packet_record.frame_type}\t{packet_record.layer_nb}\t{first_block} {last_block}\t{signal_loss}\t{snr}\t{ber_value}\n")





def write_frame_record(frame_record):
    frame_trace_name = para.trace_file_path + "/st-frame"
    with open(frame_trace_name, "a") as trace_file:
        trace_file.write(f"{frame_record.frameNb}\t"
                         f"{frame_record.frameType}\t"
                         f"{int(frame_record.frameSize/8.0 + 0.5)}\t"
                         f"{frame_record.PSNR:.12f}\t"
                         f"{frame_record.SSIM:.12f}\t"
                         f"{frame_record.bpp:.12f}\t")

        if frame_record.frameType == "M":
            trace_file.write(" ".join(map(str, frame_record.layersSizeVector)))
        else:
            trace_file.write("-")

        trace_file.write(f"\t{frame_record.captureEnergy:.12f}\t"
                         f"{frame_record.encodingEnergy:.12f}\t"
                         f"{frame_record.bitRate:.12f}\n")







def compress_vectors_to_string(block_numbers, compressions):
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

def get_ssim(captured_frame, decoded_frame):
    C1 = 6.5025
    C2 = 58.5225

    I1 = captured_frame.astype(np.float64)
    I2 = decoded_frame.astype(np.float64)

    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)

    sigma1_2 = cv2.GaussianBlur(I1 * I1, (11, 11), 1.5) - mu1 * mu1
    sigma2_2 = cv2.GaussianBlur(I2 * I2, (11, 11), 1.5) - mu2 * mu2
    sigma12 = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_2 + sigma2_2 + C2))
    return np.mean(ssim_map)

def get_psnr(block1, block2):
    mse = get_mse(block1,block2)
    if mse >= 1e-10:
        return 10.0 * np.log10(255 ** 2 / mse)
    else:
        return 600.0

def get_mse(block1, block2):
    diff = cv2.absdiff(block1, block2)
    return np.mean(np.square(diff))

def get_mse_(block1,block2):
    diffImage = cv2.absdiff(block1, block2)
    diffImage = np.float32(diffImage)
    squared_diff = np.square(diffImage)
    mse = np.mean(squared_diff)
    return mse

def get_ms(block):
    largeBlock = block.astype(np.uint16)
    largeBlock = np.multiply(largeBlock, largeBlock)
    error2 = np.sum(largeBlock)
    return error2 / (block.shape[0] * block.shape[1])

def get_row_col(block_nb, frame_width):
    assert frame_width > 0 and frame_width % 8 == 0
    blocks_per_row = frame_width // 8
    return 8 * (block_nb // blocks_per_row), 8 * (block_nb % blocks_per_row)

def parse_rcv_file(line):
    tokens = line.strip().split('\t')
    frame_nb = int(tokens[3])
    frame_type = tokens[4]
    layer_nb = int(tokens[5])
    blocks = list(map(int, re.findall(r'\d+', tokens[6])))
    signal_loss = float(tokens[7])
    snr = float(tokens[8])
    ber = float(tokens[9])
    return frame_nb, frame_type, layer_nb, blocks, signal_loss, snr, ber

def parse_rcv_file_noise(line):
    tokens = line.strip().split('\t')
    time, seq_nb, pkt_size = tokens[:3]
    frame_nb = int(tokens[3])
    frame_type = tokens[4]
    layer_nb = int(tokens[5])
    blocks = list(map(int, re.findall(r'\d+', tokens[6])))
    return time, seq_nb, pkt_size, frame_nb, frame_type, layer_nb, blocks

def parse_rcv_file_roi(line):
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
    levels = [[int(start), int(end), level] for start, end, level in matches]
    return frame_nb, frame_type, layer_nb, levels, signal_loss, snr, ber

def get_empty_layers(frame_nb):
    frame_trace_name = os.path.join(para.trace_file_path, "st-frame")
    empty_blocks = []

    with open(frame_trace_name, 'r') as trace_file:
        for line in trace_file:
            if not line.strip() or line.startswith('#') or line.startswith(' '):
                continue
            tokens = line.strip().split('\t')
            if frame_nb == int(tokens[0]):
                blocks = list(map(int, tokens[6].split()))
                empty_blocks.extend([i for i, block in enumerate(blocks) if block == 0])

    return empty_blocks

def get_row_col_ber_matrix(block_nb, frame_width):
    assert frame_width > 0
    return block_nb // frame_width, block_nb % frame_width
