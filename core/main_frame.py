# main_frame.py
# Handles main frame encoding and ROI main frame encoding for video compression,
# including DCT, quantization, block layered entropy encoding, packetization, and reference reconstructions.
# Used to initialize the compression process for base/intra-frames, using standard or ROI logic.

import numpy as np
import cv2
from core.parameters import Parameters as para
import core.dct as dct
import core.trace as trace
import core.entropy as entropy
import core.util as util
import core.network_losses_model as losses
import metrics.quality_metrics as metrics

EXIT_FAILURE = 0


def get_roi_reference_frame(quantifFrame, sad_map, r_map, g_map):
    """
    Generate a reference frame for a ROI main frame: dequantizes and reverses DCT per block
    based on the mask type (high, low, or zero quality).
    Blocks with no interest are left zero.
    """
    refFrame = np.zeros(quantifFrame.shape, dtype=np.uint8)
    for i in range(0, quantifFrame.shape[0], 8):
        for j in range(0, quantifFrame.shape[1], 8):
            quantifBlock = quantifFrame[i:i + 8, j:j + 8]
            sad_mask_block = sad_map[i:i + 8, j:j + 8]
            r_mask_block = r_map[i:i + 8, j:j + 8]
            g_mask_block = g_map[i:i + 8, j:j + 8]
            if np.any(sad_mask_block > 0): # high quality
                dequantifBlock = quantifBlock * getQuantisationMatrix(para.high_quality_factor)
            elif np.any(r_mask_block > 0): # low quality
                dequantifBlock = quantifBlock * getQuantisationMatrix(para.low_quality_factor)
            elif np.any(g_mask_block > 0): # dropped region (zero)
                dequantifBlock = quantifBlock * getQuantisationMatrix(para.zero)
            else:
                refFloatBlock = np.zeros((8, 8))
                refFrame[i:i + 8, j:j + 8] = refFloatBlock
                continue
            refFloatBlock = dct.iDCT(dequantifBlock, para.DCT, para.zone_size)
            refFloatBlock += 128
            refFloatBlock = np.clip(refFloatBlock, 0, 255).astype(np.uint8)
            refFrame[i:i + 8, j:j + 8] = refFloatBlock
    return refFrame


def get_roi_quantif_frame(frameToEncode, sad_map, r_map, g_map, frame_number):
    """
    Quantize each block of the ROI frame according to the masks.
    Assigns block compression class and updates compression vector for later trace logging and packetization.
    """
    roi_blocks_n = 0
    blocks_compression_vector = {}
    quantifFrame = np.zeros(frameToEncode.shape, dtype=np.int16)
    # Loop blocks of 8x8
    for i in range(0, frameToEncode.shape[0], 8):
        for j in range(0, frameToEncode.shape[1], 8):
            block_number = i * quantifFrame.shape[1] // 64 + j // 8
            block = frameToEncode[i:i + 8, j:j + 8]
            sad_mask_block = sad_map[i:i + 8, j:j + 8]
            g_mask_block = g_map[i:i + 8, j:j + 8]
            r_mask_block = r_map[i:i + 8, j:j + 8]
            floatBlock = block.astype(np.float32)
            floatBlock -= 128
            dctBlock = dct.DCT(floatBlock, para.DCT, para.zone_size)
            dctFloatBlock = dctBlock.astype(np.float32)
            if np.any(sad_mask_block > 0):
                quantizationMatrix = getQuantisationMatrix(para.high_quality_factor)
                blocks_compression_vector[block_number] = 'H'
                roi_blocks_n += 1
            elif np.any(r_mask_block > 0):
                quantizationMatrix = getQuantisationMatrix(para.low_quality_factor)
                blocks_compression_vector[block_number] = 'L'
                roi_blocks_n += 1
            else:
                quantizationMatrix = getQuantisationMatrix(0)
            quantifBlock = np.divide(dctFloatBlock, quantizationMatrix).astype(np.int16)
            quantifFrame[i:i + 8, j:j + 8] = quantifBlock
    return quantifFrame, blocks_compression_vector








def encode_roi_main_frame(frame,frame_record,frame_number,seq_nb,sad_map,r_map,g_map):
    quantif_frame, blocks_compression_vector = get_roi_quantif_frame(frame,sad_map,r_map,g_map,frame_number)
    reference_frame = get_roi_reference_frame(quantif_frame,sad_map,r_map,g_map)
    frame_blocks_nb = (frame.shape[0] * frame.shape[1]) // 64
    interleaved_frame = None 
    quantif_frame_i = None  
    reference_frame_i = None
    if para.interleaving == "torus":
        interleaved_frame = torus_interleaving(frame)
        quantif_frame_i = getQuantifFrame(interleaved_frame)
        reference_frame_i = getReferenceFrame(quantif_frame_i)
    packet_record_array = [trace.packet_record()]*para.level_numbers
    frame_compressed_size = 0
    entropy_cycles = 0
    compressed_blocks_number = len(blocks_compression_vector)
    if(compressed_blocks_number == 0):
        return
    iterated_blocks = 0
    for i in range(0,frame.shape[0],para.zone_size):
        for j in range(0,frame.shape[1],para.zone_size):
            block_number = i * quantif_frame.shape[1]//64 + j//8
            if (frame_number == 20):
                abc =1
            try:
                blocks_compression_vector[block_number]
                iterated_blocks += 1
            except KeyError:
                continue
            if para.DCT.startswith('s'):
                if para.interleaving == "torus":
                    quantif_block = quantif_frame_i[i:i + para.zone_size, j:j + para.zone_size]
                else:
                    quantif_block = quantif_frame[i:i + para.zone_size, j:j + para.zone_size]
            else:
                if para.interleaving == "torus":
                    quantif_block = quantif_frame_i[i:i + 8, j:j + 8]
                else:
                    quantif_block = quantif_frame[i:i + 8, j:j + 8]
            zigzag_block = zigzagScan(quantif_block)
            layers_vector = make_layers(frame_record,zigzag_block)
            for ind in range(0,len(layers_vector)):
                entropy_cycles += layers_vector[ind].entropyCycles
            frame_compressed_size += get_block_compressed_size(layers_vector)
            packetize_roi_main_frame_block(frame_blocks_nb,frame_number,block_number,layers_vector,packet_record_array,seq_nb,blocks_compression_vector,iterated_blocks,compressed_blocks_number)
    frame_record.frameSize = frame_compressed_size
    frame_record.PSNR = metrics.calculate_psnr(reference_frame,frame)
    frame_record.SSIM = metrics.calculate_ssim(reference_frame,frame)
    frame_record.BRISQUE = metrics.calculate_brisque(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    frame_record.bpp = frame_compressed_size * 1.0/(frame.shape[0]*frame.shape[1])
    frame_record.bitRate = frame_compressed_size* para.fps/100
    frame_record.captureEnergy = para.CAPTURE_E_PER_BLOCK*frame_blocks_nb*1000
    frame_record.encodingEnergy = getEncodingEnergyM(compressed_blocks_number,quantif_frame)+entropy_cycles*para.POWER/para.PROC_CLOCK/1e6
    util.write_frame_record(frame_record)
    image_name = f"{para.captured_frame_dir}/frame{frame_record.frameNb}.png"
    orig_frame = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image_name = f"{para.reference_frames_dir}/frame{frame_record.frameNb}.png"
    cv2.imwrite(image_name, reference_frame)






















def encode_main_frame(frame,frame_record,frame_number,seq_nb):
    quantif_frame = getQuantifFrame(frame)
    reference_frame = getReferenceFrame(quantif_frame)
    frame_blocks_nb = (frame.shape[0] * frame.shape[1]) // 64
    interleaved_frame = None 
    quantif_frame_i = None  
    reference_frame_i = None
    if para.interleaving == "torus":
        interleaved_frame = torus_interleaving(frame)
        quantif_frame_i = getQuantifFrame(interleaved_frame)
        reference_frame_i = getReferenceFrame(quantif_frame_i)
    packet_record_array = [trace.packet_record()]*para.level_numbers
    frame_compressed_size = 0
    entropy_cycles = 0
    for i in range(0,frame.shape[0],para.zone_size):
        for j in range(0,frame.shape[1],para.zone_size):



            if para.DCT.startswith('s'):
                if para.interleaving == "torus":
                    quantif_block = quantif_frame_i[i:i + para.zone_size, j:j + para.zone_size]
                else:
                    quantif_block = quantif_frame[i:i + para.zone_size, j:j + para.zone_size]
            else:
                if para.interleaving == "torus":
                    quantif_block = quantif_frame_i[i:i + 8, j:j + 8]
                else:
                    quantif_block = quantif_frame[i:i + 8, j:j + 8]
            zigzag_block = zigzagScan(quantif_block)
            layers_vector = make_layers(frame_record,zigzag_block)
            for ind in range(0,len(layers_vector)):
                entropy_cycles += layers_vector[ind].entropyCycles
            frame_compressed_size += get_block_compressed_size(layers_vector)
            block_number = i * quantif_frame.shape[1]//64 + j//8
            packetize_main_frame_block(frame_blocks_nb,frame_number,block_number,layers_vector,packet_record_array,seq_nb)
    frame_record.frameSize = frame_compressed_size
    frame_record.PSNR = metrics.calculate_psnr(reference_frame,frame)
    frame_record.SSIM = metrics.calculate_ssim(reference_frame,frame)
    frame_record.BRISQUE = metrics.calculate_brisque(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    frame_record.bpp = frame_compressed_size * 1.0/(frame.shape[0]*frame.shape[1])
    frame_record.bitRate = frame_compressed_size* para.fps/100
    frame_record.captureEnergy = para.CAPTURE_E_PER_BLOCK*frame_blocks_nb*1000
    frame_record.encodingEnergy = getEncodingEnergyM(frame_blocks_nb,quantif_frame)+entropy_cycles*para.POWER/para.PROC_CLOCK/1e6
    util.write_frame_record(frame_record)
    image_name = f"{para.captured_frame_dir}/frame{frame_record.frameNb}.png"
    orig_frame = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image_name = f"{para.reference_frames_dir}/frame{frame_record.frameNb}.png"
    cv2.imwrite(image_name, reference_frame)




















def get_block_compressed_size(block_layers):
    size = 0
    for layer_info in block_layers:
        size += layer_info.layerSize
    return size



def torus_interleaving(frame_to_encode):
    a = 1
    b = 1
    c = 1
    d = 2
    interleaved_frame = np.zeros_like(frame_to_encode, dtype=np.uint8)
    blocks_in_row, blocks_in_col = frame_to_encode.shape[0] // 8, frame_to_encode.shape[1] // 8

    for bi in range(blocks_in_row):
        for bj in range(blocks_in_col):
            new_bi = (a * bi + b * bj) % blocks_in_row
            new_bj = (c * bi + d * bj) % blocks_in_col

            temp_block = frame_to_encode[bi * 8:(bi + 1) * 8, bj * 8:(bj + 1) * 8]
            interleaved_frame[new_bi * 8:(new_bi + 1) * 8, new_bj * 8:(new_bj + 1) * 8] = temp_block

    return interleaved_frame


def packetize_main_frame_block(block_per_frame, frame_nb, block_nb, layers_vector, packet_record_array, seq_nb):
    try:
        snr,total_loss_db = losses.combined_loss_model(para.DISTANCE, para.FREQUENCY, para.ENVIRONMENT, para.HUMIDITY_LEVEL, para.VEGETATION_DENSITY_LEVEL)
        ber = losses.calculate_ber(snr)
    except TypeError:
        ber = 0
        snr = 0
        total_loss_db = 0


    for k in range(len(layers_vector)):
        if layers_vector[k].layerSize > para.packed_load_size * 8:
            #print("M Very small payload size !!", layers_vector[k].layerSize)
            #print(f"M layer size {layers_vector[k].layerSize} parameters { para.packed_load_size * 8}")
            exit(EXIT_FAILURE)


        if block_nb == 0:  # first block
            packet_record_array[k].frame_type = 'M'
            packet_record_array[k].packet_size = 0
            packet_record_array[k].block_seq_vector = []
            packet_record_array[k].seq_nb = k
            packet_record_array[k].frame_nb = frame_nb
            packet_record_array[k].send_time = (frame_nb - 1) * 1.0 / para.fps

        if packet_record_array[k].packet_size + layers_vector[k].layerSize < para.packed_load_size * 8:
            packet_record_array[k].block_seq_vector.append(block_nb)
            packet_record_array[k].packet_size += layers_vector[k].layerSize
        else:  # write record and create new pkt
            seq_nb[0] += 1
            packet_record_array[k].seq_nb = seq_nb[0]
            util.write_packet_record_m_frame(packet_record_array[k], para.trace_file_path,total_loss_db,snr,ber)
            # NEW pkt:
            packet_record_array[k].block_seq_vector = [block_nb]
            packet_record_array[k].packet_size = layers_vector[k].layerSize

        if block_nb == block_per_frame - 1 and packet_record_array[k].packet_size > 0:  # last block
            seq_nb[0] += 1
            packet_record_array[k].seq_nb = seq_nb[0]
            packet_record_array[k].block_seq_vector.append(block_nb)
            util.write_packet_record_m_frame(packet_record_array[k], para.trace_file_path,total_loss_db,snr,ber)


def packetize_roi_main_frame_block(block_per_frame, frame_nb, block_nb, layers_vector, packet_record_array, seq_nb,blocks_compression_vector,iterated_blocks,compressed_blocks_number):
    #calculate the signal error and the ber baed on the param
    try: 
        snr,total_loss_db = losses.combined_loss_model(para.DISTANCE, para.FREQUENCY, para.ENVIRONMENT, para.HUMIDITY_LEVEL, para.VEGETATION_DENSITY_LEVEL)
        ber = losses.calculate_ber(snr)
    except TypeError:
        ber = 0
        snr = 0
        total_loss_db = 0
    for k in range(len(layers_vector)):
        if layers_vector[k].layerSize > para.packed_load_size * 8:
            #print("M Very small payload size !!", layers_vector[k].layerSize)
            exit(EXIT_FAILURE)

        if iterated_blocks == 1:  # first block
            packet_record_array[k].frame_type = 'M'
            packet_record_array[k].packet_size = 0
            packet_record_array[k].block_seq_vector = []
            packet_record_array[k].seq_nb = k
            packet_record_array[k].frame_nb = frame_nb
            packet_record_array[k].send_time = (frame_nb - 1) * 1.0 / para.fps

        if packet_record_array[k].packet_size + layers_vector[k].layerSize < para.packed_load_size * 8:
            packet_record_array[k].block_seq_vector.append(block_nb)
            packet_record_array[k].blocks_compression_vector.append(blocks_compression_vector[block_nb])
            packet_record_array[k].packet_size += layers_vector[k].layerSize
        else:  # write record and create new pkt
            seq_nb[0] += 1
            packet_record_array[k].seq_nb = seq_nb[0]
            util.write_packet_record_roi_m_frame(packet_record_array[k], para.trace_file_path,total_loss_db,snr,ber)
            # NEW pkt:
            packet_record_array[k].block_seq_vector = [block_nb]
            packet_record_array[k].blocks_compression_vector = [blocks_compression_vector[block_nb]]

            packet_record_array[k].packet_size = layers_vector[k].layerSize

        if iterated_blocks == compressed_blocks_number:  # last block
            seq_nb[0] += 1
            packet_record_array[k].seq_nb = seq_nb[0]
            packet_record_array[k].block_seq_vector.append(block_nb)
            packet_record_array[k].blocks_compression_vector.append(blocks_compression_vector[block_nb])
            util.write_packet_record_roi_m_frame(packet_record_array[k], para.trace_file_path,total_loss_db,snr,ber)



def make_layers(frame_record, zigzag_block):
    layer_start = [0, 3, 6, 10, 15, 21, 28, 36, 43, 49, 54, 58, 61]
    level_start = 0
    layers_vector = []
    end_block = 0

    if para.DCT == "CLA" or para.DCT[0] == 's':
        end_block = zigzag_block.shape[0]

    if para.DCT == "sLLM" or para.DCT == "sBIN":
        max_levels_nb = 2 * (para.zone_size - 1)

        if para.level_numbers > max_levels_nb:
            para.level_numbers = max_levels_nb
        inc = para.zone_size - 1
        for i in range(para.zone_size, max_levels_nb):
            layer_start[i] = layer_start[i - 1] + inc
            inc -= 1

    if para.DCT == "tLLM" or para.DCT == "tBIN":
        if para.level_numbers > para.zone_size - 1:
            para.levelsNb = para.zoneSize - 1
        end_block = layer_start[para.zone_size - 1]

    for i in range(para.level_numbers):
        layer_info = LayerInfoStruct()

        if i < para.level_numbers - 1:
            layer_info.layerRawData = zigzag_block[layer_start[i]:layer_start[i + 1]]
        else:
            layer_info.layerRawData = zigzag_block[ level_start:end_block]

        layer_info.layerNb = i
        level_start += layer_info.layerRawData.shape[0]

        layer_info.layerData, layer_info.entropyCycles = entropy.entropy_coder(layer_info.layerRawData,
                                                                      layer_info.layerNb,
                                                                      para.entropy_coding)

        if layer_info.layerData == "00":
            layer_info.layerData = ""
        layer_info.layerSize = len(layer_info.layerData)

        frame_record.layersSizeVector[i] += layer_info.layerSize

        layers_vector.append(layer_info)

    return layers_vector


def getReferenceFrame(quantifFrame):
    refFrame = np.zeros(quantifFrame.shape, dtype=np.uint8)

    for i in range(0, quantifFrame.shape[0], 8):
        for j in range(0, quantifFrame.shape[1], 8):
            quantifBlock = quantifFrame[i:i + 8, j:j + 8]
            dequantifBlock = quantifBlock * getQuantisationMatrix(para.quality_factor)
            refFloatBlock = dct.iDCT(dequantifBlock, para.DCT, para.zone_size)
            refFloatBlock += 128
            refFloatBlock = np.clip(refFloatBlock, 0, 255).astype(np.uint8)
            refFrame[i:i + 8, j:j + 8] = refFloatBlock

    return refFrame


def getQuantisationMatrix(QC):
    if QC == 0:  # Check if the quantization factor is for the drop zone
        # Create a quantization matrix that will set all values to zero
        return np.full((8, 8), np.inf)
    quantizationMatrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                   [12, 12, 14, 19, 26, 58, 60, 55],
                                   [14, 13, 16, 24, 40, 57, 69, 56],
                                   [14, 17, 22, 29, 51, 87, 80, 62],
                                   [18, 22, 37, 56, 68, 109, 103, 77],
                                   [24, 35, 55, 64, 81, 104, 113, 92],
                                   [49, 64, 78, 87, 103, 121, 120, 101],
                                   [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32)

    if QC < 50:
        S = 5000 / QC
    else:
        S = 200 - 2 * QC

    temp = S * quantizationMatrix + 50
    quantifM = np.where(temp == 0, 1, temp)
    quantifM = quantifM / 100.0

    return quantifM

import numpy as np
import cv2

def getQuantifFrame(frameToEncode):
    quantifFrame = np.zeros(frameToEncode.shape, dtype=np.int16)

    for i in range(0, frameToEncode.shape[0], 8):
        for j in range(0, frameToEncode.shape[1], 8):
            block = frameToEncode[i:i + 8, j:j + 8]
            floatBlock = block.astype(np.float32)
            floatBlock -= 128

            dctBlock = dct.DCT(floatBlock, para.DCT, para.zone_size)
            dctFloatBlock = dctBlock.astype(np.float32)
            
            quantizationMatrix = getQuantisationMatrix(para.quality_factor)
            quantifBlock = np.divide(dctFloatBlock, quantizationMatrix).astype(np.int16)
            quantifFrame[i:i + 8, j:j + 8] = quantifBlock

    return quantifFrame


def zigzagScan(block):
    blockZigZag = np.zeros((block.shape[0] * block.shape[1]), dtype=np.int16)
    i, j = 0, 0
    jmax, imax = block.shape[0] - 1, block.shape[1] - 1
    count = 0

    while (j <= jmax) and (i <= imax):
        if ((i + j) % 2) == 0:
            if j == 0:
                blockZigZag[count] = block[j, i]
                count += 1
                if i == imax:
                    j += 1
                else:
                    i += 1
            elif (i == imax) and (j < jmax):
                blockZigZag[count] = block[j, i]
                count += 1
                j += 1
            elif (j > 0) and (i < imax):
                blockZigZag[count] = block[j, i]
                count += 1
                j -= 1
                i += 1
        else:
            if (j == jmax) and (i <= imax):
                blockZigZag[count] = block[j, i]
                count += 1
                i += 1
            elif i == 0:
                blockZigZag[count] = block[j, i]
                count += 1
                if j == jmax:
                    i += 1
                else:
                    j += 1
            elif (j < jmax) and (i > 0):
                blockZigZag[count] = block[j, i]
                count += 1
                j += 1
                i -= 1

        if (j == jmax) and (i == imax):
            blockZigZag[count] = block[j, i]
            break

    return blockZigZag

def count_nonzero_coefficients(quantifFrame):
    nonzero_count = 0
    for i in range(0, quantifFrame.shape[0], 8):
        for j in range(0, quantifFrame.shape[1], 8):
            block = quantifFrame[i:i + 8, j:j + 8]
            nonzero_count += np.count_nonzero(block)
    return nonzero_count



def getEncodingEnergyM(frameBlocksNb,quantifFrame):
    # 1D DCT number of ADD, MUL, or SHT ...
    llm_add = [0, 6, 20, 23, 24, 35, 26, 28, 29]
    acc_llm_add = [0, 6, 26, 49, 73, 98, 124, 152, 181]

    llm_mul = [0, 0, 6, 8, 9, 9, 10, 11, 11]
    acc_llm_mul = [0, 0, 6, 14, 23, 32, 42, 53, 64]

    bin_add = [0, 7, 13, 19, 27, 28, 28, 28, 30]
    acc_bin_add = [0, 7, 20, 39, 66, 94, 122, 150, 180]

    bin_shift = [0, 0, 2, 6, 11, 12, 12, 12, 13]
    acc_bin_shift = [0, 0, 2, 8, 19, 31, 43, 55, 68]

    dctCycles, quantifCycles = 0, 0
    addLLM, mulLLM, addBIN, shiftBIN = 0, 0, 0, 0
    energy = 0.0
    nonzero_count = count_nonzero_coefficients(quantifFrame)

    if para.DCT == "CLA":
        dctCycles = para.CLA_ADD_NB * para.CYCLES_PER_FADD + para.CLA_MUL_NB * para.CYCLES_PER_FMUL
        #quantifCycles = 64 * para.CYCLES_PER_FMUL
        quantifCycles = nonzero_count * para.CYCLES_PER_FMUL


    if para.DCT[0] == 't':
        #quantifCycles = para.zone_size * (para.zone_size + 1) / 2
        quantifCycles = nonzero_count  # Updated to use non-zero count
        addLLM = llm_add[para.zone_size] * para.zone_size + acc_llm_add[para.zone_size]
        mulLLM = llm_mul[para.zone_size] * para.zone_size + acc_llm_mul[para.zone_size]
        addBIN = bin_add[para.zone_size] * para.zone_size + acc_bin_add[para.zone_size]
        shiftBIN = bin_shift[para.zone_size] * para.zone_size + acc_bin_shift[para.zone_size]

    if para.DCT[0] == 's':
        #quantifCycles = para.zone_size * para.zone_size
        quantifCycles = nonzero_count  # Updated to use non-zero count
        addLLM = llm_add[para.zone_size] * (para.zone_size + 8)
        mulLLM = llm_mul[para.zone_size] * (para.zone_size + 8)
        addBIN = bin_add[para.zone_size] * (para.zone_size + 8)
        shiftBIN = bin_shift[para.zone_size] * (para.zone_size + 8)

    if para.DCT[1:] == "LLM":  # float!
        quantifCycles *= para.CYCLES_PER_FMUL
        dctCycles = para.CYCLES_PER_FADD * addLLM + para.CYCLES_PER_FMUL * mulLLM

    if para.DCT[1:] == "BIN":  # int!
        quantifCycles *= para.CYCLES_PER_MUL
        dctCycles = para.CYCLES_PER_ADD * addBIN + para.CYCLES_PER_SHIFT * shiftBIN

    energy = (quantifCycles + dctCycles) * frameBlocksNb / 1000 * para.POWER /para.PROC_CLOCK
    return energy / 1000.0  # mJ




class LayerInfoStruct:
    def __init__(self):
        self.layerNb = 0            # dataPriority 0..12 for M frames
        self.layerSize = 0          # size of compressed data sizeof(layerData)
        self.layerData = ""         # compressed binary data
        self.layerRawData = None    # raw data as a NumPy array
        self.entropyCycles = 0      # Nb of cycles required by the entropy coder



