# second_frame.py
# Implements encoding and processing for S-frames (secondary or predicted frames) in the video simulation pipeline.
# Handles MS-based block prioritization, DCT, quantization, entropy encoding, and network loss simulation for S-frames.

import numpy as np
import cv2
import core.entropy as entropy
from parameters import Parameters as para
import core.util as util
import core.trace as trace
import network_losses_model as losses
import metrics.quality_metrics as metrics
from main_frame import LayerInfoStruct



EXIT_FAILURE = 0







def encodeSecondFrame(currentFrame, mainFrameNb, frameRecord, seqNb):
    """
    Encodes an S-frame by predicting difference from the main frame, processing blocks by mean-square (MS) levels.
    Applies DCT, quantization, entropy coding, and packetizes results for simulated transmission.
    """
    refMainFrame = cv2.imread(para.reference_frames_dir + "/frame" + str(mainFrameNb) + ".png", cv2.IMREAD_GRAYSCALE)
    mainFrame = cv2.imread(para.captured_frame_dir + "/frame" + str(mainFrameNb) + ".png", cv2.IMREAD_GRAYSCALE)

    currentS = currentFrame.frame.astype('int16')
    mainS = mainFrame.astype('int16')
    refMainS = refMainFrame.astype('int16')
    energy = [0]

    redFrame,compressedSize,energy = reduce_and_packetize_second_frame(currentS - mainS, currentFrame.frame_number, mainFrameNb, seqNb)    
    refCurrentS = refMainS + redFrame
    diffReferenceFrame = refCurrentS.astype('uint8')
    imageName = para.reference_frames_dir + "/frame" + str(currentFrame.frame_number) + ".png"
    cv2.imwrite(imageName, diffReferenceFrame)

    frameRecord.frameSize = compressedSize
    frameRecord.bpp = frameRecord.frameSize * 1.0 / (currentFrame.frame.shape[1] * currentFrame.frame.shape[0])
    frameRecord.bitRate = frameRecord.frameSize * para.fps / 1000
    frameRecord.PSNR = metrics.calculate_psnr(currentFrame.frame.copy(), diffReferenceFrame.copy())
    frameRecord.SSIM = metrics.calculate_ssim(currentFrame.frame.copy(), diffReferenceFrame.copy())
    frameRecord.BRISQUE = metrics.calculate_brisque(cv2.cvtColor(currentFrame.frame.copy(), cv2.COLOR_GRAY2BGR)) #3 channel image to use Brisque library
    frameBlocksNb = (currentFrame.frame.shape[1] * currentFrame.frame.shape[0]) // 64
    frameRecord.captureEnergy = para.CAPTURE_E_PER_BLOCK * frameBlocksNb * 1000  # mJ
    frameRecord.encodingEnergy = energy

    util.write_frame_record(frameRecord)







def reduce_and_packetize_second_frame(frame, frameNb, mainFrameNb, seqNb):
    """
    Computes the reduced difference frame for S-frame. Prioritizes and collects blocks by MS values,
    applies threshold and entropy coding, and organizes packets for simulation of network transfer.
    Returns the reduced frame, total compressed size, and estimated energy.
    """
    try:
        snr, total_loss_db = losses.combined_loss_model(para.DISTANCE, para.FREQUENCY, para.ENVIRONEMENT, para.HUMIDITY_LEVEL, para.VEGETATION_DENSITY_LEVEL)
        ber = losses.calculate_ber(snr)
    except:
        snr = 0
        ber = 0
        total_loss_db = 0

    compressedSize = 0
    msv = [650, 205, 51, 13]
    retFrame = np.zeros(frame.shape, dtype=np.int16)
    packetRecordArray = [trace.packet_record() for _ in range(para.MAX_S_LAYERS)]

    for i in range(para.MAX_S_LAYERS):
        packetRecordArray[i].send_time = (frameNb - 1) / para.fps
        packetRecordArray[i].packet_size = 0
        packetRecordArray[i].frame_nb = frameNb
        packetRecordArray[i].frame_type = 'S' + str(mainFrameNb)
        packetRecordArray[i].layer_nb = i
        packetRecordArray[i].block_seq_vector = []

    cycleNb = frame.shape[0] * frame.shape[1] * (para.CYCLES_PER_ADD + para.CYCLES_PER_MUL)
    entropyCycles = 0

    for i in range(0, frame.shape[0], 8):
        for j in range(0, frame.shape[1], 8):
            blockNb = i * frame.shape[1] // 64 + j // 8
            block = frame[i:i + 8, j:j + 8]
            ms = util.get_ms(block.copy())

            blockPriority = 4
            if ms > 0:
                if ms >= msv[0]: blockPriority = 0
                elif ms >= msv[1]: blockPriority = 1
                elif ms >= msv[2]: blockPriority = 2
                elif ms >= msv[3]: blockPriority = 3
                else: blockPriority = 4

                cycleNb += 192  # 3 * 64 cycles in red2 (thresholding)
                _, thresholded_block = cv2.threshold(block, para.threshold, 255, cv2.THRESH_BINARY_INV)

                if cv2.countNonZero(thresholded_block) == 0 or blockPriority > para.max_level_S:
                    continue

                retFrame[i:i + 8, j:j + 8] = thresholded_block
                linearBlock = thresholded_block.flatten()
                prevBlockNb = 0
                if len(packetRecordArray[blockPriority].block_seq_vector) != 0:
                    prevBlockNb = len(packetRecordArray[blockPriority].block_seq_vector) - 1

                # Use the unified entropy coder and layer creation
                layers_vector = make_layers(linearBlock)
                for layer in layers_vector:
                    entropyCycles += layer.entropyCycles
                    compressedSize += layer.layerSize
                
    energy = cycleNb * para.POWER / para.PROC_CLOCK / 1000

    for k in range(para.MAX_S_LAYERS):
        if packetRecordArray[k].packet_size > 0:
            packetRecordArray[k].seq_nb = seqNb[0] + 1
            util.write_packet_record_s_frame(packetRecordArray[k], para.trace_file_path, total_loss_db, snr, ber)
            compressedSize += packetRecordArray[k].packet_size

    compressedSize = compressedSize 
    return retFrame, compressedSize, energy





def packetize_s_frame_block(blocks_per_frame, frame_nb, block_nb, layers_vector, packet_record_array, seq_nb):
    try:
        snr, total_loss_db = losses.combined_loss_model(para.DISTANCE, para.FREQUENCY, para.ENVIRONMENT, para.HUMIDITY_LEVEL, para.VEGETATION_DENSITY_LEVEL)
        ber = losses.calculate_ber(snr)
    except TypeError:
        ber = 0
        snr = 0
        total_loss_db = 0

    for k in range(len(layers_vector)):
        if layers_vector[k].layerSize > para.packed_load_size * 8:
            #print("S Very small payload size !!", layers_vector[k].layerSize)
            exit(EXIT_FAILURE)

        if block_nb == 0:  # first block
            packet_record_array[k].frame_type = 'S'
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
            util.write_packet_record_s_frame(packet_record_array[k], para.trace_file_path, total_loss_db, snr, ber)
            # NEW pkt:
            packet_record_array[k].block_seq_vector = [block_nb]
            packet_record_array[k].packet_size = layers_vector[k].layerSize

        if block_nb == blocks_per_frame - 1 and packet_record_array[k].packet_size > 0:  # last block
            seq_nb[0] += 1
            packet_record_array[k].seq_nb = seq_nb[0]
            packet_record_array[k].block_seq_vector.append(block_nb)
            util.write_packet_record_s_frame(packet_record_array[k], para.trace_file_path, total_loss_db, snr, ber)










def make_layers(zigzag_block):
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

        layers_vector.append(layer_info)

    return layers_vector







# do not use
def reduce_and_packetize_second_frame_old_with_errors(frame, frameNb, mainFrameNb, seqNb):
    try:
        snr, total_loss_db = losses.combined_loss_model(para.DISTANCE, para.FREQUENCY, para.ENVIRONEMENT, para.HUMIDITY_LEVEL, para.VEGETATION_DENSITY_LEVEL)    
        ber = losses.calculate_ber(snr)
    except:
        snr = 0
        ber = 0
        total_loss_db = 0

    compressedSize = 0
    msv = [650, 205, 51, 13]
    retFrame = np.zeros(frame.shape, dtype=np.int16)
    packetRecordArray = [trace.packet_record() for _ in range(para.MAX_S_LAYERS)]
    
    for i in range(para.MAX_S_LAYERS):
        packetRecordArray[i].send_time = (frameNb - 1) / para.fps
        packetRecordArray[i].packet_size = 0
        packetRecordArray[i].frame_nb = frameNb
        packetRecordArray[i].frame_type = 'S' + str(mainFrameNb)
        packetRecordArray[i].layer_nb = i
        packetRecordArray[i].block_seq_vector = []
    
    cycleNb = frame.shape[0] * frame.shape[1] * (para.CYCLES_PER_ADD + para.CYCLES_PER_MUL)
    entropyCycles = 0

    for i in range(0, frame.shape[0], 8):
        for j in range(0, frame.shape[1], 8):
            blockNb = i * frame.shape[1] // 64 + j // 8
            block_ = frame[i:i + 8, j:j + 8]
            ms = util.get_ms(block_.copy())
            blockPriority = 4
            if ms > 0:
                if ms >=  msv[0]: blockPriority = 0
                elif ms >=  msv[1]: blockPriority = 1
                elif ms >=  msv[2]: blockPriority = 2
                elif ms >=  msv[3]: blockPriority = 3
                else: blockPriority = 4

                cycleNb += 192  # 3 * 64 cycles in red2 (thresholding)
                _, block =cv2.threshold(block_, para.threshold, 255, cv2.THRESH_BINARY_INV)

                if cv2.countNonZero(block) == 0 or blockPriority > para.max_level_S:
                    continue

                retFrame[i:i + 8, j:j + 8] = block
                linearBlock = block.flatten()
                prevBlockNb = 0
                try:
                    if len(packetRecordArray[blockPriority].block_seq_vector) != 0:
                        prevBlockNb = len(packetRecordArray[blockPriority].block_seq_vector) - 1 
                except :
                    #print('a')
                compressedLayerData, entropyCycles = entropy.entropy_coder(linearBlock, blockNb, para.entropy_coding)
                #compressedLayerSize, entropyCycles = entropy.blockEntropyCoder(para.entropy_coding, linearBlock, blockNb, prevBlockNb)
                compressedLayerSize = len(compressedLayerData)
                cycleNb += entropyCycles

                if compressedLayerSize > para.packed_load_size * 8:
                    #print(f"S frame Very small payload size !! Block {blockNb} requires {int(compressedLayerSize / 8) + 1} bytes.")
                    exit(EXIT_FAILURE)

                if packetRecordArray[blockPriority].packet_size + compressedLayerSize <  para.packed_load_size * 8:
                    packetRecordArray[blockPriority].packet_size += compressedLayerSize
                    packetRecordArray[blockPriority].layer_nb = blockPriority
                    packetRecordArray[blockPriority].block_seq_vector.append(blockNb)
                else:
                    packetRecordArray[blockPriority].seq_nb = seqNb[0] + 1
                    util.write_packet_record_s_frame(packetRecordArray[blockPriority], para.trace_file_path, total_loss_db, snr, ber)
                    compressedSize += packetRecordArray[blockPriority].packet_size
                    packetRecordArray[blockPriority].block_seq_vector = [blockNb]
                    #compressedLayerSize, entropyCycles = entropy.blockEntropyCoder(para.entropy_coding, linearBlock, blockNb, prevBlockNb)
                    packetRecordArray[blockPriority].packet_size = compressedLayerSize
                    cycleNb += entropyCycles

    energy = cycleNb * para.POWER / para.PROC_CLOCK / 1000 

    for k in range(para.MAX_S_LAYERS):
        if packetRecordArray[k].packet_size > 0:
            packetRecordArray[k].seq_nb = seqNb[0] + 1
            util.write_packet_record_s_frame(packetRecordArray[k], para.trace_file_path, total_loss_db, snr, ber)
            compressedSize += packetRecordArray[k].packet_size
    
    compressedSize = compressedSize
    return retFrame, compressedSize, energy
