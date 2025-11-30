# second_frame_.py
# Handles encoding and packetizing of secondary frames (i.e. P/B frames or S-frames), using prediction and difference from main frames for video coding.
# Contains logic for ROI- and non-ROI S-frame processing for both reference and reduction encoding.

import numpy as np
import cv2
import core.entropy as entropy
import core.util as util
import os
import core.network_losses_model as losses
from core.parameters import Parameters as parameters
import metrics.quality_metrics as metrics
import core.main_frame as main_frame
import core.dct as dct
import core.trace as trace

EXIT_FAILURE = 0


def encode_second_frame_roi(currentFrame, mainFrameNb, frameRecord, seqNb, r_map, g_map):
    """
    Encode an ROI-based secondary frame using reference and difference images between current and main frame.
    Updates frameRecord with compressed size, metrics and calls reference frame generation.
    """
    refMainFrame = cv2.imread(parameters.reference_frames_dir + "/frame" + str(mainFrameNb) + ".png", cv2.IMREAD_GRAYSCALE)
    mainFrame = cv2.imread(parameters.captured_frame_dir + "/frame" + str(mainFrameNb) + ".png", cv2.IMREAD_GRAYSCALE)
    currentS = currentFrame.frame.astype('int16')
    mainS = mainFrame.astype('int16')
    refMainS = refMainFrame.astype('int16')
    energy = [0]
    # Computing the difference, reduction, and reference update
    redFrame, compressedSize, energy = reduce_and_packetize_second_frame(currentS - mainS, currentFrame.frame_number, mainFrameNb, seqNb)
    refCurrentS = refMainS + redFrame
    diffReferenceFrame = refCurrentS.astype('uint8')
    imageName = parameters.reference_frames_dir + "/frame" + str(currentFrame.frame_number) + ".png"
    cv2.imwrite(imageName, diffReferenceFrame)
    frameRecord.frameSize = compressedSize
    frameRecord.bpp = frameRecord.frameSize * 1.0 / (currentFrame.frame.shape[1] * currentFrame.frame.shape[0])
    frameRecord.bitRate = frameRecord.frameSize * parameters.fps / 1000
    frameRecord.PSNR = metrics.calculate_psnr(currentFrame.frame.copy(), diffReferenceFrame.copy())
    frameRecord.SSIM = metrics.calculate_ssim(currentFrame.frame.copy(), diffReferenceFrame.copy())
    reconstructed_image_3ch = cv2.cvtColor(currentFrame.frame.copy(), cv2.COLOR_GRAY2BGR)
    frameRecord.BRISQUE = metrics.calculate_brisque(reconstructed_image_3ch)
    frameBlocksNb = (currentFrame.frame.shape[1] * currentFrame.frame.shape[0]) // 64
    frameRecord.captureEnergy = parameters.CAPTURE_E_PER_BLOCK * frameBlocksNb * 1000  # mJ
    frameRecord.encodingEnergy = energy
    util.write_frame_record(frameRecord)


def encodeSecondFrame(currentFrame, mainFrameNb, frameRecord, seqNb):
    """
    Encode a (non-ROI) secondary frame based on prediction/ref from main frame, using block MS value for priority.
    Applies block-level DCT, quantization, entropy coding and packetization.
    """
    refMainFrame = cv2.imread(parameters.reference_frames_dir + "/frame" + str(mainFrameNb) + ".png", cv2.IMREAD_GRAYSCALE)
    mainFrame = cv2.imread(parameters.captured_frame_dir + "/frame" + str(mainFrameNb) + ".png", cv2.IMREAD_GRAYSCALE)
    currentS = currentFrame.frame.astype('int16')
    mainS = mainFrame.astype('int16')
    refMainS = refMainFrame.astype('int16')
    energy = [0]
    redFrame, compressedSize, energy = reduce_and_packetize_second_frame(currentS - mainS, currentFrame.frame_number, mainFrameNb, seqNb)
    refCurrentS = refMainS + redFrame
    diffReferenceFrame = refCurrentS.astype('uint8')
    imageName = parameters.reference_frames_dir + "/frame" + str(currentFrame.frame_number) + ".png"
    cv2.imwrite(imageName, diffReferenceFrame)
    frameRecord.frameSize = compressedSize
    frameRecord.bpp = frameRecord.frameSize * 1.0 / (currentFrame.frame.shape[1] * currentFrame.frame.shape[0])
    frameRecord.bitRate = frameRecord.frameSize * parameters.fps / 1000
    frameRecord.PSNR = metrics.calculate_psnr(currentFrame.frame.copy(), diffReferenceFrame.copy())
    frameRecord.SSIM = metrics.calculate_ssim(currentFrame.frame.copy(), diffReferenceFrame.copy())
    reconstructed_image_3ch = cv2.cvtColor(currentFrame.frame.copy(), cv2.COLOR_GRAY2BGR)
    frameRecord.BRISQUE = metrics.calculate_brisque(reconstructed_image_3ch)
    frameBlocksNb = (currentFrame.frame.shape[1] * currentFrame.frame.shape[0]) // 64
    frameRecord.captureEnergy = parameters.CAPTURE_E_PER_BLOCK * frameBlocksNb * 1000  # mJ
    frameRecord.encodingEnergy = energy
    util.write_frame_record(frameRecord)

def reduce_and_packetize_second_frame(frame, frameNb, mainFrameNb, seqNb):
    """
    Reduces the frame size by applying block-level DCT, quantization, and entropy coding.
    Packetsize is calculated based on the block's priority and the current sequence number.
    """
    try:
        snr, total_loss_db = losses.combined_loss_model(parameters.DISTANCE, parameters.FREQUENCY, parameters.ENVIRONMENT, parameters.HUMIDITY_LEVEL, parameters.VEGETATION_DENSITY_LEVEL)    
        ber = losses.calculate_ber(snr)
    except:
        snr = 0
        ber = 0
        total_loss_db = 0

    compressedSize = 0
    msv = [650, 205, 51, 13]  # Modify these thresholds based on empirical data
    retFrame = np.zeros(frame.shape, dtype=np.int16)
    packetRecordArray = [trace.packet_record() for _ in range(parameters.MAX_S_LAYERS)]
    
    for i in range(parameters.MAX_S_LAYERS):
        packetRecordArray[i].send_time = (frameNb - 1) / parameters.fps
        packetRecordArray[i].packet_size = 0
        packetRecordArray[i].frame_nb = frameNb
        packetRecordArray[i].frame_type = 'S' + str(mainFrameNb)
        packetRecordArray[i].layer_nb = i
        packetRecordArray[i].block_seq_vector = []
    
    cycleNb = 0  # Initialize cycle count for S frames
    frame_height_blocks = frame.shape[0] // 8
    frame_width_blocks = frame.shape[1] // 8
    calcul = 0
    treated_blocs_nb = 0


    for i in range(frame_height_blocks):
        for j in range(frame_width_blocks):
            blockNb = i * frame_width_blocks + j
            block = frame[i*8:(i+1)*8, j*8:(j+1)*8]
            ms = util.getMS(block.copy())
            blockPriority = 4  # Default to lowest priority
            if ms > 0:
                # Prioritize blocks based on MS value
                if ms >=  msv[0]: blockPriority = 0
                elif ms >=  msv[1]: blockPriority = 1
                elif ms >=  msv[2]: blockPriority = 2
                elif ms >=  msv[3]: blockPriority = 3

                cycleNb += 192  # 3 * 64 cycles in red2 (thresholding)
                cv2.threshold(block, parameters.threshold, 255, cv2.THRESH_BINARY_INV)

                if cv2.countNonZero(block) == 0 or blockPriority > parameters.max_level_S:
                    continue

                retFrame[i*8:(i+1)*8, j*8:(j+1)*8] = block
                prevBlockNb = 0
                if len(packetRecordArray[blockPriority].block_seq_vector) != 0:
                    prevBlockNb = packetRecordArray[blockPriority].block_seq_vector[-1]
                floatBlock = block.astype(np.float32)
                floatBlock -= 128
                # Apply DCT
                dctBlock = dct.DCT(floatBlock, parameters.DCT, parameters.zone_size)
                dctFloatBlock = dctBlock.astype(np.float32)

                quantizationMatrix = getQuantisationMatrix(parameters.quality_factor)
                quantifBlock = np.divide(dctFloatBlock, quantizationMatrix).astype(np.int16)
                # Apply zigzag scan
                zigzagBlock = main_frame.zigzagScan(quantifBlock)
                # Apply entropy coding
                compressedLayerSize, entropyCycles = entropy.blockEntropyCoder(parameters.entropy_coding, zigzagBlock, blockNb, prevBlockNb)
                cycleNb += entropyCycles

                if compressedLayerSize > parameters.packed_load_size * 8:
                    #print(f"S frame Very small payload size !! Block {blockNb} requires {int(compressedLayerSize / 8) + 1} bytes.")
                    exit(EXIT_FAILURE)

                if packetRecordArray[blockPriority].packet_size + compressedLayerSize <  parameters.packed_load_size * 8:
                    packetRecordArray[blockPriority].packet_size += compressedLayerSize
                    packetRecordArray[blockPriority].layer_nb = blockPriority
                    packetRecordArray[blockPriority].block_seq_vector.append(blockNb)
                else:
                    packetRecordArray[blockPriority].seq_nb = seqNb[0] + 1
                    util.write_packet_record_m_frame(packetRecordArray[blockPriority], parameters.trace_file_path, total_loss_db, snr, ber)
                    compressedSize += packetRecordArray[blockPriority].packet_size
                    packetRecordArray[blockPriority].block_seq_vector = [blockNb]
                    #compressedLayerSize, entropyCycles = entropy.blockEntropyCoder(parameters.entropy_coding, quantifBlock, blockNb, prevBlockNb)
                    packetRecordArray[blockPriority].packet_size = compressedLayerSize
                    #cycleNb += entropyCycles
                treated_blocs_nb+= 1
    energy = (cycleNb * parameters.POWER / parameters.PROC_CLOCK) / 1000  # Updated energy calculation

    
    for k in range(parameters.MAX_S_LAYERS):
        if packetRecordArray[k].packet_size > 0:
            packetRecordArray[k].seq_nb = seqNb[0] + 1
            util.write_packet_record_m_frame(packetRecordArray[k], parameters.trace_file_path, total_loss_db, snr, ber)
            compressedSize += packetRecordArray[k].packet_size
    return retFrame, compressedSize, energy

def getQuantisationMatrix(QC):
    """
    Generates a quantization matrix based on the quantization factor (QC).
    For QC=0, it returns a matrix with all values set to infinity, effectively dropping all coefficients.
    For other QC values, it calculates a scaling factor S and applies it to a standard quantization matrix.
    """
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
