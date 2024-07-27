import numpy as np
import cv2
import entropy
from parameters import Parameters as para
import util
import trace
import network_losses_model as losses



EXIT_FAILURE = 0







def encodeSecondFrame(currentFrame, mainFrameNb, frameRecord, seqNb):
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
    frameRecord.PSNR = util.get_psnr(currentFrame.frame.copy(), diffReferenceFrame.copy())
    frameRecord.SSIM = util.get_ssim(currentFrame.frame.copy(), diffReferenceFrame.copy())
    frameBlocksNb = (currentFrame.frame.shape[1] * currentFrame.frame.shape[0]) // 64
    frameRecord.captureEnergy = para.CAPTURE_E_PER_BLOCK * frameBlocksNb * 1000  # mJ
    frameRecord.encodingEnergy = energy

    util.write_frame_record(frameRecord)







def reduce_and_packetize_second_frame(frame, frameNb, mainFrameNb, seqNb):
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
                if ms >=  msv[0]: blockPriority = 0
                elif ms >=  msv[1]: blockPriority = 1
                elif ms >=  msv[2]: blockPriority = 2
                elif ms >=  msv[3]: blockPriority = 3
                else: blockPriority = 4

                cycleNb += 192  # 3 * 64 cycles in red2 (thresholding)
                cv2.threshold(block, para.threshold, 255, cv2.THRESH_BINARY_INV)

                if cv2.countNonZero(block) == 0 or blockPriority > para.max_level_S:
                    continue

                retFrame[i:i + 8, j:j + 8] = block
                linearBlock = block.flatten()
                prevBlockNb = 0
                try:
                    if len(packetRecordArray[blockPriority].block_seq_vector) != 0:
                        prevBlockNb = len(packetRecordArray[blockPriority].block_seq_vector) - 1 
                except :
                    print('a')
                compressedLayerSize, entropyCycles = entropy.blockEntropyCoder(para.entropy_coding, linearBlock, blockNb, prevBlockNb)
                cycleNb += entropyCycles

                if compressedLayerSize > para.packed_load_size * 8:
                    print(f"S frame Very small payload size !! Block {blockNb} requires {int(compressedLayerSize / 8) + 1} bytes.")
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
                    compressedLayerSize, entropyCycles = entropy.blockEntropyCoder(para.entropy_coding, linearBlock, blockNb, prevBlockNb)
                    packetRecordArray[blockPriority].packet_size = compressedLayerSize
                    cycleNb += entropyCycles

    energy = cycleNb * para.POWER / para.PROC_CLOCK / 1000 

    for k in range(para.MAX_S_LAYERS):
        if packetRecordArray[k].packet_size > 0:
            packetRecordArray[k].seq_nb = seqNb[0] + 1
            util.write_packet_record_s_frame(packetRecordArray[k], para.trace_file_path, total_loss_db, snr, ber)
            compressedSize += packetRecordArray[k].packet_size
    
    compressedSize = compressedSize/8
    return retFrame, compressedSize, energy
