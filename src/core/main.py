# main.py
# Main driver for video encoding and decoding pipeline.
# Handles the overall process, manages frame capturing, and delegates to encoding/decoding functions
# Uses utilities and submodules for ROI encoding, frame handling, metric calculation, and directory management

import core.capture as cap
import math
import numpy as np
import core.util as util
import core.main_frame as main_frame
import core.second_frame_ as second_frame
import core.trace as trace
import core.decoder as decoder
import core.roi as roi
import cv2
from core.parameters import Parameters as para


def _get_frame_count_for_decoding(is_roi: bool) -> int:
    """Return frame count for decoding, using cached frames if video path is unavailable."""
    if para.captured_video_path:
        frame_nb = cap.capture_frames()
    else:
        frame_nb = util.count_captured_frames(para.captured_frame_dir)
        if frame_nb == 0:
            raise ValueError("No captured frames available. Provide --video-path or rerun encoding.")
    if is_roi:
        frame_nb = max(frame_nb - 1, 0)
    return frame_nb


def run_coding():
    """
    Encodes a video sequence based on defined parameters.
    Handles two methods:
      - 'ROI': Region of Interest-based encoding with mask and ROI trace management.
      - Else: Standard GOP coefficient-based frame encoding.
    Initializes directories, manages frame capturing, processes each frame accordingly.
    """
    seq_nb = [0]
    if para.method == 'ROI':
        captured_frames_list = []
        # Create/clean directories for output data
        # util.make_dir(para.output_directory)
        util.make_dir(para.trace_file_path)
        util.make_dir(para.captured_frame_dir)
        util.delete_folder_content(para.captured_frame_dir)
        util.make_dir(para.reference_frames_dir)
        util.delete_folder_content(para.reference_frames_dir)
        util.make_dir(para.roi_frame_dir)
        util.create_trace_files(para.trace_file_path)
        util.delete_folder_content(para.masks_dir)
        util.make_dir(para.masks_dir)
        frame_nb = cap.capture_frames()
        for i in range(1,frame_nb+1):
            frame_record = trace.frame_record()
            frame_record.frameNb = i
            frame_record.frameType = 'M'  # All frames are main in ROI mode
            frame_record.layersSizeVector.clear()
            frame_record.layersSizeVector = [0] * para.level_numbers
            frame = cap.get_captured_frame(i)
            if i == 1:  # First frame is stored as it is
                prev = frame
            else:  # Retrieve the ROI for subsequent frames
                gray_frame = frame
                sad_map, r_mask, g_mask = roi.get_masks(gray_frame, prev)
                c1, c2, c3 = roi.get_compression_masks(sad_map, r_mask, g_mask)
                # Save mask and ROI images for analysis/debug
                cv2.imwrite(f"{para.masks_dir}/sadmap_frame{i}.png", sad_map)
                cv2.imwrite(f"{para.masks_dir}/rmask_frame{i}.png", r_mask)
                cv2.imwrite(f"{para.masks_dir}/gmask_frame{i}.png", g_mask)
                cv2.imwrite(f"{para.masks_dir}/c1_frame{i}.png", c1)
                cv2.imwrite(f"{para.masks_dir}/c2_frame{i}.png", c2)
                cv2.imwrite(f"{para.masks_dir}/c3_frame{i}.png", c3)
                roi_frame = roi.apply_mask(gray_frame, g_mask)
                cv2.imwrite(f"{para.roi_frame_dir}/roi_frame{i}.png", roi_frame)
                main_frame.encode_roi_main_frame(roi_frame, frame_record, i, seq_nb, c1, c2, c3)
                prev = frame
    else:
        captured_frames_list = []
        # Create/clean directories for output data
        # util.make_dir(para.output_directory)
        util.make_dir(para.trace_file_path)
        util.make_dir(para.captured_frame_dir)
        util.delete_folder_content(para.captured_frame_dir)
        util.make_dir(para.reference_frames_dir)
        util.delete_folder_content(para.reference_frames_dir)
        util.create_trace_files(para.trace_file_path)
        frame_nb = cap.capture_frames()
        # frame_nb = cap.capture_frames_original_fps()
        for i in range(1,frame_nb+1):
            frame = cap.get_captured_frame(i)
            frame_record = trace.frame_record()
            frame_record.frameNb = i
            if i > 1:
                # Calculate the Mean Squared Error (MSE) between frames for scene change detection
                if main_frame_.frame.shape != frame.shape:
                    raise RuntimeError(f"Error: Frame number: {main_frame_.frame_number} and Frame number: {i} have different dimensions")
                    #print(f"Error: Frame number: {main_frame_.frame_number} and Frame number: {i} have different dimensions")
                mse = util.getMSE_2(main_frame_.frame, frame)
                mse = math.sqrt(mse)
            if i == 1 or mse > para.gop_coefficient:
                captured_frame = cap.Captured_Frame(frame, i, 'M')  # Main frame
                captured_frames_list.append(captured_frame)
                frame_record.frameType = 'M'
                frame_record.layersSizeVector.clear()
                frame_record.layersSizeVector = [0] * para.level_numbers
                main_frame.encode_main_frame(captured_frame.frame, frame_record, i, seq_nb)
                main_frame_ = captured_frame
                # Encode main frame
            else:
                frame_record.frameType = 'S'  # "S" for secondary or predicted/side frame
                captured_frame = cap.Captured_Frame(frame, i, 'S')
                captured_frame.reference_frame = main_frame_.frame_number
                captured_frames_list.append(captured_frame)
                second_frame.encodeSecondFrame(captured_frame, main_frame_.frame_number, frame_record, seq_nb)
                # Encode S frame
        # #print frame type information for user/debug info
        for captured_frame in captured_frames_list:
            if captured_frame.frame_type == 'M':
                #Options for debugging
                pass
                #print(f'Frame number: {captured_frame.frame_number} is {captured_frame.frame_type} frame.')
            else:
                #print(f'Frame number: {captured_frame.frame_number} is {captured_frame.frame_type} frame, its refrence frame is {captured_frame.reference_frame}.')
                pass


def run_decoding():
    """
    Handles the decoding of captured or encoded video frames.
    Delegates to ROI or non-ROI decoder pipelines depending on parameters.
    """
    util.make_dir(para.decoded_frames_dir)
    if para.method == 'ROI':
        frame_nb = _get_frame_count_for_decoding(is_roi=True)
        decoder.build_received_video_roi(frame_nb)
    else:
        frame_nb = _get_frame_count_for_decoding(is_roi=False)
        decoder.build_received_video(frame_nb)

