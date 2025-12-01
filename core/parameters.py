# parameters.py
# Project-wide configuration via Parameters class for paths, codec, and experiment options.
# All encoding/decoding operational parameters, directory paths, and codec settings are centralized here.

import os

class Parameters:
    """
    Central configuration and parameter storage for video coding pipeline.
    Every attribute can be tuned for a full experiment run. See below for directory, codec, and system options.
    """
    captured_video_path = ''  # Path to the source video file to process
    video_base_path = ''      # Base directory (stripped extension of captured), set up from captured_video_path
    captured_frame_dir = ''   # Directory for intermediate captured frames (PNG)
    masks_dir = ''            # Path to store generated ROI masks
    reference_frames_dir = '' # Directory for storing reference reconstructed frames
    roi_frame_dir = ''        # Directory for storing ROI region images
    output_directory = ''     # If set, all outputs use this root rather than auto path
    decoded_frames_dir = ''   # Directory for storing decoded images (output)
    fps = 2                   # Output video target frames per second
    default_width = 144       # Default width for processing
    default_height = 144      # Default height for processing
    gop_coefficient = 30      # MSE threshold to split GOPs (scene change)
    quality_factor = 80       # JPEG/MJPEG quality factor for standard compression
    high_quality_factor = 40  # Quality factor for high-priority ROI blocks
    low_quality_factor = 20   # Quality for medium-priority ROI blocks
    zone_size = 8             # Block size for DCT/ROI segmentation
    DCT = "CLA"               # DCT algorithm choice
    level_numbers = 1         # Layer count for scalable coding
    entropy_coding = "RLE_EG" # Entropy coding string
    trace_file_path = ""      # Directory for writing trace files (created per run)
    packed_load_size = 512    # Max packet (transport) size per layer
    interleaving = ""         # Interleaving strategy, if any
    threshold = 30            # Thresholding for binary mask or block selection
    max_level_S = 0           # Max scalability level for S-frames
    enhance_method = ''       # Enhancement method for post-decoding recovery
    enhance_param = ''        # Parameters for enhancement method
    w1 = 8                    # ROI mask window sizes for sad_map, r_map, g_map
    w2 = 4
    w3 = 2
    roi_threshold = 10        # Block mean threshold for region detection
    method = ''               # 'ROI' or alternative coding strategy
    POWER = 0.023             # Device or algorithmic power value (used for energy calc)
    PROC_CLOCK = 7.3728       # Processor clock rate (energy estimation)
    CAPTURE_E_PER_BLOCK = 2.65e-6 # Energy per captured block
    CYCLES_PER_DIV = 300
    CYCLES_PER_ADD = 1
    CYCLES_PER_FADD = 20
    CYCLES_PER_SHIFT = 1
    CYCLES_PER_MUL = 2
    CYCLES_PER_FMUL = 100
    CLA_ADD_NB = 464
    CLA_MUL_NB = 192
    MAX_S_LAYERS = 5
    DISTANCE = 100  # meters    # Transmission scenario parameters below
    FREQUENCY = 868e6  # Hz
    ENVIRONMENT = "wetland"
    HUMIDITY_LEVEL = 70  # percent
    VEGETATION_DENSITY_LEVEL = 5  # arbitrary unit
    scale_factor = 100   # Misc. scaling factor
    zero = 1             # Value for dropped/zero compression blocks

    @classmethod
    def setup_directories(cls):
        """Sets up working directory structure depending on method and parameter values."""
        if(cls.method == 'ROI'):
            directory = f"{cls.default_height}X{cls.default_width}/HQF_{cls.high_quality_factor}_LQF_{cls.low_quality_factor}/GOP_{cls.gop_coefficient}"
        else:
            directory = f"{cls.default_height}X{cls.default_width}/QF_{cls.quality_factor}/GOP_{cls.gop_coefficient}"
        cls.video_base_path, _ = os.path.splitext(cls.captured_video_path)
        if cls.video_base_path == '':
            raise ValueError("Must Select a Video File to start")
        if cls.output_directory == '':
            cls.captured_frame_dir = os.path.join(cls.video_base_path, directory, 'captured_frames')
            cls.reference_frames_dir = os.path.join(cls.video_base_path, directory, 'reference_frames')
            cls.decoded_frames_dir = os.path.join(cls.video_base_path, directory, 'decoded')
            cls.trace_file_path = os.path.join(cls.video_base_path, directory, 'trace')
            cls.roi_frame_dir = os.path.join(cls.video_base_path, directory, 'roi_frames')
            cls.masks_dir = os.path.join(cls.video_base_path, directory, 'roi_masks')
        else:
            cls.captured_frame_dir = os.path.join(cls.output_directory, directory, 'captured_frames')
            cls.reference_frames_dir = os.path.join(cls.output_directory, directory, 'reference_frames')
            cls.decoded_frames_dir = os.path.join(cls.output_directory, directory, 'decoded')
            cls.trace_file_path = os.path.join(cls.output_directory, directory, 'trace')
            cls.roi_frame_dir = os.path.join(cls.output_directory, directory, 'roi_frames')
            cls.masks_dir = os.path.join(cls.output_directory, directory, 'roi_masks')

    @classmethod
    def print_params(cls):
        """#prints current configuration values for visual confirmation/debugging."""
        print(f"""
        captured_video_path = {cls.captured_video_path}
        video_base_path = {cls.video_base_path}
        captured_frame_dir = {cls.captured_frame_dir}
        masks_dir = {cls.masks_dir}    
        reference_frames_dir = {cls.reference_frames_dir}
        roi_frame_dir = {cls.roi_frame_dir}
        output_directory = {cls.output_directory}
        decoded_frames_dir = {cls.decoded_frames_dir}
        fps = {cls.fps}
        default_width = {cls.default_width}
        default_height = {cls.default_height}
        gop_coefficient = {cls.gop_coefficient}
        quality_factor = {cls.quality_factor}
        high_quality_factor = {cls.high_quality_factor}
        low_quality_factor = {cls.low_quality_factor}
        zone_size = {cls.zone_size}
        DCT = {cls.DCT}
        level_numbers = {cls.level_numbers}
        entropy_coding = {cls.entropy_coding}
        trace_file_path = {cls.trace_file_path}
        packed_load_size = {cls.packed_load_size}
        interleaving = {cls.interleaving}
        threshold = {cls.threshold}
        max_level_S = {cls.max_level_S}
        enhance_method = {cls.enhance_method}
        enhance_param = {cls.enhance_param}
        w1 = {cls.w1}
        w2 = {cls.w2}
        w3 = {cls.w3}
        roi_threshold = {cls.roi_threshold}
        method = {cls.method}
        POWER = {cls.POWER}
        PROC_CLOCK = {cls.PROC_CLOCK}
        CAPTURE_E_PER_BLOCK = {cls.CAPTURE_E_PER_BLOCK}
        CYCLES_PER_DIV = {cls.CYCLES_PER_DIV}
        CYCLES_PER_ADD = {cls.CYCLES_PER_ADD}
        CYCLES_PER_FADD = {cls.CYCLES_PER_FADD}
        CYCLES_PER_SHIFT = {cls.CYCLES_PER_SHIFT}
        CYCLES_PER_MUL = {cls.CYCLES_PER_MUL}
        CYCLES_PER_FMUL = {cls.CYCLES_PER_FMUL}
        CLA_ADD_NB = {cls.CLA_ADD_NB}
        CLA_MUL_NB = {cls.CLA_MUL_NB}
        MAX_S_LAYERS = {cls.MAX_S_LAYERS}
        DISTANCE = {cls.DISTANCE}
        FREQUENCY = {cls.FREQUENCY}
        ENVIRONMENT = {cls.ENVIRONMENT}
        HUMIDITY_LEVEL = {cls.HUMIDITY_LEVEL}
        VEGETATION_DENSITY_LEVEL = {cls.VEGETATION_DENSITY_LEVEL}
        scale_factor = {cls.scale_factor}
        """)

    @classmethod
    def reset(cls):
        """Restore default parameter values for fresh experiment runs."""
        cls.fps = 2
        cls.default_width = 144
        cls.default_height = 144
        cls.gop_coefficient = 30
        cls.quality_factor = 80
        cls.high_quality_factor = 40
        cls.low_quality_factor = 20
        cls.zone_size = 8
        cls.DCT = "CLA"
        cls.level_numbers = 1
        cls.entropy_coding = "RLE_EG"
        cls.trace_file_path = ""
        cls.packed_load_size = 512
        cls.interleaving = "None"
        cls.threshold = 30
        cls.max_level_S = 0
        cls.enhance_method = ''
        cls.enhance_param = ''
        cls.w1 = 8
        cls.w2 = 4
        cls.w3 = 2
        cls.roi_threshold = 10
        cls.method = 'Non-ROI'
        cls.POWER = 0.023
        cls.PROC_CLOCK = 7.3728
        cls.CAPTURE_E_PER_BLOCK = 2.65e-6
        cls.CYCLES_PER_DIV = 300
        cls.CYCLES_PER_ADD = 1
        cls.CYCLES_PER_FADD = 20
        cls.CYCLES_PER_SHIFT = 1
        cls.CYCLES_PER_MUL = 2
        cls.CYCLES_PER_FMUL = 100
        cls.CLA_ADD_NB = 464
        cls.CLA_MUL_NB = 192
        cls.MAX_S_LAYERS = 5
        cls.DISTANCE = 100  # meters
        cls.FREQUENCY = 868e6  # Hz
        cls.ENVIRONMENT = "wetland"
        cls.HUMIDITY_LEVEL = 70  # percent
        cls.VEGETATION_DENSITY_LEVEL = 5  # arbitrary unit
        cls.scale_factor = 100
    



