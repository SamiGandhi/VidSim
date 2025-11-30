# trace.py
# Data structures for logging and tracking frame and packet data in the video simulation pipeline.
# Defines record classes used for keeping metadata, statistics, and buffering between stages in encoding/decoding.

import numpy as np
from core.parameters import Parameters as para

class frame_record:
    """
    Stores metadata and statistics for an encoded frame.
    Attributes:
        frameNb: Frame sequence number.
        frameType: String (e.g., I-frame, P-frame) or role in stream.
        frameSize: Compressed size in bytes.
        PSNR: Peak Signal-to-Noise Ratio of the frame.
        SSIM: Structural Similarity Index of the frame.
        BRISQUE: Blind assessment metric for perceived quality.
        bpp: Bits-per-pixel.
        bitRate: Bits per second or Kbps.
        layersSizeVector: List of bits per scalable layer for the frame.
        captureEnergy: Energy to capture the frame (mJ).
        encodingEnergy: Energy for encoding the frame (mJ).
    """
    def __init__(self):
        self.frameNb = 0            # Frame number
        self.frameType = ""         # Frame type (e.g., I-frame, P-frame)
        self.frameSize = 0          # Frame size in bytes
        self.PSNR = 0.0             # Peak Signal-to-Noise Ratio
        self.SSIM = 0.0             # Structural Similarity Index
        self.BRISQUE = 100.0        # Perceptual quality metric
        self.bpp = 0.0              # Bits per pixel
        self.bitRate = 0.0          # Bit rate in Kbps
        self.layersSizeVector = []  # Scalable layer sizes (bits)
        self.captureEnergy = 0.0    # Capture energy in mJ
        self.encodingEnergy = 0.0   # Encoding energy in mJ

class packet_record:
    """
    Represents a transmitted packet in the simulated network.
    Attributes:
        send_time: When (relative, e.g., seconds) the packet was sent.
        seq_nb: Sequence number for ordering packets.
        packet_size: Total payload size in bits or bytes.
        frame_nb: Associated frame number.
        frame_type: The type/role of the corresponding frame.
        layer_nb: The data layer in scalable stream.
        block_seq_vector: Block indices covered by this packet.
        blocks_compression_vector: Compression level info per block if applicable.
    """
    def __init__(self):
        self.send_time = 0
        self.seq_nb = 0
        self.packet_size = 0
        self.frame_nb = 0
        self.frame_type = ''
        self.layer_nb = 0
        self.block_seq_vector = []
        self.blocks_compression_vector = []

class frame_decoded:
    """
    Keeps decoded frame data and post-decoding statistics used for analysis.
    Attributes:
        decoded_frame: Output ndarray (image) of the decoded frame.
        PSNR, SSIM, BRISQUE: Quality metrics of the decoded output vs. source/reference
        BRISQUE_ref: BRISQUE metric for ground truth comparison.
        mask_frame: Masking array indicating valid/processed/unmasked pixels.
        frame_type: Type/role after decoding (M, S, N, etc.).
        sad_map, r_map, g_map: Block-level mask/status maps for ROI analyses.
        ber_matrix: Per-block Bit Error Rate simulation map.
    """
    def __init__(self):
        self.decoded_frame = np.zeros((para.default_height, para.default_width), dtype=np.uint8)
        self.PSNR = 0.0
        self.SSIM = 0.0
        self.BRISQUE = 100.0
        self.BRISQUE_ref = 100.0
        self.mask_frame =  np.ones((para.default_height, para.default_width), dtype=np.uint8)  # Mask for frame status
        self.frame_type = "N"
        self.sad_map = np.zeros((para.default_height, para.default_width), dtype=np.uint8)
        self.r_map =  np.zeros((para.default_height, para.default_width), dtype=np.uint8)
        self.g_map =  np.zeros((para.default_height, para.default_width), dtype=np.uint8)
        self.ber_matrix = np.zeros((para.default_height//8, para.default_width//8), dtype=float)

