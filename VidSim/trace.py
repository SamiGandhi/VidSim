import numpy as np
from parameters import Parameters as para
class frame_record:
    def __init__(self):
        self.frameNb = 0            # Frame number
        self.frameType = ""         # Frame type (e.g., I-frame, P-frame)
        self.frameSize = 0          # Frame size in bytes
        self.PSNR = 0.0             # Peak Signal-to-Noise Ratio
        self.SSIM = 0.0             # Structural Similarity Index
        self.bpp = 0.0              # Bits per pixel
        self.bitRate = 0.0          # Bit rate in Kbps
        self.layersSizeVector = []  # List of layers size of a frame (bits)
        self.captureEnergy = 0.0    # Capture energy in mJ
        self.encodingEnergy = 0.0   # Encoding energy in mJ

class packet_record:
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
    def __init__(self):
        self.decoded_frame = np.zeros((para.default_height, para.default_width), dtype=np.uint8)
        self.PSNR = 0.0
        self.SSIM = 0.0
        self.mask_frame =  np.ones((para.default_height, para.default_width), dtype=np.uint8)  # Initialize this with a NumPy array (e.g., np.zeros) with appropriate dimensions
        self.frame_type = "N"
        self.sad_map = np.zeros((para.default_height, para.default_width), dtype=np.uint8) # Initialize this with a NumPy array (e.g., np.zeros) with appropriate dimensions
        self.r_map =  np.zeros((para.default_height, para.default_width), dtype=np.uint8)  # Initialize this with a NumPy array (e.g., np.zeros) with appropriate dimensions
        self.g_map =  np.zeros((para.default_height, para.default_width), dtype=np.uint8)  # Initialize this with a NumPy array (e.g., np.zeros) with appropriate dimensions
        self.ber_matrix = np.zeros((para.default_height//8, para.default_width//8), dtype=float)

