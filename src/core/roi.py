# roi.py
# Region of Interest mask calculation and mask-related utilities for video coding pipeline.
# Offers SAD, R, G block maps and composite ROI masks for frame prioritization.

import cv2 as cv
import numpy as np
from core.parameters import Parameters as para


def apply_mask(img, mask):
    """Apply a binary mask to an image using bitwise AND."""
    result = cv.bitwise_and(img, img, mask=mask)
    return result


def abs_diff(frame, previous_frame):
    """Compute the absolute difference between two grayscale frames."""
    return cv.absdiff(frame, previous_frame)


def mapping(diff_image, w):
    """
    Maps average block difference to binary output image using a threshold for region classification.
    Each block is set to 0 (background) or 255 (ROI) based on mean.
    """
    height, width = diff_image.shape
    output_image = np.zeros_like(diff_image, dtype=np.uint8)
    for x in range(0, height - w + 1, w):
        for y in range(0, width - w + 1, w):
            block_sum = np.sum(diff_image[x:x + w, y:y + w])
            block_mean = block_sum / (w * w)
            if block_mean < para.roi_threshold:
                block_mean = 0
            else:
                block_mean = 255
            output_image[x:x + w, y:y + w] = block_mean
    return output_image


def sad_map(diff_image, w1):
    """
    Generate blockwise SAD (Sum of Absolute Difference) based ROI mask using window size w1.
    """
    return mapping(diff_image, w1)


def r_map(diff_image, w1, w2):
    """
    Region mask at a larger scale (ROI-2) based on block sum of differences.
    """
    return mapping(diff_image, w1*w2)


def g_map(diff_image, w1, w2, w3):
    """
    Secondary/generic mask for least-critical region based on the largest window.
    """
    return mapping(diff_image, w1*w2*w3)


def sad_map_redone(SADmap, Rmap):
    output_image = np.where(Rmap == 0, 0, SADmap)
    return output_image


def rmap_redone(Rmap, Gmap):
    output_image = np.where(Gmap == 0, 0, Rmap)
    return output_image


def get_masks(gray_frame,prev):
    diff_image = abs_diff(gray_frame, prev)
    outputSAD = sad_map(diff_image, para.w1)
    outputR = r_map(outputSAD, para.w1, para.w2)
    outputG = g_map(outputR, para.w1, para.w2, para.w3)
    return outputSAD, outputR, outputG




#Prioritize the ROIS and gives the corresponding based on the masks
# inputs will be the frame and the three mask r and g and the sad 

'''
-The first priority class C1 = ROI-1 
represents the blocks
that are in, and only in the first ROI. Class C1 blocks having
the highest interest are coded with a higher MJPEG
quality factor Q1 before being transmitted ;
- The second priority class C2 = ROI-2  
ROI-1 includes the
labeled moving blocks that are in ROI-2 but not in ROI-1.
Class C2 blocks having a medium interest are coded, prior
to their transmission, with a lower MJPEG quality factor
Q2 < Q1 ;
- The third priority class C3 = GMR 􀀀 ROI-2 includes the
blocks that are in theGMR but are not in ROI-2. This class
blocks are considered to be of low interest and are simply
dropped.


in parameters class
    cls.default_width = width
    cls.default_height = hight

'''
#The roi masks are C1 and C2 and C3 note that the roi C1 takes in consederation only the blocks inside of the GMR or the g_map
def get_compression_masks(sad_map,r_map, g_map):

    #Create the classes masks
    c1 = np.zeros((para.default_height, para.default_width), dtype=np.uint8)
    c2 = np.zeros((para.default_height, para.default_width), dtype=np.uint8)
    c3 = np.zeros((para.default_height, para.default_width), dtype=np.uint8)
    for y in range(0, para.default_height, para.zone_size):
     for x in range(0, para.default_width, para.zone_size):
        # Extract the current block
        g_map_block = g_map[y:y+para.zone_size, x:x+para.zone_size]
        sad_map_block = sad_map[y:y+para.zone_size, x:x+para.zone_size]
        r_map_block = r_map[y:y+para.zone_size, x:x+para.zone_size]
        
        #extract the union between the gmap and the g_map
        if np.any(sad_map_block>0) and np.any(g_map_block>0):
            c1[y:y+para.zone_size, x:x+para.zone_size] = sad_map_block
        elif np.any(r_map_block>0) and np.any(g_map_block>0):
            c2[y:y+para.zone_size, x:x+para.zone_size] = r_map_block
        elif np.any(g_map_block>0):
            c3[y:y+para.zone_size, x:x+para.zone_size] = g_map_block
        
    
    return c1,c2,c3








