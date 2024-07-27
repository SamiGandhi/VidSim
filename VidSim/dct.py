import numpy as np
import cv2

def DCT(block, DCT, zoneSize):
    if DCT == "CLA":
        dctBlock = cv2.dct(np.float32(block))
    
    if DCT == "tLLM":
        dctBlock = tllm2DCT(block, zoneSize)
    
    if DCT == "tBIN":
        dctBlock = tbin2DCT(block, zoneSize)
    
    if DCT == "sLLM":
        dctBlock = sllm2DCT(block, zoneSize)
    
    if DCT == "sBIN":
        dctBlock = sbin2DCT(block, zoneSize)
    
    return dctBlock


def iDCT(dctBlock, DCT, zoneSize):
    if DCT == "CLA":
        return cv2.idct(dctBlock)

    if DCT[1:] == "LLM":
        return llm2iDCT(dctBlock)

    if DCT[1:] == "BIN":
        return bin2iDCT(dctBlock)

def tllm2DCT(block, zoneSize):
    llmBlock = llm2DCT(block)
    triangBlock = np.zeros((8, 8), dtype=np.float32)
    
    for i in range(zoneSize):
        for j in range(zoneSize - i):
            triangBlock[i, j] = llmBlock[i, j]

    return triangBlock

def tbin2DCT(block, zoneSize):
    llmBlock = bin2DCT(block)
    triangBlock = np.zeros((8, 8), dtype=np.int16)
    
    for i in range(zoneSize):
        for j in range(zoneSize - i):
            triangBlock[i, j] = llmBlock[i, j]

    return triangBlock

def sllm2DCT(block, zoneSize):
    llmBlock = llm2DCT(block)
    sqBlock = np.copy(llmBlock)
    
    return sqBlock

def sbin2DCT(block, zoneSize):
    binBlock = bin2DCT(block)
    sqBlock = np.copy(binBlock)
    
    return sqBlock

def llm2DCT(block):
    tempBlock = np.zeros((8, 8), dtype=np.float32)
    temp2Block = np.zeros((8, 8), dtype=np.float32)
    tempRow = np.zeros((1, 8), dtype=np.float32)
    
    for j in range(block.shape[0]):
        tempRow = llm1DCT(block[j, :])
        tempBlock[j, :] = tempRow

    tempBlock = np.transpose(tempBlock)

    for j in range(tempBlock.shape[0]):
        tempRow = llm1DCT(tempBlock[j, :])
        temp2Block[j, :] = tempRow

    temp2Block = np.transpose(temp2Block)
    
    return temp2Block

def bin2DCT(block):
    tempBlock = np.zeros((8, 8), dtype=np.int16)
    temp2Block = np.zeros((8, 8), dtype=np.int16)
    tempRow = np.zeros((1, 8), dtype=np.int16)
    
    for j in range(block.shape[0]):
        tempRow = bin1DCT(block[j, :])
        tempBlock[j, :] = tempRow

    tempBlock = np.transpose(tempBlock)

    for j in range(tempBlock.shape[0]):
        tempRow = bin1DCT(tempBlock[j, :])
        temp2Block[j, :] = tempRow

    temp2Block = np.transpose(temp2Block)
    
    return temp2Block


def llm2iDCT(block):
    tempBlock = np.zeros((8, 8), dtype=np.float32)
    temp2Block = np.zeros((8, 8), dtype=np.float32)
    tempRow = np.zeros((1, 8), dtype=np.float32)

    for j in range(block.shape[0]):
        tempRow = llm1iDCT(block[j, :])
        tempBlock[j, :] = tempRow

    tempBlock = np.transpose(tempBlock)

    for j in range(tempBlock.shape[0]):
        tempRow = llm1iDCT(tempBlock[j, :])
        temp2Block[j, :] = tempRow

    temp2Block = np.transpose(temp2Block)

    return temp2Block / 64.0

def bin2iDCT(block):
    tempBlock = np.zeros((8, 8), dtype=np.int16)
    temp2Block = np.zeros((8, 8), dtype=np.int16)
    tempRow = np.zeros((1, 8), dtype=np.int16)

    for j in range(block.shape[0]):
        tempRow = bin1iDCT(block[j, :])
        tempBlock[j, :] = tempRow

    tempBlock = np.transpose(tempBlock)

    for j in range(tempBlock.shape[0]):
        tempRow = bin1iDCT(tempBlock[j, :])
        temp2Block[j, :] = tempRow

    temp2Block = np.transpose(temp2Block)

    return temp2Block / 16


def bin1DCT(block):
    dct1D = np.zeros((1, 8), dtype=np.int16)

    tmp0 = block[0, 0] + block[0, 7]
    tmp1 = block[0, 0] - block[0, 7]
    tmp2 = block[0, 1] + block[0, 6]
    tmp3 = block[0, 1] - block[0, 6]
    tmp4 = block[0, 2] + block[0, 5]
    tmp5 = block[0, 2] - block[0, 5]
    tmp6 = block[0, 3] + block[0, 4]
    tmp7 = block[0, 3] - block[0, 4]

    tmp = (tmp5 << 2) - tmp5
    tmp8 = tmp3 + (tmp >> 3)
    tmp = (tmp8 << 2) + tmp8
    tmp9 = (tmp >> 3) - tmp5

    tmp10 = tmp0 + tmp6
    tmp11 = tmp0 - tmp6
    tmp12 = tmp7 + tmp9
    tmp13 = tmp7 - tmp9
    tmp14 = tmp1 - tmp8
    tmp15 = tmp1 + tmp8
    tmp16 = tmp2 + tmp4
    tmp17 = tmp2 - tmp4

    tmp = (tmp15 >> 3)
    tmp18 = tmp12 - tmp
    tmp19 = tmp10 + tmp16
    tmp = (tmp19 >> 1)
    tmp20 = tmp - tmp16
    tmp = (tmp11 << 2) - tmp11
    tmp21 = tmp17 - (tmp >> 3)
    tmp = (tmp21 << 2) - tmp21
    tmp22 = tmp11 + (tmp >> 3)
    tmp = (tmp14 << 3) - tmp14
    tmp23 = tmp13 + (tmp >> 3)
    tmp = (tmp23 >> 1)
    tmp24 = tmp14 - tmp

    dct1D[0, 0] = tmp0 + tmp6 + tmp16
    dct1D[0, 1] = tmp1 + tmp8
    dct1D[0, 2] = tmp22
    dct1D[0, 3] = tmp24
    dct1D[0, 4] = tmp20
    dct1D[0, 5] = tmp23
    dct1D[0, 6] = tmp21
    dct1D[0, 7] = tmp18

    return dct1D

def bin1iDCT(dct1D):
    block = np.zeros((1, 8), dtype=np.int16)

    tmp0 = (dct1D[0, 0] >> 1) - dct1D[0, 4]
    tmp1 = dct1D[0, 0] - tmp0
    tmp = (dct1D[0, 6] << 2) - dct1D[0, 6]
    tmp2 = dct1D[0, 2] - (tmp >> 3)
    tmp = (tmp2 << 2) - tmp2
    tmp3 = dct1D[0, 6] + (tmp >> 3)
    tmp = dct1D[0, 5] >> 1
    tmp4 = dct1D[0, 3] + tmp
    tmp = (tmp4 << 3) - tmp4
    tmp5 = dct1D[0, 5] - (tmp >> 3)
    tmp = dct1D[0, 1] >> 3
    tmp6 = dct1D[0, 7] + tmp
    tmp7 = tmp0 + tmp3
    tmp8 = tmp0 - tmp3
    tmp9 = tmp1 + tmp2
    tmp10 = tmp1 - tmp2
    tmp11 = tmp5 + tmp6
    tmp12 = tmp6 - tmp5
    tmp13 = dct1D[0, 1] - tmp4
    tmp14 = dct1D[0, 1] + tmp4
    tmp = (tmp13 << 2) + tmp13
    tmp15 = (tmp >> 3) - tmp12
    tmp = (tmp15 << 2) - tmp15
    tmp16 = tmp13 - (tmp >> 3)
    tmp17 = tmp10 + tmp11
    tmp18 = tmp10 - tmp11
    tmp19 = tmp8 + tmp15
    tmp20 = tmp8 - tmp15
    tmp21 = tmp7 + tmp16
    tmp22 = tmp7 - tmp16
    tmp23 = tmp9 + tmp14
    tmp24 = tmp9 - tmp14

    block[0, 0] = tmp23
    block[0, 1] = tmp21
    block[0, 2] = tmp19
    block[0, 3] = tmp17
    block[0, 4] = tmp18
    block[0, 5] = tmp20
    block[0, 6] = tmp22
    block[0, 7] = tmp24

    return block


def llm1iDCT(dct1D):
    block = np.zeros((1, 8), dtype=np.float32)

    a0, a1, a2, a3, b0, b1, b2, b3 = 0, 0, 0, 0, 0, 0, 0, 0
    z0, z1, z2, z3, z4 = 0, 0, 0, 0, 0
    r = [1.414214, 1.387040, 1.306563, 1.175876, 1.000000, 0.785695, 0.541196, 0.275899]

    z0 = dct1D[0, 1] + dct1D[0, 7]
    z1 = dct1D[0, 3] + dct1D[0, 5]
    z2 = dct1D[0, 3] + dct1D[0, 7]
    z3 = dct1D[0, 1] + dct1D[0, 5]
    z4 = (z0 + z1) * r[3]

    z0 = z0 * (-r[3] + r[7])
    z1 = z1 * (-r[3] - r[1])
    z2 = z2 * (-r[3] - r[5]) + z4
    z3 = z3 * (-r[3] + r[5]) + z4

    b3 = dct1D[0, 7] * (-r[1] + r[3] + r[5] - r[7]) + z0 + z2
    b2 = dct1D[0, 5] * (r[1] + r[3] - r[5] + r[7]) + z1 + z3
    b1 = dct1D[0, 3] * (r[1] + r[3] + r[5] - r[7]) + z1 + z2
    b0 = dct1D[0, 1] * (r[1] + r[3] - r[5] - r[7]) + z0 + z3

    z4 = (dct1D[0, 2] + dct1D[0, 6]) * r[6]
    z0 = dct1D[0, 0] + dct1D[0, 4]
    z1 = dct1D[0, 0] - dct1D[0, 4]
    z2 = z4 - dct1D[0, 6] * (r[2] + r[6])
    z3 = z4 + dct1D[0, 2] * (r[2] - r[6])
    a0 = z0 + z3
    a3 = z0 - z3
    a1 = z1 + z2
    a2 = z1 - z2

    block[0, 0] = a0 + b0
    block[0, 7] = a0 - b0
    block[0, 1] = a1 + b1
    block[0, 6] = a1 - b1
    block[0, 2] = a2 + b2
    block[0, 5] = a2 - b2
    block[0, 3] = a3 + b3
    block[0, 4] = a3 - b3

    return block


def llm1DCT(block):
    dct1D = np.zeros((1, 8), dtype=np.float32)
    r = [1.414214, 1.387040, 1.306563, 1.175876, 1.000000, 0.785695, 0.541196, 0.275899]
    invsqrt2 = 0.707107

    c1 = block[0, 0]
    c2 = block[0, 7]
    t0 = c1 + c2
    t7 = c1 - c2

    c1 = block[0, 1]
    c2 = block[0, 6]
    t1 = c1 + c2
    t6 = c1 - c2

    c1 = block[0, 2]
    c2 = block[0, 5]
    t2 = c1 + c2
    t5 = c1 - c2

    c1 = block[0, 3]
    c2 = block[0, 4]
    t3 = c1 + c2
    t4 = c1 - c2

    c0 = t0 + t3
    c3 = t0 - t3
    c1 = t1 + t2
    c2 = t1 - t2

    dct1D[0, 0] = c0 + c1
    dct1D[0, 4] = c0 - c1
    dct1D[0, 2] = c2 * r[6] + c3 * r[2]
    dct1D[0, 6] = c3 * r[6] - c2 * r[2]

    c3 = t4 * r[3] + t7 * r[5]
    c0 = t7 * r[3] - t4 * r[5]
    c2 = t5 * r[1] + t6 * r[7]
    c1 = t6 * r[1] - t5 * r[7]

    dct1D[0, 5] = c3 - c1
    dct1D[0, 3] = c0 - c2
    c0 = (c0 + c2) * invsqrt2
    c3 = (c3 + c1) * invsqrt2
    dct1D[0, 1] = c0 + c3
    dct1D[0, 7] = c0 - c3

    return dct1D
