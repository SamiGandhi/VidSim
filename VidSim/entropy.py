import numpy as np
import math


def entropy_coder(layer, layer_nb, entropy_encoder):

    if entropy_encoder == "EG":
        eg_encoded = encode_eg(layer)
        entropy_cycles = sum([get_group_id(val) * 3 + 7 if val != 0 else 1 for val in layer])
        return eg_encoded, entropy_cycles

    elif entropy_encoder == "RLE_VLI_EG":
        if layer_nb == 0:
            symb1, symb2 = "", ""
            amplitude = get_amplitude_size(layer[0])
            symb1 = bin(amplitude)[2:].zfill(4)
            symb2 = bin(bias_encoder(layer[0], amplitude))[2:].zfill(amplitude)
            rle_vli_eg_encoded = symb1 + symb2
            rle_vli_eg_encoded += encode_rle_eg(layer[1:])
        else:
            rle_vli_eg_encoded = encode_rle_eg(layer)

        return rle_vli_eg_encoded, 0  # Update this value if needed

    elif entropy_encoder == "RLE_EG":
        rle_eg_encoded = encode_rle_eg(layer)
        return rle_eg_encoded, 0  # Update this value if needed

    elif entropy_encoder == "VLI_EG":
        trunc_block = truncate_linear_block(layer)
        vli_eg_encoded = encode_vli_eg(trunc_block)
        return vli_eg_encoded, 0  # Update this value if needed

    elif entropy_encoder == "HUFFMAN":
        trunc_block = truncate_linear_block(layer)
        vli_huffman_encoded = encode_vli_eg(trunc_block)  
        return vli_huffman_encoded, 0  # Update this value if needed

    else:
        raise ValueError("Not recognized entropy coder: " + entropy_encoder)



def getGroupID(ne):
    groupID = 2
    while ne > ((1 << int(groupID)) - 2):
        groupID += 1
    return int(groupID - 1)

def dec2bin(dec, bits):
    bin_str = bin(abs(dec))[2:].zfill(bits)
    return bin_str if dec >= 0 else '-' + bin_str

def getAmplitudeSize(amplitude):
    if amplitude == 0:
        return 0

    if amplitude < 0:
        amplitude = abs(amplitude)

    return int(math.log2(amplitude) + 1)

def biasEncoder(amplitude):
    if amplitude < 0:
        size = getAmplitudeSize(amplitude)
        return amplitude + (2 ** size - 1)
    else:
        return amplitude

def encode_eg(value):
        eg_encoded = ""
        for val in value:
            if val >= 0:
                val = 2 * val
            else:
                val = -2 * val - 1
            if val == 0:
                eg_encoded += "0"
            else:
                group_id = get_group_id(val)
                eg_encoded += "1" * (group_id * 3 + 7)
        return eg_encoded

def encode_rle_eg(value):
    rle_eg_encoded = ""
    rle_data = []
    zero_count = 0

    for val in value:
        if val == 0:
            zero_count += 1
        else:
            rle_data.append(val)
            rle_data.append(zero_count)
            zero_count = 0

    rle_eg_encoded += encode_eg(rle_data)

    return rle_eg_encoded

def encode_vli_eg(value):
    vli_eg_encoded = ""
    if layer_nb == 0:
        amplitude = get_amplitude_size(value[0])
        symb1 = bin(amplitude)[2:].zfill(4)
        symb2 = bin(bias_encoder(value[0], amplitude))[2:].zfill(amplitude)
        vli_eg_encoded += symb1 + symb2
        vli_eg_encoded += encode_rle_eg(value[1:])

    else:
        vli_eg_encoded += encode_rle_eg(value)

    return vli_eg_encoded

def get_group_id(val):
    return int(np.log2(val))

def get_amplitude_size(val):
    return len(bin(abs(val))) - 2

def bias_encoder(val, amplitude):
    return val + (1 << (amplitude - 1)) - 1

def truncate_linear_block(value):
    return value  # Implement your truncation logic here


def blockEntropyCoder(entropy, linearBlock, blockNb, prevBlockNb):
    valueToEncode = blockNb - prevBlockNb

    amplitude_size = getAmplitudeSize(valueToEncode)
    blockNbCode = dec2bin(amplitude_size, 4) + dec2bin(biasEncoder(valueToEncode), amplitude_size)
    encoded,entropyCycles = entropy_coder(linearBlock, 5, entropy)
    layerData = blockNbCode + encoded

    return len(layerData),entropyCycles




