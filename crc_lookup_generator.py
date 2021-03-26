import argparse
import numpy as np
import itertools
import struct
from utils import bit2dec, bit2str, str2dec

def crc_remainder(input_bitstring, polynomial_bitstring, initial_filler):
    """Calculate the CRC remainder of a string of bits using a chosen polynomial.
    initial_filler should be '1' or '0'.
    """
    polynomial_bitstring = polynomial_bitstring.lstrip('0')
    len_input = len(input_bitstring)
    initial_padding = (len(polynomial_bitstring) - 1) * initial_filler
    input_padded_array = list(input_bitstring + initial_padding)
    while '1' in input_padded_array[:len_input]:
        cur_shift = input_padded_array.index('1')
        for i in range(len(polynomial_bitstring)):
            input_padded_array[cur_shift + i] \
            = str(int(polynomial_bitstring[i] != input_padded_array[cur_shift + i]))
    return ''.join(input_padded_array)[len_input:]

def crc_check(input_bitstring, polynomial_bitstring, check_value):
    """Calculate the CRC check of a string of bits using a chosen polynomial."""
    polynomial_bitstring = polynomial_bitstring.lstrip('0')
    len_input = len(input_bitstring)
    initial_padding = check_value
    input_padded_array = list(input_bitstring + initial_padding)
    while '1' in input_padded_array[:len_input]:
        cur_shift = input_padded_array.index('1')
        for i in range(len(polynomial_bitstring)):
            input_padded_array[cur_shift + i] \
            = str(int(polynomial_bitstring[i] != input_padded_array[cur_shift + i]))
    return ('1' not in ''.join(input_padded_array)[len_input:])

def generate_header(data):
    header_code = "/*\n  This header is automatically generated using crc_lookup_generator_script.\n"
    header_code += "  Please do not modify it manually.\n"
    header_code += "*/\n"

    header_code += "\nuint8_t CRC3_XOR6_LOOKUP[64] = {\n"

    for idx in data:
        header_code += str(data[idx]) + ", \n"

    header_code += "};\n"

    fileHandler = open("CRC3_XOR6.h", "w")
    fileHandler.write(header_code)
    fileHandler.close()

if __name__ == "__main__":

    data_bits = 6

    poly = [1, 1, 0, 1]
    all_data_bits = itertools.product([0,1], repeat=data_bits)
    
    gen_lookup = {}
    for data in all_data_bits:
        data_dec = bit2dec(data)
        data_str = bit2str(data)
        out = str2dec(crc_remainder(data_str, bit2str(poly), '0'))
        gen_lookup[data_dec] = out
        
    generate_header(gen_lookup)