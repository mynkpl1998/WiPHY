import struct
import argparse
import itertools
import numpy as np
from WiPHY.utils import __CONSTANTS__
from WiPHY.utils import dec2bin, bit2dec, bit2str, str2dec
from WiPHY.utils import search_sequence_cv2, crc_remainder, readYaml

parser = argparse.ArgumentParser(description="Script to radio configuration header for tx.")
parser.add_argument("-o", "--out-path", type=str, required=True, help="Out file path.")
parser.add_argument("-c", "--radio-config", type=str, required=True, help="Radio Configuration file.")

def get_crc_data(data_bits, crc_polynomial):
    """Returns CRC lookup table for 3bit CRC.
       
       Inputs
       ------
       * data_bits (int):                            Payload length.
       * crc_polynomial (int):                       CRC polynomial to use for encoding.

       Returns
       * dict                                        Map containing payload and its checksum.
    """
    poly = dec2bin(crc_polynomial, 4)
    all_data_bits = itertools.product([0,1], repeat=data_bits)
    gen_lookup = {}
    for data in all_data_bits:
        data_dec = bit2dec(data)
        data_str = bit2str(data)
        out = str2dec(crc_remainder(data_str, poly, '0'))
        gen_lookup[data_dec] = out
    return gen_lookup

def generate_radio_config_header(radio_config, out_path):
    """Generates the radio configuration header file for the TX.

    Inputs
    ------
    * radio_config (dict):                             Radio configuration data.
    * out_path (str):                                  Path to generate the header.
    """
    header_str = "/*\n  \t\tThis header is automatically generated using crc_lookup_generator_script.py.\n"
    header_str += "  \t\tPlease do not modify it manually, unless you know what you are doing.\n"
    header_str += "*/\n\n"

    symbol_dur = float(radio_config['sdr_settings']['baseband_symbol_dur'])
    data_pin = int(radio_config['tx_settings']['data_pin'])
    premable_len = __CONSTANTS__['FRAME_PREAMBLE_BITS']
    premable = int(radio_config['frame_detector']['barker_seq'])
    seq_id_len = __CONSTANTS__['FRAME_SEQ_ID_BITS']
    payload_len = __CONSTANTS__['FRAME_PAYLOAD_BITS']
    checksum_len = __CONSTANTS__['FRAME_CHECKSUM_BITS']
    checksum_poly = int(radio_config['frame_detector']['crc_polynomial'])
    enable_tx_debugging = int(radio_config['tx_settings']['enbale_tx_debug_prints'])

    header_str += "#ifndef RADIO_CONFIG_H\n#define RADIO_CONFIG_H\n\n"
    header_str += "#define BASEBAND_SYM_DUR " + str(symbol_dur) + "\n"
    header_str += "#define DATA_PIN " + str(data_pin) + "\n"
    header_str += "#define PREAMBLE_LEN " + str(premable_len) + "\n"
    header_str += "#define PREAMBLE " + str(premable) + "\n"
    header_str += "#define SEQ_ID_LEN " + str(seq_id_len) + "\n"
    header_str += "#define PAYLOAD_LEN " + str(payload_len) + "\n"
    header_str += "#define CHECKSUM_LEN " + str(checksum_len) + "\n"
    header_str += "#define CHECKSUM_POLY " + str(checksum_poly) + "\n"
    header_str += "#define ENABLE_TX_DEBUG_PRINTS " + str(enable_tx_debugging) + "\n"
    
    header_str += "\n// 3-bits CRC lookup table for 6-bits payload values. \n"

    # Start generating look up table
    crc_lookup_table = get_crc_data(payload_len, checksum_poly)
    header_str += "\nstatic uint8_t CRC3_XOR6_LOOKUP[64] = { "
    for payload in crc_lookup_table:
        header_str += str(crc_lookup_table[payload]) + ", "
    header_str += "};"

    header_str += "\n\n#endif"
    fileHandler = open(out_path + "radio_config.h", "w")
    fileHandler.write(header_str)
    fileHandler.close()

def generate_header(data, path):
    """Generates the lookup header file for 3bit CRC.

    Inputs
    ------
    * data (list):                          CRC bits for all possible payload values.
    * path (str):                           Absolute path to generate the header file.
    """
    header_code = "/*\n  \t\tThis header is automatically generated using crc_lookup_generator_script.py.\n"
    header_code += "  \t\tPlease do not modify it manually, unless you know what you are doing.\n"
    header_code += "*/\n"

    header_code += "\nuint8_t CRC3_XOR6_LOOKUP[64] = {\n"

    for idx in data:
        header_code += str(data[idx]) + ", \n"

    header_code += "};\n"

    fileHandler = open(path + "CRC3_XOR6.h", "w")
    fileHandler.write(header_code)
    fileHandler.close()

if __name__ == "__main__":

    # Parse command line arguments.
    args = parser.parse_args()

    # Radio configuration data
    radio_config_dict = readYaml(args.radio_config)

    '''
    # CRC configuration
    data_bits = 6
    poly = dec2bin(args.crc_polynomial, 4)
    all_data_bits = itertools.product([0,1], repeat=data_bits)
    
    gen_lookup = {}
    for data in all_data_bits:
        data_dec = bit2dec(data)
        data_str = bit2str(data)
        out = str2dec(crc_remainder(data_str, poly, '0'))
        gen_lookup[data_dec] = out
        
    generate_header(gen_lookup, args.out_path)
    '''
    generate_radio_config_header(radio_config_dict, args.out_path)