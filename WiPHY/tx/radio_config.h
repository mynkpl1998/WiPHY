/*
  		This header is automatically generated using crc_lookup_generator_script.py.
  		Please do not modify it manually, unless you know what you are doing.
*/

#ifndef RADIO_CONFIG_H
#define RADIO_CONFIG_H

#define BASEBAND_SYM_DUR 0.001
#define DATA_PIN 12
#define PREAMBLE_LEN 5
#define PREAMBLE 29
#define SEQ_ID_LEN 2
#define PAYLOAD_LEN 6
#define CHECKSUM_LEN 3
#define CHECKSUM_POLY 13
#define ENABLE_TX_DEBUG_PRINTS 0

// 3-bits CRC lookup table for 6-bits payload values. 

static uint8_t CRC3_XOR6_LOOKUP[64] = { 0, 5, 7, 2, 3, 6, 4, 1, 6, 3, 1, 4, 5, 0, 2, 7, 1, 4, 6, 3, 2, 7, 5, 0, 7, 2, 0, 5, 4, 1, 3, 6, 2, 7, 5, 0, 1, 4, 6, 3, 4, 1, 3, 6, 7, 2, 0, 5, 3, 6, 4, 1, 0, 5, 7, 2, 5, 0, 2, 7, 6, 3, 1, 4, };

#endif