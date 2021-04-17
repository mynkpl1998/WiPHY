#ifndef UTILS_H
#define UTILS_H

#include <Arduino.h>

typedef struct {

  /* Stores frame preamble. */
  uint8_t preamble;

  /* Stores frame sequence id */
  uint8_t frame_seq;

  /* Stores frame payload. */
  uint8_t payload;

  /* Stores payload checksum */
  uint8_t checksum;
  
} Frame;

#define TX_BUF_LEN PREAMBLE_LEN + SEQ_ID_LEN + PAYLOAD_LEN + CHECKSUM_LEN

/* Utility functions */
bool populate_frame(uint8_t payload, uint8_t seq_id);
void print_frame_contents();
uint8_t get_mask(uint8_t mask_len);
void transmit_frame();
void write_frame_to_tx_buffer();
void print_frame_from_tx_buffer();
uint8_t bit2dec(uint8_t start_index, uint8_t end_index, uint8_t offset);
uint8_t get_checksum(uint8_t payload, uint8_t seq_id);

#endif
