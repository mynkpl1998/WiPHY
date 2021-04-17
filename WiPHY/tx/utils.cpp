#include <Arduino.h>
#include "utils.h"
#include "radio_config.h"
#define DEBUG_ENABLED 1

#if ENABLE_TX_DEBUG_PRINTS == DEBUG_ENABLED
    extern Frame tx_buff_frame;
#endif

#define SYMBOL_DUR_MS BASEBAND_SYM_DUR*1000

extern Frame frame;
extern uint8_t tx_buff[TX_BUF_LEN];

bool populate_frame(uint8_t payload, uint8_t seq_id)
{
   frame.preamble = PREAMBLE;
   frame.frame_seq = seq_id;
   frame.payload = payload;
   frame.checksum = get_checksum(frame.payload, frame.frame_seq);
   return true;
}

uint8_t get_mask(uint8_t mask_len)
{
   uint8_t mask = 0xFF;
   return mask>>(8-mask_len);
}

void print_frame_contents()
{
  uint8_t seq_id = get_mask(SEQ_ID_LEN) & frame.frame_seq;
  uint8_t payload = get_mask(PAYLOAD_LEN) & frame.payload;
  uint8_t preamble = get_mask(PREAMBLE_LEN) & frame.preamble;
  uint8_t checksum = get_mask(CHECKSUM_LEN) & frame.checksum;

  Serial.print("Frame Content => ");
  Serial.print("Preamble: ");
  Serial.print(preamble);
  Serial.print(", Frame Seq: ");
  Serial.print(seq_id);
  Serial.print(", Payload: ");
  Serial.print(payload);
  Serial.print(", Checksum: ");
  Serial.print(checksum);
  Serial.println(".");
}

void write_frame_to_tx_buffer()
{
   uint8_t index = 0;
   uint8_t frame_seq_index = 0;
   uint8_t payload_index = 0;
   uint8_t checksum_index = 0;

   for(uint8_t bit_idx=0; bit_idx<8; bit_idx++)
   {
      /* Read preamble into the buffer.*/
      if(bit_idx > (8 - PREAMBLE_LEN - 1))
      {
         tx_buff[index] = bitRead(frame.preamble, 7-bit_idx);
         index += 1;
      }

      /* Read frame sequence into the tx buffer. */
      if(bit_idx > (8 - SEQ_ID_LEN - 1))
      {
         tx_buff[frame_seq_index + PREAMBLE_LEN] = bitRead(frame.frame_seq, 7-bit_idx);
         frame_seq_index += 1;
      }

      /* Read payload into the tx buffer. */
      if(bit_idx > (8 - PAYLOAD_LEN - 1))
      {
         tx_buff[payload_index + PREAMBLE_LEN + SEQ_ID_LEN] = bitRead(frame.payload, 7-bit_idx);
         payload_index += 1;
      }

      /* Read checksum into the tx buffer. */
      if(bit_idx > (8 - CHECKSUM_LEN - 1))
      {
         tx_buff[checksum_index + PREAMBLE_LEN + SEQ_ID_LEN + PAYLOAD_LEN] = bitRead(frame.checksum, 7-bit_idx);
         checksum_index += 1;
      }
   }

   /*
   for(int i=0; i<TX_BUF_LEN; i++)
   {
      Serial.print(tx_buff[i]);
      Serial.print(" ");
   }
   Serial.println(" "); */
}

uint8_t bit2dec(uint8_t start_index, uint8_t end_index, uint8_t offset)
{
   float dec_sum = 0.0;
   uint8_t buffer_index = 0;
   uint8_t base_index = (end_index - start_index);
   
   for(uint8_t index=start_index; index<=end_index; index++)
   {
      dec_sum += (tx_buff[offset + buffer_index] * pow(2, base_index));
      Serial.print(tx_buff[offset + buffer_index]);
      buffer_index += 1;
      base_index -= 1;
   }
   Serial.println("");
   return dec_sum;
}


void print_frame_from_tx_buffer()
{
   uint8_t preamble = (tx_buff[0] * pow(2, 4)) + (tx_buff[1] * pow(2, 3)) + (tx_buff[2] * pow(2, 2)) + (tx_buff[3] * pow(2, 1)) + (tx_buff[4] * pow(2, 0));
   uint8_t payload = (tx_buff[7] * pow(2, 5)) + (tx_buff[8] * pow(2, 4)) + (tx_buff[9] * pow(2, 3)) + (tx_buff[10] * pow(2, 2)) + (tx_buff[11] * pow(2, 1)) + (tx_buff[12] * pow(2, 0));
   uint8_t seq_id = (tx_buff[5] * pow(2, 1)) + (tx_buff[6] * pow(2, 0));
   uint8_t checksum = (tx_buff[13] * pow(2, 2)) + (tx_buff[14] * pow(2, 1)) + (tx_buff[15] * pow(2, 0));
   
   Serial.print("Frame Buffer => ");
   Serial.print("Preamble: ");
   Serial.print(preamble);
   Serial.print(", Frame Seq: ");
   Serial.print(seq_id);
   Serial.print(", Payload: ");
   Serial.print(payload);
   Serial.print(", Checksum: ");
   Serial.print(checksum);
   Serial.println(".");

   /* Update tx frame object contents */
   #if ENABLE_TX_DEBUG_PRINTS == DEBUG_ENABLED
     tx_buff_frame.preamble = preamble;
     tx_buff_frame.payload = payload;
     tx_buff_frame.frame_seq = seq_id;
     tx_buff_frame.checksum = checksum;
   #endif
}

uint8_t get_checksum(uint8_t payload, uint8_t seq_id)
{
  uint8_t masked_payload = (get_mask(PAYLOAD_LEN) & payload);
  uint8_t masked_seq_id = get_mask(SEQ_ID_LEN) & seq_id;
  masked_seq_id = masked_seq_id<<6;
  uint8_t lookup = seq_id | masked_payload;
  return CRC3_XOR6_LOOKUP[lookup];
  
}

void transmit_frame()
{
    boolean state = LOW;
    for(uint8_t idx=0; idx<TX_BUF_LEN; idx++)
    {
      state = (tx_buff[idx] == 0) ? LOW:HIGH;
      digitalWrite(DATA_PIN, state);
      delay(SYMBOL_DUR_MS);
    }
    // Turn of TX once frame is transmitted.
    digitalWrite(DATA_PIN, LOW);
}
