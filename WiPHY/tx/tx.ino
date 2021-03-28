#include "radio_config.h"
#include "utils.h"
#include <assert.h>
#define DEBUG_ENABLED 1

#if ENABLE_TX_DEBUG_PRINTS == DEBUG_ENABLED
    Frame tx_buff_frame;
#endif

/* Frame object to hold current frame content. */
Frame frame;
/* TX Buffer */
uint8_t tx_buff[TX_BUF_LEN];
/* Book keeping variables */
uint8_t payload;
uint8_t frame_seq;

void setup() {

  /* 
   Initialize Serial connection 
   to Arduino for debugging purposes.
  */
  Serial.begin(9600);

  /* Setup Arduino pins. */
  pinMode(DATA_PIN, OUTPUT);
  digitalWrite(DATA_PIN, LOW);

  /* Initialize frame content to valid data. */
  frame.preamble = PREAMBLE;
  frame.frame_seq = 0;
  frame.payload = 0;
  frame.checksum = CRC3_XOR6_LOOKUP[frame.payload];

  /* Initialize tx transmit */
  payload = 0;
  frame_seq = 0;

  /* Clear TX buffer content. */
  memset(&tx_buff, 0, sizeof(uint8_t)*TX_BUF_LEN);

  /* Wait for 2ms to initialize. */
  delay(2);
}

void loop() {

   /* Get a frame to transmit */
   populate_frame(payload, frame_seq);

   /* Debug Prints - Prints the frame object content */
   if (ENABLE_TX_DEBUG_PRINTS)
   {
      Serial.print("{\n\t");
      print_frame_contents();
      Serial.print("\t");
   }
   /* Copy the frame object content to tx buffer. */
   write_frame_to_tx_buffer();

   /* Debug Prints - Print the frame from tx buffer. */
   if (ENABLE_TX_DEBUG_PRINTS)
   {
      print_frame_from_tx_buffer();
      Serial.print("}\n");

      /* Validate frame contents. */
      #if ENABLE_TX_DEBUG_PRINTS == DEBUG_ENABLED
        assert(tx_buff_frame.payload == frame.payload);
        assert(tx_buff_frame.checksum == frame.checksum);
        assert(tx_buff_frame.frame_seq == frame.frame_seq);
        assert(tx_buff_frame.preamble == frame.preamble);
      #endif
   }
   
   /* Transmit the frame contents. */
   transmit_frame();

   /* Payload and frame sequence overflow check */
   if(payload >= (pow(2, PAYLOAD_LEN)-1))
      payload = 0;
   else
      payload += 1;

   if(frame_seq >= (pow(2, SEQ_ID_LEN)-1))
      frame_seq = 0;
   else
      frame_seq += 1;

   /* Wait for 1ms before sending out next frame. */
   delay(1);
}
