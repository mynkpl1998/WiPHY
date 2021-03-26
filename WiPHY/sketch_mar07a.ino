#include <TimerOne.h>
#include "CRC3_XOR6.h"
#define DATA_PIN 12
#define SYMBOL_PERIOD_SECS 0.001

#define BARKER_CODE_LEN 5
#define TX_BUF_LEN BARKER_CODE_LEN + 11

typedef struct frame 
{
  /* Length 5 Barker Code to detech frame start */
  uint8_t barker_code;

  /* Frame Sequence */
  uint8_t frame_seq;
  
  /* 6 Data bits */
  uint8_t data;
  
  /* Frame 3-CRC bits */
  uint8_t crc_bits;
  
} frame_t;


/* Global Variables */
const long symbol_duration_ms = SYMBOL_PERIOD_SECS * 1e3;
static uint8_t tx_buff[TX_BUF_LEN];
volatile uint8_t data;

frame_t frame;


void setup() {

  /* Debugging purposes */
  Serial.begin(9600);
  
  /* Tx Initialization */
  pinMode(DATA_PIN, OUTPUT);
  digitalWrite(DATA_PIN, LOW);

  /*
  for(int i=0; i<64; i++)
  {
    Serial.print(CRC3_XOR6_LOOKUP[i]);
    Serial.print(" ");
  }
  Serial.println(' ');
  */
  
  /* Intialize the frame data and memory */
  data = 3;
  frame.barker_code = 29;
  frame.frame_seq = 0;
  frame.data = data;
  frame.crc_bits = 0;
  for(int idx=0; idx<TX_BUF_LEN; idx++)
    tx_buff[idx] = 0;

  /* Allow everything to settle. Hence delay of 10ms */
  delay(10);
}

/* Prepares the frame for tranmission */
void get_frame(uint8_t data, uint8_t seq_id)
{
   frame.barker_code = 29;
   
   frame.frame_seq = seq_id;
   frame.frame_seq = 2;
   
   frame.data = data;
   //frame.data = 42;
   
   frame.crc_bits = CRC3_XOR6_LOOKUP[data];
}


void populate_tx_buff()
{
   int index = 0;
   int frame_seq_index = 0;
   int data_index = 0;
   int crc_index = 0;
   
   for(int bit_idx=0; bit_idx<8; bit_idx++)
   {
      /* Read barker code */
      if(bit_idx > 2)
      {
        tx_buff[index] = bitRead(frame.barker_code, 7-bit_idx);
        index += 1;
      }

      /* Read frame seq */
      if(bit_idx > 5)
      {
        tx_buff[frame_seq_index + 5] = bitRead(frame.frame_seq, 7-bit_idx);
        frame_seq_index += 1;
      }

      /* Read data bits */
      if(bit_idx > 1)
      {
        tx_buff[data_index + 7] = bitRead(frame.data, 7-bit_idx);
        data_index += 1;
      }

      /* Read CRC data */
      if(bit_idx > 4)
      {
        tx_buff[crc_index + 13] = bitRead(frame.crc_bits, 7-bit_idx);
        crc_index += 1;
      }

   }
}


void print_frame_from_buf()
{
  uint8_t bcode = (tx_buff[0] * pow(2, 4)) + (tx_buff[1] * pow(2, 3)) + (tx_buff[2] * pow(2, 2)) + (tx_buff[3] * pow(2, 1)) + (tx_buff[4] * pow(2, 0));
  uint8_t data = (tx_buff[7] * pow(2, 5)) + (tx_buff[8] * pow(2, 4)) + (tx_buff[9] * pow(2, 3)) + (tx_buff[10] * pow(2, 2)) + (tx_buff[11] * pow(2, 1)) + (tx_buff[12] * pow(2, 0));
  uint8_t seq_id = (tx_buff[5] * pow(2, 1)) + (tx_buff[6] * pow(2, 0));
  uint8_t crc = (tx_buff[13] * pow(2, 2)) + (tx_buff[14] * pow(2, 1)) + (tx_buff[15] * pow(2, 0));
  
  Serial.print("Frame Seq: ");
  Serial.print(seq_id);
  Serial.print(", preamble: ");
  Serial.print(bcode);
  Serial.print(", payload: ");
  Serial.print(data);
  Serial.print(", CRC: ");
  Serial.print(crc);
  Serial.println(".");
  
}

/* Transmit frame */

void transmit_frame()
{
   populate_tx_buff();
   /*
   for(int i=0; i<TX_BUF_LEN; i++)
   {
        Serial.print(tx_buff[i]);
        Serial.print(", ");
   }
   
   Serial.println("");
   */
   
   //print_frame_from_buf();
   
   boolean state = LOW;
   for(int idx=0; idx<TX_BUF_LEN; idx++)
   {
      state = (tx_buff[idx] == 0) ? LOW:HIGH;
      digitalWrite(DATA_PIN, state);
      delay(symbol_duration_ms);
   }
}


void print_frame()
{
  uint8_t seq_id = 0x3 & frame.frame_seq;
  uint8_t data = 0x3F & frame.data;
  uint8_t bcode = 0x1F & frame.barker_code;
  uint8_t crc = 0x7 & frame.crc_bits;
  
  Serial.print("Frame Seq: ");
  Serial.print(seq_id);
  Serial.print(", preamble: ");
  Serial.print(bcode);
  Serial.print(", payload: ");
  Serial.print(data);
  Serial.print(", CRC: ");
  Serial.print(crc);
  Serial.println(".");
}

void loop() {
  
  // Get a frame to transmit
  get_frame(data, data);

  // Print frame information
  //print_frame();

  // Transmit frame
  transmit_frame();
  
  data += 1;
  if(data > 63)
    data = 0;

  // Wait for 1 ms before sending out the next frame.
  delay(1);
}
