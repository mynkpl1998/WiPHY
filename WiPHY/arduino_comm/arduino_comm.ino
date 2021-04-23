#define FRAME_BUFFER_LEN 16
#define SYMBOL_DUR_MS 1

const int dataPin = 12;
char serial_out_buffer[3];
char serial_buffer[3];

void setup() {
   // initialize serial communication
   Serial.begin(115200);
   // initialize the data pin
   pinMode(dataPin, OUTPUT);
   // Put the tx in low mode
   digitalWrite(dataPin, LOW);
}

void transmit_frame()
{
    boolean state = LOW;
    uint8_t second_idx = 0;
    
    for(int idx=0; idx<FRAME_BUFFER_LEN; idx++)
    {
       if(idx > 7)
       {
          state = bitRead(serial_out_buffer[2], 7-second_idx) == 0 ? LOW:HIGH;
          second_idx += 1;
       }
       else
          state = bitRead(serial_out_buffer[1], 7-idx) == 0 ? LOW:HIGH;
       digitalWrite(dataPin, state);
       delay(SYMBOL_DUR_MS);
    }
    digitalWrite(dataPin, LOW);
}

void loop()
{
  if(Serial.available() > 0)
  {
    Serial.readBytes(serial_buffer, 3);
    
    // Detect the start of the frame
    if( serial_buffer[0] == 'R' )
    {
       serial_out_buffer[0] = 65;
       serial_out_buffer[1] = serial_buffer[1];
       serial_out_buffer[2] = serial_buffer[2];

       transmit_frame();
       
       // ACK
       Serial.write(serial_out_buffer, 3);
       Serial.flush();
    }
  }
}
