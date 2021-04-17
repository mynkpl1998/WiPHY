import time
import struct
import binascii
from serial import Serial
from utils import Frame

class ASK_Tx():
    """
    An interface to simplify sending data using 
    ASK modules.
    """

    def __init__(self, 
                 comm_port):
        self.__arduino_comm_handler = Serial(comm_port,
                                             115200,
                                             timeout=1)
        
    
    def transmit(self, frame):
        
        num_retries = 5
        try_count = 0
        success = False
        data = frame.get_frame_bytes()
        
        for count in range(0, num_retries):
            if success != True:  
                self.__arduino_comm_handler.write(struct.pack('>BBB', 82, data[0], data[1]))
                time.sleep(0.001)
                ack = self.__arduino_comm_handler.read(size=3)
                recv_frame = ack[1:].hex('-', 1).split('-')
                recv_frame = [ chr(int(b, 16)) for b in recv_frame]
                
                if chr(ack[0]) == 'A':
                    # Check whether data recieved by the Arduino 
                    # matches the transmitted data.
                    frame_str = recv_frame[0] + recv_frame[1]
                    if frame_str != frame.get_frame_byte_string():
                        success = False, count
                    else:
                        success = True
                        return success, count
                else:
                    success = False, count
        
        return success, count


if __name__ == "__main__":
    obj= ASK_Tx('/dev/ttyACM0')

    seq_id = 1
    payload = 32
    crc_polynomial = 13
    checksum = 0
    premable = 29

    f = Frame(preamble=premable,
              seq_id=seq_id,
              payload=payload,
              crc_polynomial=crc_polynomial,
              checksum=checksum)

    while True:
        status, count = obj.transmit(f)
        assert status
        time.sleep(0.001)