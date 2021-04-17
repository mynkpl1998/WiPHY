import time
import atexit
import struct
from wasabi import msg
from serial import Serial
from WiPHY.utils import Frame, dec2bin, bit2dec, crc_remainder, __CONSTANTS__

class ASK_Tx():
    """
    An Arduino and Python based transmitter for 
    315-433 MHZ ASK modules.
    The module tranmits the frame using arduino
    via serial commands from python.

    Inputs
    ------
    * comm_port (str):                           Arduino communication port.
    * baud_rate (int):                           Arduino serial communication baud rate.
    * timeout (float):                           Arduino serial communication timeout. Default: 1s.
    * num_retransmit_retries (int):              Number of re-transmission for a frame. Default: 5.
    * barker_seq (int):                          Barker seq to use in frame.
    * crc_polynomial (int):                      CRC Polynomial to use for frame checksum.
    
    Attributes
    ----------
    * comm_port (int):                           Returns Arduino-Python serial communication port.
    * baud_rate (int):                           Returns Arduino-Python serial communication baud rate.
    * timeout (float):                           Returns Arduino-Python serial communication timeout.
    * num_retransmit_tries (int):                Returns number of frame retry count.
    * arduino_comm_handler (Serial Object):      Returns Arduino-Python serial communication object.
    * barker_seq (int):                          Returns barker seq in use by frame detection. Default: 29.
    * crc_polynomial (int):                      Returns CRC polynomial in to generate checksum. Default: 13.
    """

    def __init__(self, 
                 comm_port,
                 baud_rate,
                 timeout=1,
                 num_retransmit_retries=5,
                 barker_seq=29,
                 crc_polynomial=13):
        
        try:
            self.__arduino_comm_handler = Serial(comm_port,
                                                baud_rate,
                                                timeout=timeout)
        except Exception as e:
            raise RuntimeError("Failed to open serial connection with device at %s. Root cause: %s"%(comm_port, 
                                                                                                     e))
        self.__comm_port = str(comm_port)
        self.__baud_rate = int(baud_rate)
        self.__timeout = float(timeout)
        self.__num_retransmit_retries = int(num_retransmit_retries)
        self.__barker_seq = int(barker_seq)
        self.__crc_polynomial = int(crc_polynomial)

        msg.good("Opened the Serial connection between Python and Arduino using comm port %s, Baud rate: %d."%(comm_port, baud_rate))
        
        # meta data
        self.__crc_polynomial_bit_string = dec2bin(self.crc_polynomial, __CONSTANTS__['FRAME_CHECKSUM_BITS'] + 1)

        # Register the clean up function.
        # Will be called if the object is destroyed for any reason.
        atexit.register(self.cleanup)
    
    @property
    def comm_port(self):
        """Returns Arduino-Python serial communication port.
        """
        return self.__comm_port
    
    @property
    def baud_rate(self):
        """Returns Arduino-Python serial communication baud rate.
        """
        return self.__baud_rate
    
    @property
    def timeout(self):
        """Returns Arduino-Python serial communication timeout.
        """
        return self.__timeout
    
    @property
    def num_retransmit_tries(self):
        """Returns number of frame retry count.
        """
        return self.__num_retransmit_retries
    
    @property
    def arduino_comm_handler(self):
        """Returns Arduino-Python serial communication object.
        """
        return self.__arduino_comm_handler
    
    @property
    def crc_polynomial(self):
        """Returns CRC polynomial in use to
           calculate checksum.
        """
        return self.__crc_polynomial
    
    @property
    def barker_seq(self):
        """Returns the barker sequence
           in use by Frame Detector.
        """
        return self.__barker_seq
    
    def send(self, payload, seq_id):
        """Sends the requested data and returns
        whether it was successfully transmitted or not.

        Inputs
        ------
        * payload (int):                          Must be 6-bit integer.
        * seq_id (int):                           2 bit frame sequence.

        Returns
        -------
        * (bool, int):                            Returns whether the frame is transmitted
                                                   successfully and the number of retries.
        """

        if payload != int(payload):
            raise TypeError("Expected payload to be of type int. Got: %s"%(type(payload)))
        
        if seq_id != int(seq_id):
            raise TypeError("Expected seq id to be of type int. Got: %s"%(type(seq_id)))
        
        payload = int(payload)
        seq_id = int(seq_id)

        if payload < 0 or payload > (2**6)-1:
            raise ValueError("Payload must be between [0-63]. Got: %d"%(payload))
        
        frame_bits = dec2bin(seq_id, __CONSTANTS__['FRAME_SEQ_ID_BITS']) + dec2bin(payload, __CONSTANTS__['FRAME_PAYLOAD_BITS'])
        
        checksum = crc_remainder(frame_bits, self.__crc_polynomial_bit_string, '0')
        checksum = bit2dec(checksum)

        frame = Frame(preamble=self.barker_seq,
                      seq_id=seq_id,
                      payload=payload,
                      checksum=checksum,
                      crc_polynomial=self.crc_polynomial)
        assert frame.is_checksum_valid == True
        return self.__transmit(frame)
    
    def cleanup(self):
        self.arduino_comm_handler.close()
        msg.info("Closed the serial connection between Arduino and Python.")
        
    def __transmit(self, frame):
        """Transmits the content of the utils.Frame 
        using ASK modulation scheme.

        Inputs
        ------
        * frame (utils.Frame Object):                   Frame to transmit.
        
        Returns
        -------
        * (bool, int):                                  Returns whether the frame is transmitted
                                                         successfully and the number of count it took
                                                         to transmit the frame successfully.
        """
        success = False
        data = frame.get_frame_bytes()
        
        for count in range(0, self.num_retransmit_tries):
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