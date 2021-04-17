import cv2
import yaml
import binascii
import numpy as np
from cv2 import matchTemplate as cv2m

__CONSTANTS__ = {
    'FRAME_BITS': 16,
    'FRAME_PREAMBLE_BITS': 5,
    'FRAME_SEQ_ID_BITS': 2,
    'FRAME_PAYLOAD_BITS': 6,
    'FRAME_CHECKSUM_BITS': 3,
    'FRAME_CONTENT_BITS': 11,
    'FRAME_PAYLOAD_AND_SEQ_ID_BITS': 8,
}

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

def readYaml(file_path):
    """Read the yaml file and returns the content
       in the python dict.

       Inputs
       ------
       * file_path (str):                      Yaml file path to read.

       Returns
       -------
       (dict):                                 Returns YAML file contents.
    """
    with open(file_path, "r") as handle:
        configDict = yaml.load(handle, Loader=yaml.FullLoader)
    return configDict

def bit2dec(bit_str):
    """Converts the list of ones and zeros to 
       corresponding base 10 (Decimal) number.

    Inputs
    ------
    * bit_str (list or np.array):        List of zeros and ones.

    Returns
    -------
    * (int):                             Base 10 (Decimal).
    """
    return int(''.join(str(i) for i in bit_str), 2)

def bit2str(bit_str):
    """Converts the list of ones and zeros to 
       string of ones and zeros.

    Inputs
    ------
    * bit_str (list or np.array):        List of zeros and ones.

    Returns
    -------
    * (str):                             Corresponding string of zeros and ones.
    """
    return ''.join(str(i) for i in bit_str)

def str2dec(bit_str):
    """Converts the string of ones and zeros to 
       correspoding Base 10 (Decimal) representation.

    Inputs
    ------
    * bit_str (list or np.array):        String of zeros and ones.

    Returns
    -------
    * (int):                             Corresponding Base 10 (Decimal) Number.
    """
    return int(bit_str, 2)

def dec2bin(dec, width):
    """Converts the Base 10 unsigned integer to 
       corresponding string of ones and zeros (Binary Represenation).

    Inputs
    ------
    * dec (int):                         Unsigned integer. 
    * width (unsigned width):            Width of the binary string.
    
    Returns
    -------
    * (str):                             Corresponding string of zeros and ones.
    """
    return np.binary_repr(dec, width)

def str2list(_str_, dtype):
    """Splits the string content and returns it as
       the list by converting it to required dtype.

       Inputs
       ------
       * _str_ (str):                      String to be converted.
       * dtype (np.dtype):                 Datatype to be applied after split.
                                             Must be a valid datatype.
        
        Returns
        -------
        * (list):                          List with string content.
    """
    str_split = [dtype(char) for char in _str_]
    return str_split


def search_sequence_cv2(arr, seq):
    """ Find sequence in an array using cv2.
    Copied from Source: https://stackoverflow.com/a/36535397

    Inputs
    ------
    * arr (list or np.array):                 The array to search the sequence in.
    * seq (list or np.array):                 The sequence to be searched.

    Returns
    -------
    * (list):                                 Index(s) where the sequence match was found.
                                               Returns empty list if not found any.
    """
    # Run a template match with input sequence as the template across
    # the entire length of the input array and get scores.
    arr = np.array(arr)
    seq = np.array(seq)
    S = cv2m(arr.astype('uint8'),seq.astype('uint8'),cv2.TM_SQDIFF)

    # Now, with floating point array cases, the matching scores might not be 
    # exactly zeros, but would be very small numbers as compared to others.
    # So, for that use a very small to be used to threshold the scorees 
    # against and decide for matches.
    thresh = 1e-5 # Would depend on elements in seq. So, be careful setting this.

    # Find the matching indices
    idx = np.where(S.ravel() < thresh)[0]
    
    if len(idx) > 0:
        return idx
    else:
        return []

class Frame:

    """Implements a 16-bit network Frame.
           
        The Frame is made up of four components and 
        uses 3-bit CRC to validate the integrity of 
        the received data(payload). The following is
        the frame structure.

        * Preamble -       5 Bits
        * Frame Sequence - 2 Bits
        * Payload -        6 Bits
        * Checksum -       3 Bits

        
        Inputs
        ------
        * preamble (int):                     Frame preamble.
        * seq_id (int):                       Frame sequence id.
        * payload (int):                      Frame payload.
        * checksum (int):                     Frame checksum.
        * crc_polynomial (int):               CRC polynomial used for encoding
                                                the payload.

        Attributes
        ----------
        * preamble (int):                     Returns premable.
        * seq_id (int):                       Returns sequence id.
        * payload (int):                      Returns payload.
        * checksum (int):                     Returns checksum.
        * crc_polynomial (str):               Returns CRC polynomial.
        * is_checksum_valid (bool):           Returns the integrity of the 
                                                payload by running CRC check. 
                                                True if passes else False.
    """

    def __init__(self,
                 preamble,
                 seq_id,
                 payload,
                 checksum,
                 crc_polynomial):

        if preamble < 0 or preamble > 31:
            raise ValueError("Expected frame premable to be > 0 and < 32. Got: %d"%(preamble))

        if seq_id < 0 or seq_id > 3:
            raise ValueError("Expected frame sequence id to be > 0 and < 4. Got: %d"%(seq_id))

        if payload < 0 or payload > 63:
            raise ValueError("Expected frame payload to be > 0 and < 64. Got: %d"%(payload))

        if checksum < 0 or checksum > 7:
            raise ValueError("Expected frame checksum to be > 0 and < 8. Got: %d"%(checksum))
        
        if crc_polynomial != int(crc_polynomial):
            raise ValueError("Expected CRC Polynomial to be integer. Got: %d"%(type(crc_polynomial)))

        self.__preamble_len = __CONSTANTS__['FRAME_PREAMBLE_BITS']
        self.__seq_id_len = __CONSTANTS__['FRAME_SEQ_ID_BITS']
        self.__payload_len = __CONSTANTS__['FRAME_PAYLOAD_BITS'] 
        self.__checksum_len = __CONSTANTS__['FRAME_CHECKSUM_BITS']

        self.__preamble = preamble
        self.__seq_id = seq_id
        self.__payload = payload
        self.__checksum = checksum
        self.__crc_polynomial = dec2bin(crc_polynomial, self.__checksum_len + 1)
        self.__is_checksum_valid = self.__validate_checksum()
    
    @property
    def payload(self,):
        """Returns the frame payload.
        """
        return self.__payload
    
    @property
    def seq_id(self,):
        """Returns the frame sequence id.
        """
        return self.__seq_id
    
    @property
    def preamble(self, ):
        """Returns the frame preamble.
        """
        return self.__preamble
    
    @property
    def checksum(self, ):
        """Returns the frame checksum.
        """
        return self.__checksum
    
    @property
    def crc_polynomial(self, ):
        """Returns the CRC polynomial used to 
           check the frame integrity.
        """
        return self.__crc_polynomial
    
    @property
    def is_checksum_valid(self):
        """Returns whether checksum failed or
        passed.
        """
        return self.__is_checksum_valid
    
    def __repr__(self):
        return "<Frame, at 0x%x>: "%(id(self)) + self.get_frame_str()

    def __validate_checksum(self):
        """Checks the integrity of the payload.
        """
        payload_mask = 0xFF
        payload_mask = payload_mask>>(8-self.__payload_len)
        merged_data = (self.seq_id<<6) | (self.payload & payload_mask)
        return crc_check(dec2bin(merged_data, self.__payload_len + self.__seq_id_len),
                         self.crc_polynomial,
                         dec2bin(self.checksum, self.__checksum_len))
    
    def get_frame_bytes(self):
        """Returns the frame content as sequence of bytes.
        """
        arr = dec2bin(self.__preamble, self.__preamble_len)
        arr += dec2bin(self.__seq_id, self.__seq_id_len)
        arr += dec2bin(self.__payload, self.__payload_len)
        arr += dec2bin(self.__checksum, self.__checksum_len)
        assert len(arr) == self.__preamble_len + self.__seq_id_len + self.__payload_len + self.__checksum_len
        frame_seq = [int(arr[0:8],2), int(arr[8:16], 2)]
        return frame_seq
        
    def get_frame_byte_string(self):
        """Converts the frame in sequence of bits array and
           returns the corresponding char array.
        """
        arr = dec2bin(self.__preamble, self.__preamble_len)
        arr += dec2bin(self.__seq_id, self.__seq_id_len)
        arr += dec2bin(self.__payload, self.__payload_len)
        arr += dec2bin(self.__checksum, self.__checksum_len)
        assert len(arr) == self.__preamble_len + self.__seq_id_len + self.__payload_len + self.__checksum_len
        frame_str = chr(int(arr[0:8],2)) + chr(int(arr[8:16], 2))
        return frame_str

    def get_frame_str(self):
        """Returns the frame structre in string format.
        """
        return "Preamble: %d Frame Seq: %d, Payload, %d, Checksum: %d, Integrity: %s."%(self.preamble, 
                                                                                        self.seq_id,
                                                                                        self.payload,
                                                                                        self.checksum,
                                                                                        "PASS" if self.is_checksum_valid else "FAIL")

class lowPassFilter:

    def __init__(self, sample_rate, cutoff, order):
        self.__sample_rate = sample_rate
        self.__cutoff = cutoff
        self.__order = order
    
        # Nyquist frequency
        self.__nyquist_freq = 0.5 * sample_rate
        normal_cutoff = cutoff/self.__nyquist_freq
        self.__b, self.__a = butter(order, normal_cutoff, btype='low', analog=False)
    
    def apply(self, data):
        return lfilter(self.__b, self.__a, data)

class Muller:

    """Time synchronization block. 
       Implements Mueller and Muller clock recovery technique.

       Inputs
       ------
       * sps (float):                           Expected samples per symbol.
       * alpha (float):                         Update/Learning rate.

       Attributes
       ----------
       * mu (float):                            Current sampling offset.
       * sps (float):                           Current samples per symbol.
       * alpha (float):                         Learning rate.
    """
    def __init__(self,
                 sps,
                 alpha):
        
        if sps != float(sps):
            raise TypeError("Expected sps to be of type float. Got: %s"%(type(sps)))

        if alpha != float(alpha):
            raise TypeError("Expected alpha to be of type float. Got: %s"%(type(alpha)))
        
        self.__mu = float(0)
        self.__sps = float(sps)
        self.__alpha = float(alpha)

        if self.__alpha < 0 or self.__alpha > 1:
            raise ValueError("Expected alpha to be within [0-1]. Got: %.2f"%(self.__alpha))
        
        if self.__sps < 0 :
            raise ValueError("Expected sps > 0. Got %.2f"%(self.__sps))
    
    @property
    def mu(self, ):
        """Returns current sampling offset.
        """
        return self.__mu
    
    @property
    def sps(self):
        """Returns current estimate of samples per symbol.
        """
        return self.__sps
    
    @property
    def alpha(self):
        """Returns update/learning rate.
        """
        return self.__alpha
    
    def __repr__(self):
        return "<Muller, at 0x%x>: mu: %.2f, sps: %.2f, alpha: %.2f."%(id(self),
                                                                      self.mu,
                                                                      self.sps,
                                                                      self.alpha)

    def sync(self, samples):
        """Performs the symbol synchronization on the input samples
            and returns the downsampled signal.

        Inputs
        ------
        * samples (np.array of type np.complex):  Samples to synchronize.

        Returns
        -------
        * (np.array of type np.complex):          Synchronized samples.
        """

        if samples.dtype != np.complex_:
            raise TypeError("Expected samples of dtype np.complex. Got: %s"%(samples.dtype))

        out = np.zeros(samples.shape[0] + 10, dtype=np.complex_)
        out_rail = np.zeros(samples.shape[0] + 10, dtype=np.complex_)
        i_in = 0
        i_out = 2
        while i_out < samples.shape[0] and i_in < samples.shape[0]:
            out[i_out] = samples[i_in + int(self.__mu)]
            out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
            x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
            y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
            mm_val = np.real(y - x)
            self.__mu += self.__sps + self.__alpha*mm_val
            i_in += int(np.floor(self.__mu))
            self.__mu = self.__mu - np.floor(self.__mu)
            i_out += 1
        out = out[2:i_out]
        return out


class ASK_Demodulator:
    """
     Amplitude Shift Keying Demodulator block.
     Performs ASK demodulation on the input signals
     based upon the decision thershold.

     Inputs
     ------
     * decision_thershold (float):              ASK decision thershold. Default: 0.5

     Attributes
     ----------
     * decision_thershold (float):              Returns ASK decision thershold.
    """
    def __init__(self, decision_thershold=0.5):
        
        if decision_thershold != float(decision_thershold):
            raise TypeError("Expected decision thershold to be of type float. Got: %s"%(type(decision_thershold)))

        self.__thershold = float(decision_thershold)
    
    @property
    def decision_thershold(self):
        """Returns ASK decision thershold.
        """
        return self.__thershold
    
    def __repr__(self):
        return "<ASK Demodulator, at 0x%x>: decision thershold: %.2f."%(id(self),
                                                                        self.decision_thershold)
    
    def demodulate(self, samples):
        """Performs ASK thershold based demodulation 
           and returns the demodulated signal.
        
        Inputs
        ------
        * samples (np.array or list):                   Signal to be demodulated.

        Returns
        -------
        * (np.array or list):                           ASK demodulated signal.
        """
        samples = np.array(samples)
        baseband_sig = (samples > self.decision_thershold).astype(int)
        return baseband_sig


class FrameDetector:
    """Implements a block to detect 16-bit network frames
       from sample captures. 

       Inputs
       ------
       * barker_seq (int):                  Preamble/Barker sequence, to detect start of the frame.
       * crc_polynomial (int):              CRC polynomial used to encode the payload in the frame.
                                              Used to check the integrity of received frame.

       Attributes
       ----------
       * barker_seq (int):                  Returns the barker sequence to which detector is tuned to.
       * crc_polynomial (int):              Returns the CRC polynomial.
    """
    def __init__(self, barker_seq, crc_polynomial):
        
        if barker_seq != int(barker_seq):
            raise TypeError("Expected premable to be type int. Got: %s."%(type(barker_seq)))
        
        if crc_polynomial != int(crc_polynomial):
            raise TypeError("Expected preamble of type int. Got: %s."%(type(crc_polynomial)))
            
        if barker_seq < 0:
            raise ValueError("Expected preamble to be < 0. Got: %d."%(barker_seq))

        self.__barker_seq = str2list(dec2bin(int(barker_seq), __CONSTANTS__['FRAME_PREAMBLE_BITS']), int)
        
        if len(self.__barker_seq) != __CONSTANTS__['FRAME_PREAMBLE_BITS']:
            raise ValueError("Expected preamble of size 5. Got: %d."%(self.__barker_seq.size))

        self.__crc_polynomial = int(crc_polynomial)
    
    def __repr__(self):
        return "<FrameDetector, at 0x%x>: Barker Seq: %d, CRC Polynomial: %d."%(id(self),
                                                                                self.barker_seq,
                                                                                self.crc_polynomial)
    
    @property
    def barker_seq(self):
        """Returns the barker sequence/premable to which currently
           frame detector is tuned to.
        """
        return bit2dec(self.__barker_seq) 

    @property
    def crc_polynomial(self):
        """Returns the CRC polynomial in use to verify the integrity 
           of the detected frames.
        """
        return self.__crc_polynomial
    
    def __extractFrame(self, indexes, samples):
        """Returns the detected frames at given index(s) 
           from the sample captures by wrapping 
           it in class of Frame.

           Inputs
           ------
           * indexes (list):                           The position of the start of frames in 
                                                        sample captures.
           * samples (np.array or list):               Sample captures.

           Returns
           -------
           * (list of type class Frame):               Returns a list of type class Frames.
                                                         If no frames found, returns an empty list.
        """
        frames = []
        for index in indexes:
            frame_data = samples[index:index + __CONSTANTS__['FRAME_CONTENT_BITS']]
            if frame_data.shape[0] < __CONSTANTS__['FRAME_CONTENT_BITS']:
                continue
            
            seq_id = frame_data[0:__CONSTANTS__['FRAME_SEQ_ID_BITS']]
            payload = frame_data[__CONSTANTS__['FRAME_SEQ_ID_BITS']:__CONSTANTS__['FRAME_SEQ_ID_BITS'] + __CONSTANTS__['FRAME_PAYLOAD_BITS']]
            checksum = frame_data[__CONSTANTS__['FRAME_PAYLOAD_AND_SEQ_ID_BITS']:__CONSTANTS__['FRAME_PAYLOAD_AND_SEQ_ID_BITS'] + __CONSTANTS__['FRAME_CHECKSUM_BITS']]
            frames.append(Frame(preamble=self.barker_seq,
                                seq_id=bit2dec(seq_id), 
                                payload=bit2dec(payload),
                                checksum=bit2dec(checksum),
                                crc_polynomial=self.crc_polynomial))
        return frames
      
    def step(self, samples):
        """Detects the frames in the sample captures and
           returns a list of detected frames.
        
        Inputs
        ------
        * samples (np.array of type np.int):                  Samples captures.

        Returns
        * (list with objects of type class Frame):            Returns detected frames.
                                                                Returns empty list, if no frame are found.
        """
        # Detect start for frame in the sample captures.
        """
        Note: 
            Don't use self.barker_seq function here.
            It returns the preamble in integer format.
        """
        samples = np.array(samples, dtype=np.int8)
        detected_frame_idxs = search_sequence_cv2(samples, self.__barker_seq)

        # Offset the starting of the frame based upon the preamble size.
        for idx, value in enumerate(detected_frame_idxs):
            detected_frame_idxs[idx] = value + len(self.__barker_seq)
        
        # Extract frames from the sample captures
        extracted_frames = self.__extractFrame(detected_frame_idxs, samples)
        return extracted_frames