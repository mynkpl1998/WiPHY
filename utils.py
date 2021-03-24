import cv2
import numpy as np
from cv2 import matchTemplate as cv2m

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

        self.__preamble_len = 5
        self.__seq_id_len = 2
        self.__payload_len = 6
        self.__checksum_len = 3

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
        return crc_check(dec2bin(self.payload, self.__payload_len),
                         self.crc_polynomial,
                         dec2bin(self.checksum, self.__checksum_len))

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


class frameDetector:

    def __init__(self, barker_seq):
        self.__barker_seq = barker_seq

    def extractFrame(self, indexes, samples):
        frames = []
        for index in indexes:
            frame_data = samples[index:index+11]
            if frame_data.shape[0] < 11:
                continue
            
            seq_id = frame_data[0:2]
            payload = frame_data[2:2+6]
            crc = frame_data[8:8+3]
            frames.append(Frame(bit2dec(seq_id), bit2dec(payload), bit2dec(crc)))
        return frames
      
    def step(self, samples):
        
        # Detect start for frame in the capture
        detected_frame_idxs = search_sequence_cv2(samples, self.__barker_seq)
        
        # Offset the starting of the frame based upon the preamble size
        for idx, value in enumerate(detected_frame_idxs):
            detected_frame_idxs[idx] = value + len(self.__barker_seq)
        
        # Extract frames from the capture
        extracted_frames = self.extractFrame(detected_frame_idxs, samples)
        
        return extracted_frames