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

def search_sequence_cv2(arr,seq):
    """ Find sequence in an array using cv2.
    """
    # Run a template match with input sequence as the template across
    # the entire length of the input array and get scores.
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

    def __init__(self,
                 seq_id,
                 payload,
                 crc):
        self.seq_id = seq_id
        self.payload = payload
        self.crc = crc

        self.checksum = self.validate_frame()
    
    def validate_frame(self):
        return crc_check(dec2bin(self.payload),
                         '1101',
                         dec2bin(self.crc))

    def get_frame_str(self):
        return "Frame Seq: %d, Payload, %d, CRC: %d, Checksum: %s."%(self.seq_id, 
                                                                    self.payload,
                                                                    self.crc,
                                                                    "PASS" if self.checksum else "FAIL")

def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Source: https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found


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

class muller:

    def __init__(self,
                 sps):
        self.__mu = 0
        self.__sps = sps
    
    def sync(self, samples):

        out = np.zeros(samples.shape[0] + 10, dtype=np.complex)
        out_rail = np.zeros(samples.shape[0] + 10, dtype=np.complex)
        i_in = 0
        i_out = 2
        while i_out < samples.shape[0] and i_in < samples.shape[0]:
            out[i_out] = samples[i_in + int(self.__mu)]
            out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
            x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
            y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
            mm_val = np.real(y - x)
            self.__mu += self.__sps + 0.1*mm_val
            i_in += int(np.floor(self.__mu))
            self.__mu = self.__mu - np.floor(self.__mu)
            i_out += 1
        out = out[2:i_out]

        return out


class ASKdemod:

    def __init__(self, thershold=0.5):
        self.__thershold = thershold
    
    def demod(self, samples):
        baseband_sig = (np.abs(samples) > self.__thershold).astype(int)
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