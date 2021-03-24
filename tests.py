import pytest
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from utils import bit2dec, bit2str, str2dec, dec2bin, search_sequence_cv2, Frame, Muller, ASK_Demodulator

def test_bit2dec():
    test_data = np.random.randint(0, 100, size=20)
    for test_case in test_data:
        out = np.binary_repr(test_case, 8)
        out_in = [int(bit) for bit in out]
        calculated_out = bit2dec(out_in)
        assert calculated_out == test_case

def test_bit2str():
    test_data = [
        [1, 1, 0 , 1],
        [0, 0],
        [1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0]
    ]
    golden = ['1101', '00', '1', '111111100', '000']
    for idx, test_case in enumerate(test_data):
        calculated_out = bit2str(test_case)
        assert calculated_out == golden[idx]
    
def test_str2dec():
    test_data = ['1101', '00', '1', '111111100', '000']
    golden = [13, 0, 1, 508, 0]
    for idx, test_case in enumerate(test_data):
        calculated_out = str2dec(test_case)
        assert calculated_out == golden[idx]

def test_dec2bin():
    golden = ['1101', '0', '1', '111111100', '000']
    test_data = [13, 0, 1, 508, 0]
    width_data = [4, 1, 1, 9, 3]

    for idx, test_case in enumerate(test_data):
        calculated_out = dec2bin(test_case, width_data[idx])
        assert calculated_out == golden[idx]

def test_search_sequence_cv2():
    test_arr = np.concatenate((np.arange(0, 100), np.arange(70, 80)))
    test_seq = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [76, 77]]
    golden_out = [[0], [10], [76, 106] ]
    for idx, seq in enumerate(test_seq):
        out = search_sequence_cv2(test_arr, seq)
        assert len(out) == len(golden_out[idx])
        for local_index, index in enumerate(out):
            assert golden_out[idx][local_index] == index

def test_Frame():
    frame1 = Frame(preamble=29, 
                   seq_id=1,
                   payload=63,
                   checksum=4,
                   crc_polynomial=13)
    assert frame1.preamble == 29
    assert frame1.seq_id == 1
    assert frame1.checksum == 4
    assert frame1.payload == 63
    assert frame1.is_checksum_valid == True
    assert frame1.crc_polynomial == '1101'

    frame1 = Frame(preamble=29, 
                   seq_id=1,
                   payload=63,
                   checksum=2,
                   crc_polynomial=13)
    assert frame1.preamble == 29
    assert frame1.seq_id == 1
    assert frame1.checksum == 2
    assert frame1.payload == 63
    assert frame1.is_checksum_valid == False
    assert frame1.crc_polynomial == '1101'
    #print(frame1)


def test_Muller():
    # Muller update rate.
    alpha = 0.1
    # Signal sample rate.
    sample_rate = 300e3
    # Signal symbol duration.
    symbol_dur = 0.001 
    # Calculate samples per symbol.
    sps = sample_rate * symbol_dur
    num_symbols = 4
    signal_dur = num_symbols * symbol_dur
    time = np.linspace(0, signal_dur, int(sps * num_symbols), endpoint=True)
    sig = np.clip(signal.square(2*np.pi*(1/symbol_dur/2)*time), 0, 1).astype(np.complex_)

    ''' 
    plt.plot(time, np.abs(sig))
    plt.show()
    '''

    m1 = Muller(sps=sps, alpha=alpha)
    assert m1.mu == 0.0
    assert m1.sps == sps
    assert m1.alpha == alpha

    out = m1.sync(sig)
    assert out.size == num_symbols
    assert np.abs(out).sum() == 2
    #print(m1)

def test_ASK_Demodulator():
    demod = ASK_Demodulator(decision_thershold=0.1)
    assert demod.decision_thershold == 0.1

    time = np.linspace(0, 1, 10)
    freq = 2
    ampl = 1.5
    input_signal = ampl * np.sin(2 * np.pi * freq * time)
    
    '''
    plt.plot(time, input_signal)
    plt.plot(time, input_signal, 'x')
    plt.show()
    '''
    
    out = demod.demodulate(input_signal)
    assert out.sum() == 4
    #print(demod)
