import pytest
import numpy as np
from utils import bit2dec, bit2str, str2dec, dec2bin

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
