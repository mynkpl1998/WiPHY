import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import Muller

fileHandler = open("samples_captures.pkl", "rb")
data = pickle.load(fileHandler)
fileHandler.close()

print(data['sdr_settings'])
raw_samples = np.array(data['raw'][0:1]).flatten()
m1 = Muller(sps=data['sdr_settings']['sample_rate'] * data['sdr_settings']['symbol_dur'],
            alpha=0.1)

out = m1.sync(raw_samples)
print(raw_samples.shape[0])
print(out.shape[0])