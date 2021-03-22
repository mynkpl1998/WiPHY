import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import muller

fileHandler = open("samples_captures.pkl", "rb")
data = pickle.load(fileHandler)
fileHandler.close()

raw_samples = np.array(data['raw'][0:1]).flatten()
print(data['ask_demod'][0])