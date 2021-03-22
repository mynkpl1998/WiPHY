import time
import pickle
import atexit
import numpy as np
from wasabi import msg
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
from utils import lowPassFilter, muller, ASKdemod, frameDetector

class AskRecv():

	def __init__(self, 
	            sample_rate=1e6, 
				center_freq=350e6, 
				freq_corr=0,
				baseband_bw=1e3, 
				gain='auto', 
				log_captues=False):
		self.sample_rate = sample_rate
		self.center_freq = center_freq
		self.freq_corr = freq_corr
		self.baseband_bw = float(baseband_bw)
		self.gain = gain
		self.log_captues = log_captues

		# Calculations
		self.__symbol_duration = 1/self.baseband_bw

		self.__sdr = None
		self.tune_sdr(self.sample_rate,
		              self.center_freq,
					  self.freq_corr,
					  self.gain)
		
		# Register clean up function
		atexit.register(self.cleanup)

		# Register all signal processing blocks
		self.__processing_blocks = [
			#self.low_pass_filtering,
			self.time_sync,
			self.ask_demod,
			self.frame_detection
		]

		# Meta data variables
		self.__stage_processing_time_mean = 0.0
		self.__stage_processing_time_step = 0

		if self.log_captues:
			self.sample_captures_data = {}
			self.sample_captures_data['sdr_settings'] = {}
			self.sample_captures_data['sdr_settings']['sample_rate'] = self.sample_rate
			self.sample_captures_data['sdr_settings']['center_freq'] = self.center_freq
			self.sample_captures_data['sdr_settings']['symbol_dur'] = self.__symbol_duration
			self.sample_captures_data['sdr_settings']['freq_corr'] = self.freq_corr
			self.sample_captures_data['sdr_settings']['gain'] = self.gain
			
			self.sample_captures_data['raw'] = []
			self.sample_captures_data[self.low_pass_filtering.__name__] = []
			self.sample_captures_data[self.time_sync.__name__] = []
			self.sample_captures_data[self.ask_demod.__name__] = []
			self.sample_captures_data[self.frame_detection.__name__] = []
		
		# Override dead beaf samples of the sdr buffer
		for _ in range(0, 4):
			self.__sdr.read_samples(1024)
		
		'''
			Initialize signal processing blocks
		'''
		# Time Sync 
		sps = self.sample_rate * self.__symbol_duration
		self.__time_sync_block = muller(sps)

		# ASK demodulator
		self.__modulator = ASKdemod(thershold=0.6)

		# Frame Detector
		self.__frameDetector = frameDetector(barker_seq=np.array([1, 1, 1, 0, 1]))
	
	def low_pass_filtering(self, input_samples):
		return self.__low_pass_filter.apply(input_samples)
	
	def time_sync(self, input_samples):
		return self.__time_sync_block.sync(input_samples)
	
	def ask_demod(self, input_samples):
		return self.__modulator.demod(input_samples)
	
	def frame_detection(self, input_samples):
		return self.__frameDetector.step(input_samples)
	
	def read_sdr_settings(self, ):
		if self.__sdr is None:
			msg.fail("No active SDR device found.")
		return (self.__sdr.sample_rate,
		        self.__sdr.center_freq,
				self.freq_corr,
				self.__sdr.gain)

	def tune_sdr(self, sample_rate, center_freq, freq_corr, gain):
		
		if self.__sdr is None:
			self.__sdr = RtlSdr()
		
		self.__sdr.sample_rate = sample_rate
		self.__sdr.center_freq = center_freq
		self.__sdr.freq_correction = freq_corr
		self.__sdr.gain = gain

		msg.good("SDR is tuned to sample rate: %d, center freq: %d, freq corr: %d, gain %s."%(self.read_sdr_settings()))

	def step(self):
		self.__stage_processing_time_step += 1
		start = time.process_time()
		
		samples = self.__sdr.read_samples(8192)
		input = samples
		
		if self.log_captues:
			self.sample_captures_data['raw'].append(input)
		
		for operation in self.__processing_blocks:
			output = operation(input)

			if self.log_captues:
				self.sample_captures_data[operation.__name__].append(output)	
			input = output

		end = time.process_time()
		
		self.__stage_processing_time_mean = (((self.__stage_processing_time_step-1)/self.__stage_processing_time_step) * self.__stage_processing_time_mean) \
			+ ((1/self.__stage_processing_time_step) * (end - start))
	
	def listen(self):
		try:
			while True:
				self.step()
		except KeyboardInterrupt:
			pass
	
	def cleanup(self):
		if self.__sdr is not None:
			self.__sdr.close()
			msg.info("Closed connection to the device. Avg step proc time %.6fs"%(self.__stage_processing_time_mean))
			if self.log_captues:
				fileHanlder = open('samples_captures.pkl', 'wb')
				pickle.dump(self.sample_captures_data, fileHanlder)
				fileHanlder.close()
				msg.good("Dumped sample captures.")

if __name__ == "__main__":

	obj = AskRecv(sample_rate=245e3,
	              center_freq=315e6,
				  freq_corr=-207,
				  gain='auto',
				  log_captues=True)
	
	obj.listen()