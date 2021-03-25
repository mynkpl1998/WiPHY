import time
import pickle
import atexit
import numpy as np
from wasabi import msg
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
from WiPHY.utils import lowPassFilter, Muller, ASK_Demodulator, FrameDetector

class ASK_Rx():
    """Implements a Amplitude Shift Keying RX using
       the RTL-SDR.
       
       Inputs
       ------
       * sample_rate (int):                       SDR sampling rate in samples per second.
       * center_freq (int):                       SDR center frequency in Hz.
       * freq_corr (int):                         SDR frequency correction in ppm.
       * symbol_dur (float):                      Baseband signal duration in seconds.
       * gain (float or str):                     SDR gain.
       * log_captures (bool):                     Enables/Disables the capture logging after each
                                                    processing blocks.
       * alpha (float):                           Update/Learning rate. Used by time sync block.
       * decision_thershold (float):              ASK demodulator decision thershold. Used by ASK 
                                                    demodulator block
       * barker_seq (int):                        Barker seq to use for frame detection. Used by Frame
                                                    detector block.
       * crc_polynomial (int):                    CRC Polynomial to use for frame detection. Used by Frame
                                                    detector block. 
       
       Attributes
       ----------
       * sample_rate (int):                       Returns sample rate to which device is tuned to, 
                                                    in samples per second.
       * center_freq (int):                       Returns center freq to which device is tuned to, 
                                                    in Hz.
       * freq_corr (int):                         Returns freq corr value of the device in ppm.
       * symbol_dur (float):                      Returns baseband symbol duration in seconds.
       * gain (float or str):                     Returns SDR gain.
       * log_capture (bool):                      Returns whther the capture logging is enabled/disabled.
       * alpha (float):                           Returns update/learning rate in use by time sync block.
       * decision_thershold (float):              Returns decision thershold in used by ASK 
                                                    demodulator block.
       * barker_seq (int):                        Returns barker seq in use by frame detection block.
       * crc_polynomial (int):                    Returns CRC polynomial in use by frame detector block. 
    """
    def __init__(self, 
                 sample_rate=1e6, 
                 center_freq=350e6, 
                 freq_corr=0, 
                 symbol_dur=1e-3, 
                 gain='auto', 
                 log_captures=False,
                 alpha=0.1,
                 decision_thershold=0.5,
                 barker_seq=29,
                 crc_polynomial=13):
        
        self.__sample_rate = int(sample_rate)
        self.__center_freq = int(center_freq)
        self.__freq_corr = int(freq_corr)
        self.__sym_dur = float(symbol_dur)
        self.__gain = gain
        self.__log_captues = bool(log_captures)
        self.__alpha = float(alpha)
        self.__decision_thershold = float(decision_thershold)
        self.__barker_seq = int(barker_seq)
        self.__crc_polynomial = int(crc_polynomial)

        self.__sdr = None

        # Tune the radio to required settings.
        self.__tune_sdr(self.sample_rate,
                        self.center_freq,
                        self.freq_corr,
                        self.gain)

        # Register the clean up function.
        # This function will close the SDR connection if case anything goes wrong.
        atexit.register(self.cleanup)

        # Registers all the signal processing blocks.
        self.__processing_blocks = [
            #self.low_pass_filtering,
            #self.time_sync,
            #self.ask_demod,
            #self.frame_detection
        ]

        # Book keeping variables.
        self.__stage_processing_time_mean = 0.0
        self.__stage_processing_time_step = 0

        if self.log_captues:
            self.sample_captures_data = {}
            self.sample_captures_data['sdr_settings'] = {}
            self.sample_captures_data['sdr_settings']['sample_rate'] = self.sample_rate
            self.sample_captures_data['sdr_settings']['center_freq'] = self.center_freq
            self.sample_captures_data['sdr_settings']['symbol_dur'] = self.symbol_dur
            self.sample_captures_data['sdr_settings']['freq_corr'] = self.freq_corr
            self.sample_captures_data['sdr_settings']['gain'] = self.gain
            self.sample_captures_data['sdr_settings']['alpha'] = self.alpha
            self.sample_captures_data['sdr_settings']['crc_polynomial'] = self.crc_polynomial
            self.sample_captures_data['sdr_settings']['decision_thershold'] = self.decision_thershold
            self.sample_captures_data['sdr_settings']['barker_seq'] = self.barker_seq
            
            self.sample_captures_data['raw'] = []
            self.sample_captures_data[self.low_pass_filtering.__name__] = []
            self.sample_captures_data[self.time_sync.__name__] = []
            self.sample_captures_data[self.ask_demod.__name__] = []
            self.sample_captures_data[self.frame_detection.__name__] = []

        # Over write dead beaf samples of the sdr buffer.
        for _ in range(0, 4):
            self.__sdr.read_samples(1024)
        
        """
           Initialize all the signal processing blocks.
        """
        # Time Sync block.
        sps = self.sample_rate * self.symbol_dur
        alpha = self.alpha
        self.__time_sync_block = Muller(sps=sps,
                                        alpha=alpha)

        # ASK demodulator block.
        decision_thershold = self.decision_thershold
        self.__modulator = ASK_Demodulator(decision_thershold=decision_thershold)

        # Frame Detector block.
        barker_seq = self.barker_seq
        crc_polynomial = self.crc_polynomial
        self.__frameDetector = FrameDetector(barker_seq=barker_seq,
                                             crc_polynomial=crc_polynomial)
    
    @property
    def sample_rate(self):
        """Returns SDR sample rate in samples per seconds.
        """
        return self.__sample_rate
    
    @property
    def center_freq(self):
        """Returns SDR Center freq in Hz.
        """
        return self.__center_freq
    
    @property
    def freq_corr(self):
        """Returns SDR freq correction in ppm.
        """
        return self.__freq_corr
    
    @property
    def symbol_dur(self):
        """Returns baseband signal duration in seconds.
        """
        return self.__sym_dur
    
    @property
    def gain(self):
        """Returns SDR gain.
        """
        return self.__gain
    
    @property
    def alpha(self):
        """Returns Muller learning/update rate.
        """
        return self.__alpha
    
    @property
    def decision_thershold(self):
        """Returns ASK Demodulator decision
           thershold.
        """
        return self.__decision_thershold
    
    @property
    def crc_polynomial(self):
        """Returns CRC polynomial used by
           Frame Detector.
        """
        return self.__crc_polynomial
    
    @property
    def barker_seq(self):
        """Returns the barker sequence
           in use by Frame Detector.
        """
        return self.__barker_seq
    
    @property
    def log_captues(self):
        """Return whether sample capture logging is enabled/disabled
        """
        return self.__log_captues
	
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

    def __tune_sdr(self, sample_rate, center_freq, freq_corr, gain):
        """Tunes the radio to desired settings.
        
        Inputs
        ------

        * sample_rate (float):                     SDR sampling rate in samples per second.
        * center_freq (int):                       SDR center frequency in Hz.
        * freq_corr (int):                         SDR frequency correction in ppm.
        * gain (float or str):                     SDR gain.
        """
        
        if self.__sdr is None:
            self.__sdr = RtlSdr()
        
        self.__sdr.sample_rate = sample_rate
        self.__sdr.center_freq = center_freq
        if freq_corr != 0:
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
            self.__sdr = None
            if self.log_captues:
                fileHanlder = open('samples_captures.pkl', 'wb')
                pickle.dump(self.sample_captures_data, fileHanlder)
                fileHanlder.close()
                msg.good("Dumped sample captures.")
