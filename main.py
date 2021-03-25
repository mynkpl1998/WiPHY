import argparse
from WiPHY.rx import ASK_Rx
from WiPHY.utils import readYaml

parser = argparse.ArgumentParser(description="Script to initiate ASK RX to start listening for frame using RTL-SDR.")
parser.add_argument("-c", "--radio-config", type=str, required=True, help="Radio Configuration File.")

if  __name__ == "__main__":

	# Parse command line arguments.
	args = parser.parse_args()

	# Load radio configuration file.
	radio_config_dict = readYaml(file_path=args.radio_config)

	# Get SDR settings.
	sample_rate = int(float(radio_config_dict['sdr_settings']['sample_rate']))
	center_freq = int(float(radio_config_dict['sdr_settings']['center_freq']))
	freq_corr = int(radio_config_dict['sdr_settings']['freq_corr'])
	gain = radio_config_dict['sdr_settings']['gain']
	if gain != "auto":
		gain = int(gain)
	log_captues = bool(radio_config_dict['sdr_settings']['log_captures'])
	baseband_symbol_dur = float(radio_config_dict['sdr_settings']['baseband_symbol_dur'])
	
	# Time Sync block settings.
	alpha = float(radio_config_dict['time_sync']['alpha'])

	# Frame Detector block settings.
	crc_polynomial = int(radio_config_dict['frame_detector']['crc_polynomial'])
	barker_seq = int(radio_config_dict['frame_detector']['barker_seq'])

	# ASK Demodulator settings.
	decision_thershold = int(radio_config_dict['ask_demodulator']['decision_thershold'])

	# Create radio object.
	radio = ASK_Rx(sample_rate=sample_rate,
	               center_freq=center_freq,
				   freq_corr=freq_corr,
				   symbol_dur=baseband_symbol_dur,
				   gain=gain,
				   log_captures=log_captues,
				   alpha=alpha,
				   decision_thershold=decision_thershold,
				   barker_seq=barker_seq,
				   crc_polynomial=crc_polynomial)