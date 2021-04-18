import time
import argparse
from WiPHY.rx import ASK_Rx
from WiPHY.tx import ASK_Tx
from WiPHY.utils import readYaml
from alive_progress import alive_bar

parser = argparse.ArgumentParser(description="Script to analyze Tx-Rx performance.")
parser.add_argument("-c", "--radio-config", type=str, required=True, help="Radio Configuration File.")

if  __name__ == "__main__":

    # Parse command line arguments.
    args = parser.parse_args()

    # Load radio configuration file.
    radio_config_dict = readYaml(file_path=args.radio_config)

    # Get SDR and rx settings.
    sample_rate = int(float(radio_config_dict['sdr_settings']['sample_rate']))
    center_freq = int(float(radio_config_dict['sdr_settings']['center_freq']))
    freq_corr = int(radio_config_dict['sdr_settings']['freq_corr'])
    gain = radio_config_dict['sdr_settings']['gain']
    if gain != "auto":
        gain = int(gain)
    log_captues = bool(radio_config_dict['sdr_settings']['log_captures'])
    baseband_symbol_dur = float(radio_config_dict['sdr_settings']['baseband_symbol_dur'])
    capture_len = int(radio_config_dict['sdr_settings']['capture_len'])
    logs_buffer_max_size = int(radio_config_dict['sdr_settings']['max_logs_buffer_size'])

    # Time Sync block settings.
    alpha = float(radio_config_dict['rx_settings']['time_sync']['alpha'])

    # Frame Detector block settings.
    crc_polynomial = int(radio_config_dict['common_settings']['crc_polynomial'])
    barker_seq = int(radio_config_dict['common_settings']['barker_seq'])

    # ASK Demodulator settings.
    decision_thershold = float(radio_config_dict['rx_settings']['ask_demodulator']['decision_thershold'])

    # Create rx radio object.
    radio_rx = ASK_Rx(sample_rate=sample_rate,
                      center_freq=center_freq,
                      freq_corr=freq_corr,
                      symbol_dur=baseband_symbol_dur,
                      gain=gain,
                      log_captures=log_captues,
                      alpha=alpha,
                      decision_thershold=decision_thershold,
                      barker_seq=barker_seq,
                      crc_polynomial=crc_polynomial,
                      capture_len=capture_len,
                      max_logs_buffer_size=logs_buffer_max_size)

    # Get Tx settings
    comm_port = str(radio_config_dict['tx_settings']['comm_port'])
    baud_rate = int(radio_config_dict['tx_settings']['arduino_comm_baud_rate'])
    timeout = float(radio_config_dict['tx_settings']['arduino_comm_timeout'])
    retries_count = int(radio_config_dict['tx_settings']['transmit_retries'])
    crc_polynomial = int(radio_config_dict['common_settings']['crc_polynomial'])
    barker_seq = int(radio_config_dict['common_settings']['barker_seq'])

    # Create tx radio object.
    radio_tx = ASK_Tx(comm_port=comm_port,
                      baud_rate=baud_rate,
                      timeout=timeout,
                      num_retransmit_retries=retries_count,
                      barker_seq=barker_seq,
                      crc_polynomial=crc_polynomial)

    # Initialize the data to send.
    payload = 5
    seq_id = 1
    num_frames = logs_buffer_max_size

    with alive_bar(num_frames) as bar:
        for num_frame in range(0, num_frames):
            
            # Start capturing the frames.
            radio_rx.start_captures()
            time.sleep(0.01)
            
            # Send the data from tx.
            radio_tx.send(payload=payload, seq_id=seq_id)
            time.sleep(0.01)

            # Stop captures
            radio_rx.stop_captures()
            
            # Analyze captures.
            radio_rx.process_start_captures()
            bar()

