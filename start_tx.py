import time
import argparse
from WiPHY.tx import ASK_Tx
from pynput.keyboard import Listener
from WiPHY.utils import readYaml, Frame

parser = argparse.ArgumentParser(description="Script to initiate ASK TX to start transmitting frame using 315-433 MHz ASK modules connected to Arduino.")
parser.add_argument("-c", "--radio-config", type=str, required=True, help="Radio Configuration File.")

# Parse command line arguments.
args = parser.parse_args()

# Load radio configuration file.
radio_config_dict = readYaml(file_path=args.radio_config)

# Get tx settings.
comm_port = str(radio_config_dict['tx_settings']['comm_port'])
baud_rate = int(radio_config_dict['tx_settings']['arduino_comm_baud_rate'])
timeout = float(radio_config_dict['tx_settings']['arduino_comm_timeout'])
retries_count = int(radio_config_dict['tx_settings']['transmit_retries'])
crc_polynomial = int(radio_config_dict['common_settings']['crc_polynomial'])
barker_seq = int(radio_config_dict['common_settings']['barker_seq'])

# Create tx object
radio = ASK_Tx(comm_port=comm_port,
                baud_rate=baud_rate,
                timeout=timeout,
                num_retransmit_retries=retries_count,
                barker_seq=barker_seq,
                crc_polynomial=crc_polynomial)

def on_press(counter):

    # Start transmitting the frame.
    payload = 1
    seq_id = 0

    print("Sending toggle packet...")
    status, count = radio.send(payload=payload,
                            seq_id=seq_id)
    assert status
    print("Sending toggle packet done.")



if  __name__ == "__main__":
    
    with Listener(on_press=on_press) as listener:
        listener.join()