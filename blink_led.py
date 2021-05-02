import numpy as np
from PIL import Image
import PySimpleGUI as sg
from WiPHY.tx import ASK_Tx
from WiPHY.rx import ASK_Rx
import matplotlib.pyplot as plt
from WiPHY.utils import readYaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

class ledToggler():

    def __init__(self, 
                 radio_config):

        self.__radio_config = radio_config
        self.__config = None
        self.__tx_settings = None
        self.__rx_settings = None
        self.__sdr_Settings = None
        self.__common_settings = None

        self.__tx_handle = None
        self.__rx_handle = None

        self.__orig_img_PIL = None
        self.__orig_array = None
        self.__led_stat = False
    
    @property
    def led_stat(self):
        return self.__led_stat
    
    def toggle(self):
        self.__led_stat = not self.__led_stat
    
    def get_orig_PIL_img(self):
        if self.__orig_array is None:
            return (False, None)
        data = {}
        data["arr"] = self.__orig_array
        data["size"] = self.__orig_array.nbytes
        return (True, data)
    
    def get_recv_img(self):
        if self.__orig_array is None:
            return (False, None)
        self.__recv_img = np.random.randint(0, 256, size=(self.__orig_array.shape), dtype=np.uint8)
        return (True, self.__recv_img)

    def init_tx(self):
        if self.__tx_settings is None or self.__common_settings is None:
            return False
        
        comm_port = self.__tx_settings['comm_port']
        baud_rate = self.__tx_settings['arduino_comm_baud_rate']
        timeout = self.__tx_settings['arduino_comm_timeout']
        retires = self.__tx_settings['transmit_retries']
        crc_poly = self.__common_settings['crc_poly']
        preamble = self.__common_settings['preamble']

        self.__tx_handle = ASK_Tx(comm_port=comm_port,
                                  baud_rate=baud_rate,
                                  timeout=timeout,
                                  num_retransmit_retries=retires,
                                  crc_polynomial=crc_poly,
                                  barker_seq=preamble)
        
        return True
    
    def load_image(self, path):
        self.__orig_img_PIL = Image.open(path)
        self.__orig_array = np.asarray(self.__orig_img_PIL)
        return True
    
    def get_tx_handler(self):
        if self.__tx_handle is None:
            return (False, None)
        return (True, id(self.__tx_handle))
    
    def get_rx_handler(self):
        if self.__rx_handle is None:
            return (False, None)
        return (True, id(self.__rx_handle))
    
    def init_rx(self):
        if self.__sdr_Settings is None or self.__common_settings is None or self.__rx_settings is None:
            return False
        
        sample_rate = self.__sdr_Settings['sample_rate']
        freq = self.__sdr_Settings['freq']
        freq_corr = self.__sdr_Settings['freq_corr']
        gain = self.__sdr_Settings['gain']
        log_captures = self.__sdr_Settings['log_captures']
        capture_len = self.__sdr_Settings['capture_len']
        symbol_dur = self.__sdr_Settings['symbol_dur']
        buffer_size = self.__sdr_Settings['log_buffer_size']
        crc_poly = self.__common_settings['crc_poly']
        preamble = self.__common_settings['preamble']
        alpha = self.__rx_settings['time_sync_alpha']
        dec_thershold = self.__rx_settings['dec_thershold']

        self.__rx_handle = ASK_Rx(sample_rate=sample_rate,
                                  center_freq=freq,
                                  freq_corr=freq_corr,
                                  symbol_dur=symbol_dur,
                                  gain=gain,
                                  log_captures=log_captures,
                                  capture_len=capture_len,
                                  barker_seq=preamble,
                                  crc_polynomial=crc_poly,
                                  max_logs_buffer_size=buffer_size,
                                  alpha=alpha,
                                  decision_thershold=dec_thershold)
        return True
    
    def get_sdr_settings(self):
        if self.__config is None:
            return (False, None)
        
        self.__sdr_Settings = {}
        self.__sdr_Settings['sample_rate'] = self.__config['sdr_settings']['sample_rate']
        self.__sdr_Settings['freq'] = self.__config['sdr_settings']['center_freq']
        self.__sdr_Settings['freq_corr'] = self.__config['sdr_settings']['freq_corr']
        self.__sdr_Settings['gain'] = self.__config['sdr_settings']['gain']
        self.__sdr_Settings['log_captures'] = self.__config['sdr_settings']['log_captures']
        self.__sdr_Settings['capture_len'] = self.__config['sdr_settings']['capture_len']
        self.__sdr_Settings['symbol_dur'] = self.__config['sdr_settings']['baseband_symbol_dur']
        self.__sdr_Settings['log_buffer_size'] = self.__config['sdr_settings']['max_logs_buffer_size']
        return (True, self.__sdr_Settings)
    
    def get_common_settings(self):
        if self.__config is None:
            return (False, None)
        
        self.__common_settings = {}
        self.__common_settings['crc_poly'] = self.__config['common_settings']['crc_polynomial']
        self.__common_settings['preamble'] = self.__config['common_settings']['barker_seq']
        return (True, self.__common_settings)


    def get_rx_settings(self):
        if self.__config is None:
            return (False, None)
        
        self.__rx_settings = {}
        self.__rx_settings['time_sync_alpha'] = self.__config['rx_settings']['time_sync']['alpha']
        self.__rx_settings['dec_thershold'] = self.__config['rx_settings']['ask_demodulator']['decision_thershold']
        return (True, self.__rx_settings)

    def get_tx_settings(self):
        if self.__config is None:
            return (False, None)
        
        self.__tx_settings = {}
        self.__tx_settings['data_pin'] = self.__config['tx_settings']['data_pin']
        self.__tx_settings['arduino_comm_baud_rate'] = self.__config['tx_settings']['arduino_comm_baud_rate']
        self.__tx_settings['arduino_comm_timeout'] = self.__config['tx_settings']['arduino_comm_timeout']
        self.__tx_settings['transmit_retries'] = self.__config['tx_settings']['transmit_retries']
        self.__tx_settings['comm_port'] = self.__config['tx_settings']['comm_port']
        return (True, self.__tx_settings)
        
    def load_radio_config(self):
        try:
            self.__config = readYaml(self.__radio_config)
            return True
        except Exception as e:
            self.__config = False
            raise Warning(e)
            return False
    
    @property
    def radio_config(self):
        return self.__config

row_2_layout_size = (30, 1)
row_3_layout_size_left = (15, 1)
row_3_layout_size_right = (5, 1)
row_3_layout_size_right_two = (11, 1)
row_3_layout_size_right_three = (12, 1)

row_2_layout_left = [
    [sg.Text("Tx Settings", justification="center", size=row_2_layout_size)],
]


row_2_layout_right_one = [
    [sg.Text("Rx Settings", justification="center", size=row_2_layout_size)],
]

row_2_layout_right_two = [
    [sg.Text("SDR Settings", justification="center", size=row_2_layout_size)],
]

row_2_layout_right_three = [
    [sg.Text("Common Settings", justification="center", size=row_2_layout_size)],
]


row_3_layout_left = [
    [sg.Text("Data Pin: ", justification='left'), sg.Text("", justification='left', key='--tx-data-pin--', size=row_3_layout_size_left)],
    [sg.Text("Comm Port: "), sg.Text("", justification='left', key='--tx-comm-port--', size=row_3_layout_size_left)],
    [sg.Text("Baud Rate: "), sg.Text("", justification='left', key='--tx-baud-rate--', size=row_3_layout_size_left)],
    [sg.Text("Timeout: "), sg.Text("", justification='left', key='--tx-timeout--', size=row_3_layout_size_left)],
    [sg.Text("Num retries: "), sg.Text("", justification='left', key='--tx-num-retires--', size=row_3_layout_size_left)]
]

row_3_layout_right =  [
    [sg.Text("Time Sync update rate: ", justification='left'), sg.Text("", justification='left', key='--rx-time-sync-alpha--', size=row_3_layout_size_right)],
    [sg.Text("ASK dec. thershold", justification="left"), sg.Text("", justification='left', key='--rx-ask-dec-thershold--', size=row_3_layout_size_right)],
]

row_3_layout_right_two = [
    [sg.Text("Sample Rate: ", justification='left'), sg.Text("", justification='left', key='--sdr-samp-rate--', size=row_3_layout_size_right_two)],
    [sg.Text("Freq (Hz): "), sg.Text("", justification='left', key='--sdr-freq--', size=row_3_layout_size_right_two)],
    [sg.Text("Freq Corr: "), sg.Text("", justification='left', key='--sdr-freq-corr--', size=row_3_layout_size_right_two)],
    [sg.Text("Gain: "), sg.Text("", justification='left', key='--sdr-gain--', size=row_3_layout_size_right_two)],
    [sg.Text("Log Captures: "), sg.Text("", justification='left', key='--sdr-log-captures--', size=row_3_layout_size_right_two)],
    [sg.Text("Capture Len: "), sg.Text("", justification='left', key='--sdr-capture-len--', size=row_3_layout_size_right_two)],
    [sg.Text("Symbol Dur (s): "), sg.Text("", justification='left', key='--sdr-symbol-dur--', size=row_3_layout_size_right_two)],
    [sg.Text("Log buffer size: "), sg.Text("", justification='left', key='--sdr-log-buffer--', size=row_3_layout_size_right_two)],
]

row_3_layout_right_three =  [
    [sg.Text("CRC Poly: ", justification='left'), sg.Text("", justification='left', key='--common-crc-poly--', size=row_3_layout_size_right_three)],
    [sg.Text("Preamble: ", justification="left"), sg.Text("", justification='left', key='--common-preamble--', size=row_3_layout_size_right_three)],
]

radio_settings_layout = [
[sg.Column(row_2_layout_left, element_justification='c'), sg.VSeperator(), sg.Column(row_2_layout_right_one, element_justification='c'), 
      sg.VSeperator(), sg.Column(row_2_layout_right_two, element_justification='c'), sg.VSeperator(), sg.Column(row_2_layout_right_three, element_justification='c')],
    [sg.Column(row_3_layout_left, element_justification='l'), sg.VSeperator(), sg.Column(row_3_layout_right, element_justification='l'), sg.VSeperator(),
       sg.Column(row_3_layout_right_two, element_justification='l'), sg.VSeperator(), sg.Column(row_3_layout_right_three, element_justification='l')],]

transmiision_tab_layout = [
    [sg.Text("Select an Image: "), sg.Input(key='--orig-img-file--', size=(40, 1)), sg.FilesBrowse(), sg.Button("Toggle", key="--led-toggle-button--")],
    [sg.Frame(title="LED", layout=[[sg.Graph(canvas_size=(100, 100), graph_bottom_left=(0, 0), graph_top_right=(400, 400), background_color='white', key= 'graph')], ])],
]

layout = [
    [sg.Text("Radio Configuration File: "), sg.Input(key='--radio-config-file--', size=(90, 1)), sg.FilesBrowse(), sg.Button("Load", key="--radio-config-load--")],
    [sg.Button("Init Tx !", key='--init-tx-button--'), sg.Text("None", key="--init-tx-status--", size=(40, 1))],
    [sg.Button("Init Rx !", key='--init-rx-button--'), sg.Text("None", key="--init-rx-status--", size=(40, 1))],
    [sg.TabGroup(
        [
            [sg.Tab('Radio setting', radio_settings_layout)],
            [sg.Tab('Led Toggler', transmiision_tab_layout)]
        ])
    ],
    
]

def update_tx_settings(window, config):
    window['--tx-data-pin--'].update(config['data_pin'])
    window['--tx-comm-port--'].update(config['comm_port'])
    window['--tx-baud-rate--'].update(config['arduino_comm_baud_rate'])
    window['--tx-timeout--'].update(config['arduino_comm_timeout'])
    window['--tx-num-retires--'].update(config['transmit_retries'])

def update_rx_settings(window, config):
    window['--rx-time-sync-alpha--'].update(config['time_sync_alpha'])
    window['--rx-ask-dec-thershold--'].update(config['dec_thershold'])

def update_sdr_settings(window, config):
    window['--sdr-samp-rate--'].update(config['sample_rate'])
    window['--sdr-freq--'].update(config['freq'])
    window['--sdr-freq-corr--'].update(config['freq_corr'])
    window['--sdr-gain--'].update(config['gain'])
    window['--sdr-capture-len--'].update(config['capture_len'])
    window['--sdr-symbol-dur--'].update(config['symbol_dur'])
    window['--sdr-log-buffer--'].update(config['log_buffer_size'])
    window['--sdr-log-captures--'].update(config['log_captures'])

def update_common_settings(window, config):
    window['--common-crc-poly--'].update(config['crc_poly'])
    window['--common-preamble--'].update(config['preamble'])

window = sg.Window("WiPHY LED toggler", layout, finalize=True)
graph = window['graph']
led_graph = graph.draw_circle((200, 200), 70, fill_color='white', line_color='black')

def draw_led(status):
    if status:
        graph.TKCanvas.itemconfig(led_graph, fill = "red")
    else:
        graph.TKCanvas.itemconfig(led_graph, fill = "white")

def draw_fig_on_canvas():
    pass

if __name__ == "__main__":

    streamer = None
    is_tx_initialized = False
    is_rx_initialized = False
    
    # Orig image figure
    orig_img_fig = plt.figure(figsize=(5,3))
    orig_ax = orig_img_fig.add_subplot(1, 1, 1)
    orig_img_fig.tight_layout(pad=0)

    # Recv image figure
    recv_img_fig = plt.figure(figsize=(5,3))
    recv_ax = recv_img_fig.add_subplot(1, 1, 1)
    recv_img_fig.tight_layout(pad=0)

    while True:

        event, values = window.read()

        if event == '--radio-config-load--':
            streamer = ledToggler(radio_config=values['--radio-config-file--'])
            if streamer.load_radio_config():
                
                # Load tx settings
                stat, config = streamer.get_tx_settings()
                if stat:
                    update_tx_settings(window, config)

                # Load rx settings
                stat, config = streamer.get_rx_settings()
                if stat:
                    update_rx_settings(window, config)
                
                # Load SDR settings
                stat, config = streamer.get_sdr_settings()
                if stat:
                    update_sdr_settings(window, config)
                
                # Load Common settings
                stat, config = streamer.get_common_settings()
                if stat:
                    update_common_settings(window, config)

            else:
                print("failed to load")

        if event == '--init-tx-button--':
            if streamer is not None:
                init_stat = streamer.init_tx()
                if init_stat:
                    stat, _id_ = streamer.get_tx_handler()
                    if stat:
                        window['--init-tx-status--'].update("Initialized Tx object at: %x"%(_id_))
            else:
                print("Failed: Streamer object is None")
        
        if event == '--init-rx-button--':
            if streamer is not None:
                init_stat = streamer.init_rx()
                if init_stat:
                    stat, _id_ = streamer.get_rx_handler()
                    if stat:
                        window['--init-rx-status--'].update("Initialized Rx object at: %x"%(_id_))
            else:
                print("failed: Streamer object is None")
        
        if event == "--led-toggle-button--":
            tx_handle = streamer.get_tx_handler()
            rx_handle = streamer.get_rx_handler()

            seq_id = 0
            if streamer.led_stat:
                payload = 0
            else:
                payload = 1
            
            streamer.toggle()
            draw_led(streamer.led_stat)

        if event == sg.WIN_CLOSED:
            break
    