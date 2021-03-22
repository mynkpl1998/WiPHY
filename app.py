import re
import time
import PySimpleGUI as sg
from rtlsdr import RtlSdr

class RTLSdrDevice:

    def __init__(self,
                 serial,
                 device_index):
    
        self.serial = serial
        self.device_index = device_index
        self.sdr = RtlSdr(device_index)

    def query_device_settings(self):
        data = {}

        # Get sample rate of the device
        data['sample_rate'] = {}
        data['sample_rate']['value'] = self.sdr.get_sample_rate()
        data['sample_rate']['min'] = 250e3
        data['sample_rate']['max'] = 2.4e6

        # Get center freq of the device
        data['cent_freq'] = {}
        data['cent_freq']['value'] = self.sdr.get_center_freq()
        data['cent_freq']['max'] = 1700e6
        data['cent_freq']['min'] = 50e6

        # Get freq correction value
        data['freq_corr'] = {}
        data['freq_corr']['value'] = self.sdr.get_freq_correction()
        data['freq_corr']['max'] = 1000
        data['freq_corr']['min'] = -1000

        # Get gains
        data['gains'] = {}
        data['gains']['value'] = self.sdr.get_gain()
        data['gains']['values'] = self.sdr.get_gains()
        data['gains']['values'].append('auto')
        return data
        

class ASK_RECV:
    
    def __init__(self):

        self.title = "RTL-SDR ASK Receiever"

        # Use to store data
        self.data_dict = {}
        self.console_log_data = ''

        self.get_devices()
    
        # Initialize application layout
        self.layout = self.init_layout(self.data_dict)

        # Initialize application window with layout
        self.window = self.init_window(self.layout,
                                       size=11)
        
        # Application state
        self.device_selected = False
        self.device = None
        self.widgets_state_to_change= [
            '--RADIO_DEINIT_BUTON--',
            '--QUERY_DEVICE_BUTON--',
            '--CENT_FREQ_SLIDER--',
            '--SAMP_RATE_SLIDER--',
            '--SAMP_RATE_IN--',
            '--CENT_FREQ_IN--',
            '--FREQ_CORR_IN--',
            '--GAINS_DROPDOWN--'
        ]

        self.slider_events = ['--CENT_FREQ_SLIDER--',
            '--SAMP_RATE_SLIDER--'
        ]
        self.slider_events_input_map = {
            '--CENT_FREQ_SLIDER--': '--CENT_FREQ_IN--',
            '--SAMP_RATE_SLIDER--': '--SAMP_RATE_IN--'
        }
    
    def get_devices(self):
        serial_numbers = RtlSdr.get_device_serial_addresses()
        device_indexes = [ RtlSdr.get_device_index_by_serial(serial_number) for serial_number in serial_numbers ]
        # Dummy Device
        serial_numbers.append("00000000")
        device_indexes.append("-1")

        # Device strings
        device_strings = [ "Device Index: " + str(device_index) + ", Device Serial: " + serial_number for serial_number,device_index in zip(serial_numbers, device_indexes)]

        self.data_dict['sdr_devices'] = {}
        self.data_dict['sdr_devices']['serials'] = serial_numbers
        self.data_dict['sdr_devices']['indexes'] = device_indexes
        self.data_dict['sdr_devices']['device_strings'] = device_strings

        print("[i]: Found %d devices."%(len(serial_numbers)-1))

    def init_layout(self, data_dict):
        return [ 
            [sg.Text("Select Radio Device(SDR): "),  sg.Combo(values=self.data_dict['sdr_devices']['device_strings'], key='--DEV_SELECTER--'), sg.Button('Query Device Settings', key='--QUERY_DEVICE_BUTON--'), sg.Button('De-Init', key='--RADIO_DEINIT_BUTON--'), sg.Button('Tune', key='--TUNE_RADIO_BUTON--')], 
            [sg.Text('_'*106)],
            [sg.Text('Sample Rate: '), sg.Input(size=(20, 10), key='--SAMP_RATE_IN--'), sg.Text('Hz'), sg.Slider(range=(6, 172), orientation='h', enable_events=True, key='--SAMP_RATE_SLIDER--'), sg.Text('Min: '), sg.Text('0.0 Hz', size=(13, 0), key='--SAMP_RATE_MIN--'), sg.Text('Max: '), sg.Text('0.0 Hz', size=(13, 0), key='--SAMP_RATE_MAX--')],
            [sg.Text('Center Freq: '), sg.Input(size=(20, 10), key='--CENT_FREQ_IN--'), sg.Text('Hz'), sg.Slider(range=(6, 172), orientation='h', enable_events=True, key='--CENT_FREQ_SLIDER--'), sg.Text('Min: '), sg.Text('0.0 Hz', size=(13, 0), key='--CENT_FREQ_MIN--'), sg.Text('Max: '), sg.Text('0.0 Hz', size=(13, 0), key='--CENT_FREQ_MAX--')],
            [sg.Text('Freq Correc: '), sg.Input(size=(20, 10), key='--FREQ_CORR_IN--'), sg.Text('ppm'), sg.Text('Gain: '), sg.Text('', size=(10, 0), key='--GAIN_VALUE--'), sg.Text("Gain Multiplier: "), sg.Combo(values=[], size=(10, 0), key='--GAINS_DROPDOWN--')],
            [sg.Text("Console:")],
            [sg.Multiline(size=(104, 20), key='--CONSOLE_TEXT--', autoscroll=True)]
        ]

    def init_window(self, layout, font="Helvetica", size=15):
        return sg.Window(self.title, 
                        layout,
                        font=(font, size))
    
    def log_message(self, msg):
        msg_str = time.strftime("%m/%d/%Y, %H:%M:%S") + ": " + msg + "\n"
        self.console_log_data += msg_str
    
    def change_widget_state(self, widgets=[], disabled=True):
        for widget in widgets:
            self.window[widget].update(disabled=disabled)
        
    def device_selection_handler(self, events, values):
        selected_device_string = values['--DEV_SELECTER--']
        dev = re.findall(r'[+-]?\d+', selected_device_string)
        if not self.device_selected and len(dev) != 0 and dev[0] != '-1':
            dev_index, device_serial = dev
            dev_index = int(dev_index)
            self.device = RTLSdrDevice(serial=device_serial,
                                       device_index=dev_index)
            self.device_selected = True
            msg = "Successfully opened connection to dev_index: %d, dev_serial: %s"%(dev_index, device_serial)
            self.log_message(msg)
            self.window['--DEV_SELECTER--'].update(value='')
            self.window['--DEV_SELECTER--'].update(disabled=True)
    
    def device_de_init_handler(self, events, values):
        if events == '--RADIO_DEINIT_BUTON--':
            if self.device is not None:
                self.device.sdr.close()
                self.device_selected = False
                self.window['--DEV_SELECTER--'].update(disabled=False)
                msg = "Successfully closed connection to dev_index: %d, dev_serial: %s"%(self.device.device_index, self.device.serial)
                self.log_message(msg)
    
    def slider_change_handler(self, event, values):

        if self.device_selected and event in self.slider_events:
            self.window[self.slider_events_input_map[event]].update(value=values[event])


    def tune_device_settings(self, event, values):

        if event == '--TUNE_RADIO_BUTON--' and self.device is not None:
            cent_freq = values['--CENT_FREQ_IN--']
            samp_rate = values['--SAMP_RATE_IN--']
            freq_corr = values['--FREQ_CORR_IN--']
            gain = values['--GAINS_DROPDOWN--']
            if gain == '':
                gain = 'auto'
            cent_freq = float(cent_freq)
            samp_rate = float(samp_rate)
            freq_corr = int(freq_corr)

            msg = "Requesting device to tune to, Cent Freq: %d Hz, Sample Rate: %d Hz, Freq Corr: %d ppm, Gain: %s."%(float(cent_freq),
                                                                                                                      float(samp_rate),
                                                                                                                      int(freq_corr),
                                                                                                                      str(gain))
            self.log_message(msg)

            self.device.sdr.sample_rate = samp_rate
            self.device.sdr.center_freq = cent_freq
            self.device.sdr.freq_correction = freq_corr
            self.device.sdr.gain = gain

            self.log_message("Radio is tuned to requested settings.")
    
    def query_device_settings_handler(self, events, values):
        
        if events == '--QUERY_DEVICE_BUTON--' and self.device is not None:
                self.log_message(msg="Starting to query device for settings.")
                queried_settings = self.device.query_device_settings()

                self.window['--SAMP_RATE_SLIDER--'].update(range=(queried_settings['sample_rate']['min'], queried_settings['sample_rate']['max']))
                self.window['--SAMP_RATE_IN--'].update(value=queried_settings['sample_rate']['value'])
                self.window['--SAMP_RATE_SLIDER--'].update(value=queried_settings['sample_rate']['value'])
                self.window['--SAMP_RATE_MIN--'].update(value=queried_settings['sample_rate']['min'])
                self.window['--SAMP_RATE_MAX--'].update(value=queried_settings['sample_rate']['max'])

                self.window['--CENT_FREQ_SLIDER--'].update(range=(queried_settings['cent_freq']['min'], queried_settings['cent_freq']['max']))
                self.window['--CENT_FREQ_IN--'].update(value=queried_settings['cent_freq']['value'])
                self.window['--CENT_FREQ_SLIDER--'].update(value=queried_settings['cent_freq']['value'])
                self.window['--CENT_FREQ_MIN--'].update(value=queried_settings['cent_freq']['min'])
                self.window['--CENT_FREQ_MAX--'].update(value=queried_settings['cent_freq']['max'])

                self.window['--FREQ_CORR_IN--'].update(value=queried_settings['freq_corr']['value'])
                self.window['--GAIN_VALUE--'].update(value=queried_settings['gains']['value'])
                self.window['--GAINS_DROPDOWN--'].update(values=queried_settings['gains']['values'])

                msg = "Got device settings. Center Freq: %d Hz, Sample Rate: %d, Freq Corr : %d, Gain: %d."%(queried_settings['cent_freq']['value'],
                                                                                                              queried_settings['sample_rate']['value'],
                                                                                                              queried_settings['freq_corr']['value'],
                                                                                                              queried_settings['gains']['value'])
                self.log_message(msg)
    def run(self):

        while True:
            start_time = time.time()
            events, values = self.window.read(timeout=100)
            print(events)

            end_time = time.time()

            # Handle Device Selection
            self.device_selection_handler(events, values)

            # Handle device de-init
            self.device_de_init_handler(events, values)

            # Handle slider events
            self.slider_change_handler(events, values)

            # Handle query device settings
            self.query_device_settings_handler(events, values)

            # Tune radio
            self.tune_device_settings(events, values)
            
            if events == 'Exit' or events == sg.WIN_CLOSED:
                if self.device is not None:
                    self.device.sdr.close()
                self.window.close()
                break
            
            self.window['--CONSOLE_TEXT--'].update(self.console_log_data)

            if self.device_selected:
                self.window['--RADIO_DEINIT_BUTON--'].update(button_color=("white", "green"))
                self.change_widget_state(self.widgets_state_to_change, disabled=False)
            else:
                self.window['--RADIO_DEINIT_BUTON--'].update(button_color=("white", "red"))
                self.change_widget_state(self.widgets_state_to_change, disabled=True)
            #print("FPS %.2f\r"%(1/(end_time-start_time)))
        




if __name__ == "__main__":
    app = ASK_RECV()
    app.run()