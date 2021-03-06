import pickle
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from WiPHY.utils import search_sequence_cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

raw_samples_layout = [[sg.Canvas(key='-raw-canvas-')]]
filtered_samples_layout = [[sg.Canvas(key='-filtered-canvas-')]]
time_sync_layoyt = [[sg.Canvas(key='-time-sync-canvas-')]]
time_sync_const = [[sg.Canvas(key='-time-sync-const-canvas-')]]
baseband_sig_layout = [
                         [sg.Canvas(key='-baseband-sig-canvas-')],
                         [sg.Text("Extracted Frames: ")],
                         [sg.Multiline(size=(140, 10), key='--CONSOLE_TEXT--', autoscroll=True)]
                      ]

data = None

layout = [
    [sg.Text("Capture File: "), sg.Input(key='_CAPTURE_FILE_'), sg.FilesBrowse(), sg.Button("Load", key='_CAPTURE_FILE_LOAD_')],
    [sg.Multiline(size=(140, 7), key='--SETTINGS_TEXT--', autoscroll=True)],
    [sg.Text('Capture Number:'), sg.Combo(list(range(0, 3000)), key='--FRAME_NUM_VAL--'), sg.Button('Analyze', key='--FRAME_NUM--'), sg.Text('Valid capture indexs:'), sg.Combo(list(range(0, 3000)), key='--Valid-capture-indexs--' ) ],
    [sg.TabGroup(
        [
            [sg.Tab('Raw Samples', raw_samples_layout)],
            [sg.Tab('Low Pass filtered samples', filtered_samples_layout)],
            [sg.Tab('Time Sync Signal', time_sync_layoyt)],
            [sg.Tab('Time Sync Signal - Constellation', time_sync_const)],
            [sg.Tab('Base Band Signal', baseband_sig_layout)]
        ]
    )],
]

def plot_samples(frame_index):
    
    # Raw Captures
    raw_samples_fig = plt.figure(figsize=(10, 5), dpi=100)
    raw_axis_time = raw_samples_fig.add_subplot(2, 1, 1)
    raw_axis_freq = raw_samples_fig.add_subplot(2, 1, 2)
    raw_samples_fig.subplots_adjust(left=0.1, right=0.98, top=0.99, bottom=0.1)

    '''
        raw Time signal
    '''
    raw_samples = np.array(data['raw'][frame_index]).flatten()
    total_capture_time = raw_samples.shape[0]/data['sdr_settings']['sample_rate']
    time = np.linspace(0, total_capture_time, num=raw_samples.shape[0])
    raw_axis_time.plot(time, np.abs(raw_samples))
    raw_axis_time.set_xlabel("Time (seconds)")
    raw_axis_time.set_ylabel("Amplitude (V)")
    raw_axis_time.grid()

    '''
        raw PSD signal
    '''
    raw_axis_freq.clear()
    raw_axis_freq.psd(raw_samples, NFFT=raw_samples.shape[0], Fs=data['sdr_settings']['sample_rate']/1e6, Fc=data['sdr_settings']['center_freq']/1e6)
    raw_axis_freq.set_xlabel("Freq (MHz)")
    raw_axis_freq.set_ylabel("PSD (dB)")

    # Time Sync
    
    time_sync_fig = plt.figure(figsize=(10, 5), dpi=100)
    time_sync_axis = time_sync_fig.add_subplot(2, 1, 1)
    time_sync_raw_axis = time_sync_fig.add_subplot(2, 1, 2)
    time_sync_fig.subplots_adjust(left=0.1, right=0.98, top=0.99, bottom=0.1)

    time_sync_samples = np.array(data['time_sync'][frame_index]).flatten()
    time = np.linspace(0, total_capture_time, num=time_sync_samples.shape[0])
    time_sync_axis.plot(time, np.abs(time_sync_samples), color='orange', label="time sync sig")
    time_sync_axis.plot(time, np.abs(time_sync_samples), 'x', color='blue',label="time sync sig (sampled)")
    time_sync_axis.set_xlabel("Time (seconds)")
    time_sync_axis.set_ylabel("Amplitude (V)")
    time_sync_axis.grid()
    time_sync_axis.legend()

    # Raw signal
    time = np.linspace(0, total_capture_time, num=raw_samples.shape[0])
    time_sync_raw_axis.plot(time, np.abs(raw_samples))
    #time_sync_raw_axis.plot(time, np.abs(time_sync_samples), 'x', color='red',)
    time_sync_raw_axis.set_xlabel("Time (seconds)")
    time_sync_raw_axis.set_ylabel("Amplitude (v)")
    time_sync_raw_axis.grid()

    
    # Time Sync Constellation
    time_sync_const_fig = plt.figure(figsize=(10, 5), dpi=100)
    time_sync_const_axis = time_sync_const_fig.add_subplot(1, 1, 1)
    time_sync_const_fig.subplots_adjust(left=0.1, right=0.98, top=0.99, bottom=0.1)
    y = np.zeros(time_sync_samples.shape[0])
    time_sync_const_axis.scatter(np.abs(time_sync_samples), y,)
    time_sync_const_axis.grid()
    time_sync_const_axis.set_xlim([-2.5, 2.5])
    time_sync_const_axis.set_ylim([-2.5, 2.5])
    time_sync_const_axis.set_xlabel("In-Phase")
    time_sync_const_axis.set_ylabel("Quadrature")

    # Base Band Signal
    baseband_sig_fig = plt.figure(figsize=(10, 3), dpi=100)
    demod_signal_axis = baseband_sig_fig.add_subplot(1, 1, 1)
    baseband_sig_fig.subplots_adjust(left=0.1, right=0.98, top=0.99, bottom=0.1)
    baseband_sig_samples = np.array(data['ask_demod'][frame_index]).flatten()
    bar1 = demod_signal_axis.bar(np.arange(0, baseband_sig_samples.shape[0]), baseband_sig_samples, color='white', edgecolor='black', label='BaseBand Signal (Bits)')
    demod_signal_axis.set_ylim([0, 1.7])
    #demod_signal_axis.get_xaxis().set_visible(False)
    
    for rect in bar1:
        height = rect.get_height()
        demod_signal_axis.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

    nrz_signal = np.array(data['ask_demod'][frame_index]).flatten()
    sig_match_indexs = search_sequence_cv2(nrz_signal, np.array([1, 1, 1, 0, 1]))
    corr_sig = np.zeros(nrz_signal.shape)
    for index in sig_match_indexs:
        corr_sig[index] = 1
    
    demod_signal_axis.plot(corr_sig, label="Signal correlation with preamble")
    demod_signal_axis.legend()
    demod_signal_axis.grid()
    return (raw_samples_fig, time_sync_fig, time_sync_const_fig, baseband_sig_fig)

def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

window = sg.Window('Sample Captures Analyzer', layout, default_element_size=(40,40), finalize=True)  
frames_detected_capture_indxs = []

if __name__ == '__main__':
    
    raw_samples_fig = None
    time_sync_fig = None
    time_sync_const_fig = None
    baseband_sig_fig = None

    while True:
        
        event, values = window.read()
        
        if event == '--FRAME_NUM--' or event == '_CAPTURE_FILE_LOAD_':
            
            if event == '_CAPTURE_FILE_LOAD_':
                fileHandler = open(values['_CAPTURE_FILE_'], "rb")
                data = pickle.load(fileHandler)
                fileHandler.close()
                
                # Find capture indexs where the frames were detected.
                for capture_idx, frames in enumerate(data['frame_detection']):
                    if len(frames) > 0:
                        frames_detected_capture_indxs.append(capture_idx)
                
                window['--Valid-capture-indexs--'].update(values=frames_detected_capture_indxs)
                device_info_string = "Capture time stamp: %s.\n"%(data['cature_time_stamp'])
                device_info_string += "Downlink performance => FER: %.3f, Frames detected: %d, Failed Frames: %d. \n"%(data['rx_performance_metrics']['fer'],
                                                                                                                      data['rx_performance_metrics']['frames_detected'],
                                                                                                                      data['rx_performance_metrics']['failed_frames'])
                
                device_info_string += 'SDR Settings => Sample rate: %d, Center freq :%d, Gain: %s, Freq Corr: %d, Capture Len: %d.'%(data['sdr_settings']['sample_rate'],
                                                                                                                                     data['sdr_settings']['center_freq'],
                                                                                                                                     data['sdr_settings']['gain'],
                                                                                                                                     data['sdr_settings']['freq_corr'],
                                                                                                                                     len(data['raw'][0]))
                device_info_string += '\nBaseband Singal Info => Symbol duration %.6f seconds.'%(data['sdr_settings']['symbol_dur'])
                device_info_string += '\nTime Sync Block Info => Alpha: %.2f.'%(data['sdr_settings']['alpha'])
                device_info_string += '\nASK Demod Block Info => Thershold: %.2f.'%(data['sdr_settings']['decision_thershold'])
                device_info_string += '\nFrame Detector Block Info => Barker seq: %d, CRC Polynomial: %d.'%(data['sdr_settings']['barker_seq'],
                                                                                                            data['sdr_settings']['crc_polynomial'])
                window['--SETTINGS_TEXT--'].update(device_info_string)
                window['--FRAME_NUM_VAL--'].update(values=list(range(0, len(data['raw']) )))
            
            if event == '--FRAME_NUM--':
                if raw_samples_fig is not None:
                    delete_fig_agg(raw_samples_fig)
                    delete_fig_agg(time_sync_fig)
                    delete_fig_agg(time_sync_const_fig)
                    delete_fig_agg(baseband_sig_fig)

                raw_samples_fig_mat, time_sync_fig_mat, time_sync_const_fig_mat, baseband_sig_fig_mat = plot_samples(int(values['--FRAME_NUM_VAL--']))
                raw_samples_fig = draw_figure(window['-raw-canvas-'].TKCanvas, raw_samples_fig_mat)
                time_sync_fig = draw_figure(window['-time-sync-canvas-'].TKCanvas, time_sync_fig_mat)
                time_sync_const_fig = draw_figure(window['-time-sync-const-canvas-'].TKCanvas, time_sync_const_fig_mat)
                baseband_sig_fig = draw_figure(window['-baseband-sig-canvas-'].TKCanvas, baseband_sig_fig_mat)
                window.Refresh()
                # Update extraced frames
                frame_str = ''
                extracted_frames = data['frame_detection'][int(values['--FRAME_NUM_VAL--'])]
                for frame in extracted_frames:
                    frame_str += frame.get_frame_str() + "\n"
                window['--CONSOLE_TEXT--'].update(frame_str)
                
        if event == sg.WIN_CLOSED:           # always,  always give a way out!    
            break  