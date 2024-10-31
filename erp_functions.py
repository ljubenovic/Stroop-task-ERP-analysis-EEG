import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.signal import find_peaks
from utility_functions import *

############################################################################################################
def segment_into_epoch(eeg, fs, event_latencies, epoch_duration, pre_stim_duration):

    event_type = np.concatenate((np.ones(32,),2*np.ones(32,)),axis=-1)

    epoch_duration_samp = int(epoch_duration/1000*fs)
    pre_stim_duration_samp = int(pre_stim_duration/1000*fs)
 
    n_stim = len(event_latencies)

    epochs_1 = []
    epochs_2 = []

    for i in range(n_stim):
        
        epoch_start = np.max([0,event_latencies[i] - pre_stim_duration_samp])

        epoch_end = np.min([epoch_start + epoch_duration_samp, eeg.shape[1]])
        
        epoch = eeg[:, epoch_start : epoch_end]

        if epoch.shape[1] == epoch_duration_samp:

            if event_type[i] == 1:  # neutral stimulus
                epochs_1.append(epoch)
            elif event_type[i] == 2:   # incogruent stimulus
                epochs_2.append(epoch)

    epochs_1 = np.stack(epochs_1, axis=-1)
    epochs_2 = np.stack(epochs_2, axis=-1)

    return epochs_1, epochs_2

############################################################################################################
def epoch_baseline_correction(epochs, fs, baseline_duration):

    (n_chan, epoch_len, n_epochs) = epochs.shape

    baseline_len = int(baseline_duration/1000*fs)

    epoch_len_new = epoch_len - baseline_len
    epochs_new = np.zeros((n_chan, epoch_len_new, n_epochs))

    for i in range(n_epochs):

        epoch = epochs[:,:,i]

        baseline = epoch[:, :baseline_len, np.newaxis]
        baseline_mean = np.mean(baseline, axis=1)

        epoch_new = epoch - baseline_mean
        epoch_new = epoch_new[:,baseline_len:]

        epochs_new[:,:,i] = epoch_new

    return epochs_new
    #return epochs[:,baseline_len:] # without baseline correction

############################################################################################################
def epoch_averaging(epochs):

    # epochs: (n_chan, epoch_duration, n_epochs)

    epoch_avg = np.mean(epochs, axis=-1)

    return epoch_avg

############################################################################################################
def epoch_aligned_averaging(epochs, fs, time_window, n_samp):

    (n_chan, epoch_duration, n_epochs) = epochs.shape

    amps = np.zeros((n_chan, n_epochs))
    lats = np.zeros((n_chan, n_epochs))

    epochs_clipped = np.zeros((n_chan, n_samp, n_epochs))

    for i in range(n_epochs):

        for j in range(n_chan):

            (amp, lat) = erp_extraction(epochs[j,:,i], fs, time_window)

            amps[j,i] = np.round(amp,3)
            lats[j,i] = lat

            epochs_clipped[j,:,i] = epochs[j, int(lat-n_samp//2) : int(lat+n_samp//2), i]

    epochs_aligned_avg = epoch_averaging(epochs_clipped)

    return epochs_aligned_avg, amps, lats

############################################################################################################
def erp_extraction(epoch, fs, time_window):

    time_window = (int(time_window[0]/1000*fs), int(time_window[1]/1000*fs))

    peaks, _ = find_peaks(-epoch[time_window[0]:time_window[1]])
    peaks = peaks + time_window[0]
    if np.any(peaks):
        peak_idx = np.argmin(epoch[peaks])
        peak = peaks[peak_idx]

        erp_amp = np.abs(epoch[peak])   # absolute value of N200 peak amplitude
        #erp_amp = np.mean(epoch[time_window[0]:time_window[1]])
        erp_latency = peak/fs*1000      # ms
    else:
        erp_amp = np.nan
        erp_latency = np.nan

    return (np.round(erp_amp,3), erp_latency)

############################################################################################################  
def erp_extraction_all_channels(epochs, fs, time_window):

    n_chan, _ = epochs.shape
    amps = np.zeros((n_chan,))
    lats = np.zeros((n_chan,))

    for i in range(n_chan):
        epoch = epochs[i,:]
        amp, lat = erp_extraction(epoch, fs, time_window)
        amps[i] = amp
        lats[i] = lat
    
    return (amps, lats)

############################################################################################################
def erp_extraction_per_channel(epochs, fs, chanlocs, S_id, type, time_window, channel = 'FZ'):

    (n_chan, _) = epochs.shape

    csv_path = os.path.join('data','N200','S{}_N200_{}.csv'.format(S_id,type))
    if os.path.exists(csv_path):
        os.remove(csv_path)

    with open(csv_path, mode='w', newline='') as file:
        
        writer = csv.writer(file)
        writer.writerow(['channel', 'N200 amplitude', 'N200 latency'])  # header

        for i in range(n_chan):

            ch = chanlocs[i]

            epoch_chan = epochs[i,:]

            n200_amp, n200_latency = erp_extraction(epoch_chan, fs, time_window)
            writer.writerow([ch, n200_amp, n200_latency])

            if ch == channel and n200_amp:

                amp = n200_amp
                lat = n200_latency

                print('S_id: '+ str(S_id))
                print('-' + type + ' stimulus-')
                print('channel {}:'.format(channel))
                print('\tN200 amplitude: {:.1f}'.format(n200_amp))
                print('\tN200 latency: {:.0f} ms'.format(n200_latency))

    return (amp, lat)

############################################################################################################
def single_trial_erp_analysis(epochs, fs, chanlocs, time_window, channel):

    (n_ch, n_samp, n_stim) = epochs.shape

    ch_idx = chanlocs.index(channel)
    epochs_ch = epochs[ch_idx,:,:]
    
    n200_amp_arr = []
    n200_latency_arr = []
    for i in range(n_stim):
        epoch = epochs_ch[:,i]

        (n200_amp, n200_latency) = erp_extraction(epoch, fs, time_window)
        n200_amp_arr.append(n200_amp)
        n200_latency_arr.append(n200_latency)
    
    n200_amp_arr = np.array(n200_amp_arr)
    n200_latency_arr = np.array(n200_latency_arr)

    return (n200_amp_arr, n200_latency_arr)

