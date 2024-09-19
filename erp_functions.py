import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.signal import find_peaks

def segment_into_epoch(eeg, fs, event_type, event_latencies, epoch_duration, pre_stim_duration):

    #event_type = np.concatenate((np.ones(32,),2*np.ones(32,)),axis=-1)

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


def epoch_averaging(epochs):

    # epochs: (n_chan, epoch_duration, n_epochs)

    epoch_avg = np.mean(epochs, axis=-1)

    return epoch_avg


def n200_extraction(epoch, fs, time_window):

    time_window = (int(time_window[0]/1000*fs), int(time_window[1]/1000*fs))

    epoch_mask = np.zeros_like(epoch)
    epoch_mask[time_window[0]:time_window[1]] = np.ones((time_window[1]-time_window[0],))

    epoch_masked = np.where(epoch_mask, epoch, np.nan)

    min_idx = np.nanargmin(epoch_masked)

    n200_amp = np.abs(epoch[min_idx])   # absolute value of N200 peak amplitude
    n200_latency = min_idx/fs*1000      # ms

    return (n200_amp, n200_latency)

    
def erp_extraction_per_channel(epochs, fs, chanlocs, S_id, type, time_window, channel = 'FZ'):

    (n_chan, _) = epochs.shape

    csv_path = os.path.join('data','N200','S{}_{}.csv'.format(S_id,type))
    if os.path.exists(csv_path):
        os.remove(csv_path)

    with open(csv_path, mode='w', newline='') as file:
        
        writer = csv.writer(file)
        writer.writerow(['channel', 'N200 amplitude', 'N200 latency'])  # header

        for i in range(n_chan):

            ch = chanlocs[i]

            epoch_chan = epochs[i,:]

            n200_amp, n200_latency = n200_extraction(epoch_chan, fs, time_window)
            writer.writerow([ch, n200_amp, n200_latency])

            if ch == channel and n200_amp:

                print('S_id: '+ str(S_id))
                print('-' + type + ' stimulus-')
                print('channel {}:'.format(channel))
                print('\tN200 amplitude: {:.1f}'.format(n200_amp))
                print('\tN200 latency: {:.0f} ms'.format(n200_latency))

    return


def erp_extraxtion_averaged_channels(epochs, fs, chanlocs, S_id, type, time_window, relevant_channels):

    (_, n_samp) = epochs.shape

    epochs_relevant = []
    for ch in relevant_channels:
        if ch in chanlocs:
            idx = chanlocs.index(ch)
            epochs_relevant.append(epochs[idx,:])

    epochs_relevant = np.reshape(epochs_relevant, newshape=(len(epochs_relevant), len(epochs_relevant[0])))
    
    epoch_avg = np.mean(epochs_relevant, axis=0)

    time_window = (int(time_window[0]/1000*fs), int(time_window[1]/1000*fs))
    
    (n200_amp, n200_latency) = n200_extraction(epoch_avg, fs, time_window)

    t = 1000*np.arange(0, n_samp/fs, 1/fs)

    plt.figure(figsize=(10,4))
    plt.plot(t, epoch_avg)
    plt.title('Averaged epoches across the channels: ' + ", ".join(relevant_channels) + 
              "\n N200: latency = {:.0f} ms, amplitude = {:.1f} a.u.".format(n200_latency, n200_amp))
    plt.xlabel('t [ms]'); plt.ylabel('amplitude [a.u.]')
    plt.grid(which='both')
    plt.xlim(time_window)
    
    path = os.path.join('data','Graphs','S{}'.format(S_id),'N200_avg_{}.png'.format(type))
    plt.savefig(path)
    plt.close()

    return (n200_amp, n200_latency)


def single_trial_erp_analysis(epochs, fs, chanlocs, S_id, type, time_window, channel):

    (n_ch, n_samp, n_stim) = epochs.shape

    ch_idx = chanlocs.index(channel)
    epochs_ch = epochs[ch_idx,:,:]
    
    n200_amp_arr = []
    n200_latency_arr = []
    for i in range(n_stim):
        epoch = epochs_ch[:,i]

        (n200_amp, n200_latency) = n200_extraction(epoch, fs, time_window)
        n200_amp_arr.append(n200_amp)
        n200_latency_arr.append(n200_latency)
    
    n200_amp_arr = np.array(n200_amp_arr)
    n200_latency_arr = np.array(n200_latency)

    plt.figure()


    return

















