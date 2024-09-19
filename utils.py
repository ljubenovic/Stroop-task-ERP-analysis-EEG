import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import os

def load_mat(S_id, name, dir_path):

    path = os.path.join(dir_path, 'S' + str(S_id) + '_' + name +'.mat')
    data = loadmat(path)

    data = data[name]

    signal = np.array(data['data'][0,0])
    fs = data['srate'][0,0][0,0]

    if data['event'][0,0].shape[0] == 1 and data['event'][0,0].shape[1] != 3:    # Raw EEG files

        chanlocs = np.squeeze(np.array([chan[0] for chan in data['chanlocs'][0,0][0,:]]).T)

        events = data['event'][0,0][0,:]

        event_type = np.squeeze(np.array([event[0] for event in events]))
        event_latency = np.squeeze(np.array([event[1] for event in events]))
        urevent = np.squeeze(np.array([event[2] for event in events]))

    else:   # Processed EEG data and EOG data
        chanlocs = data['chanlocs'][0,0][0]
        chanlocs = [ch[:3] if ch[2].isalnum() else ch[:2] for ch in chanlocs]

        events = data['event'][0,0]
        print(events.shape)
        event_type = np.squeeze(np.array([events[0]]))
        event_latency = [int(x[0,0]) for x in np.squeeze(np.array([events[1]]))]
        urevent = np.squeeze(np.array([events[2]]))

    (n_chan, n_samp) = signal.shape
    t = np.arange(0, n_samp/fs, 1/fs)

    print('\nS{}\n------'.format(S_id))
    print(name+'\n------\nfs: {} Hz\nduration: {:.3f} s\nn_channels: {}\nchannels: {}\n'.format(fs, t[-1], n_chan, chanlocs))

    return (signal, fs, chanlocs, event_type, event_latency, urevent)


from scipy.io import savemat

def save_to_mat(data, fs, chanlocs, event_type, event_latency, urevent, S_id, name, dir_path):
    """
    Save the EEG data and metadata to a .mat file.

    Parameters:
    - eeg: EEG data (2D numpy array, shape: n_channels x n_samples)
    - fs: Sampling frequency (in Hz)
    - chanlocs: Channel locations (2D numpy array, shape: n_channels x n_features)
    - event_type: Event types (1D numpy array)
    - event_latency: Event latencies (1D numpy array)
    - urevent: Event unique identifiers (1D numpy array)
    - file_path: Path to the .mat file where data will be saved
    """

    # Prepare data for saving
    data_to_save = {
        'data': data,
        'srate': np.array([[fs]]),
        'chanlocs': np.expand_dims(chanlocs, axis=0),
        'event': np.array([event_type, event_latency, urevent], dtype=object)
    }

    event = np.array([event_type, event_latency, urevent], dtype=object)

    file_path = os.path.join(dir_path,'S' + str(S_id) + '_' + name + '.mat')
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save to .mat file
    savemat(file_path, {name: data_to_save})


def plot_multichannel(data, fs, chanlocs, event_type, event_latency, S_id, filename=None, time_window=None):

    (n_chan, n_samp) = data.shape    # (34, 243080)
    t = np.arange(0, n_samp/fs, 1/fs)
    if np.squeeze(t.shape) > data.shape[1]:
        t = t[:data.shape[1]]

    if n_chan <= 3:
        plt.figure(figsize=(10,8))
    else:
        plt.figure(figsize=(10,50))
    for i in range(n_chan):
        plt.subplot(n_chan,1,i+1)
        plt.plot(t,data[i,:], linewidth=0.8)
        if np.any(event_latency):
            for j in range(len(event_latency)):
                if event_type[j] == 1:
                    plt.axvline(x=t[event_latency[j]], color='g', alpha=0.3)
                else:
                    plt.axvline(x=t[event_latency[j]], color='r', alpha=0.3)
        plt.xlabel('t [s]')
        plt.ylabel('amplituda [a.u.]')
        plt.title('ch: {}'.format(chanlocs[i]))
        if time_window:

            time_window_s = (time_window[0]/1000, time_window[1]/1000)
            time_window_samp = (int(time_window_s[0]*fs), int(time_window_s[1]*fs))

            plt.xlim(time_window_s)
            plt.ylim((np.min(data[i,time_window_samp[0]:time_window_samp[1]]),
                      np.max(data[i,time_window_samp[0]:time_window_samp[1]])))
        plt.grid(True, which='both', alpha=0.5)
    plt.tight_layout()

    if filename:
        path = os.path.join("data","Graphs", "S{}".format(S_id), filename + ".png")
        plt.savefig(path)

    plt.close()

    return


def plot_multichannel_with_peaks(data, peaks, fs, chanlocs, S_id, filename=None):

    (n_chan, n_samp) = data.shape    # (34, 243080)
    t = np.arange(0, n_samp/fs, 1/fs)
    if np.squeeze(t.shape) > data.shape[1]:
        t = t[:data.shape[1]]

    if n_chan <= 3:
        plt.figure(figsize=(10,8))
    else:
        plt.figure(figsize=(10,50))
    for i in range(n_chan):
        plt.subplot(n_chan,1,i+1)
        plt.plot(t,data[i,:], linewidth=0.8)
        plt.scatter(t[peaks[i]],data[i,peaks[i]], c='r', edgecolors='r')
        plt.xlabel('t [s]')
        plt.ylabel('amplituda [a.u.]')
        plt.title('ch: {}'.format(chanlocs[i]))
        plt.grid(True, which='both', alpha=0.5)
    plt.tight_layout()

    if filename:
        path = os.path.join("data","Graphs", "S{}".format(S_id), filename + ".png")
        plt.savefig(path)

    plt.close()

    return


def plot_multichannel_fft(data, fs, chanlocs, S_id, filename=None):

    (n_chan, n_samp) = data.shape    # (34, 243080)
    freqs = np.fft.fftfreq(n_samp, 1/fs)[:n_samp // 2]

    if n_chan <= 3:
        plt.figure(figsize=(10,8))
    else:
        plt.figure(figsize=(10,50))
    for i in range(n_chan):
        signal = data[i,:]
        fft_signal = np.fft.fft(signal)
        fft_signal = np.abs(fft_signal[:n_samp // 2])

        plt.subplot(n_chan,1,i+1)
        plt.plot(freqs, fft_signal, linewidth=0.8)
        plt.xlabel('f [Hz]')
        plt.ylabel('magnituda [a.u.]')
        plt.title('Amplitudski spektar (ch: {})'.format(chanlocs[i]))
        plt.grid(True, which='both', alpha=0.5)
        plt.xlim((0,50))
    plt.tight_layout()

    if filename:
        path = os.path.join("data","Graphs", "S{}".format(S_id), filename + ".png")
        plt.savefig(path)

    plt.close()

    return


def save_to_csv(data, S_id, filename, dir_path):

    path = os.path.join(dir_path, 'S{}_'.format(S_id) + filename + '.csv')
    if os.path.exists(path):
        os.remove(path)

    np.savetxt(path, data, delimiter=',')


def load_csv(S_id, filename, dir_path):

    path = os.path.join(dir_path, 'S{}_'.format(S_id) + filename + '.csv')
    data = np.loadtxt(path, delimiter=',')
    return data