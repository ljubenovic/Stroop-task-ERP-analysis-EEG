import numpy as np
from scipy.io import loadmat, savemat
import os


############################################################################################################
def load_mat(S_id, name, dir_path):

    '''Učitava podatke sačuvane u .mat formatu'''

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
        event_type = np.squeeze(np.array([events[0]]))
        event_latency = [int(x[0,0]) for x in np.squeeze(np.array([events[1]]))]
        urevent = np.squeeze(np.array([events[2]]))

    if len(event_latency) > 64:
        # For some subjects, the endpoint of EEG recording is saved as the last event
        event_latency = event_latency[:64]
        event_type = event_type[:64]
        urevent = urevent[:64]

    (n_chan, n_samp) = signal.shape
    t = np.arange(0, n_samp/fs, 1/fs)

    print('\nS{}\n------'.format(S_id))
    print(name+'\n------\nfs: {} Hz\nduration: {:.3f} s\nn_channels: {}\nchannels: {}\n'.format(fs, t[-1], n_chan, chanlocs))

    return (signal, fs, chanlocs, event_type, event_latency, urevent)

############################################################################################################
def save_to_mat(data, fs, chanlocs, event_type, event_latency, urevent, S_id, name, dir_path):

    '''Čuva podatke u .mat format'''

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

############################################################################################################
def save_to_csv(data, S_id, filename, dir_path):

    '''Čuva podatke u .csv format'''

    if S_id:
        path = os.path.join(dir_path, 'S{}_'.format(S_id) + filename + '.csv')
    else:
        path = os.path.join(dir_path, filename + '.csv')
    if os.path.exists(path):
        os.remove(path)

    np.savetxt(path, data, delimiter=',')

############################################################################################################
def load_csv(S_id, filename, dir_path):

    '''Učitava podatke iz .csv formata'''

    if S_id:
        path = os.path.join(dir_path, 'S{}_'.format(S_id) + filename + '.csv')
    else:
        path = os.path.join(dir_path, filename + '.csv')
    data = np.loadtxt(path, delimiter=',')
    return data


