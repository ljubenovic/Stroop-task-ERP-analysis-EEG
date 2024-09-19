import numpy as np

def rereference_eeg(eeg, chanlocs, reference_chan='CZ'):

    if reference_chan not in chanlocs:
        raise ValueError(f"Reference channel '{reference_chan}' not found in channel locations.")
    
    chanlocs_list = chanlocs.tolist()
    ref_index = chanlocs_list.index(reference_chan)

    eeg_reref = eeg - eeg[ref_index, :]
    eeg_reref = eeg_reref[chanlocs != reference_chan,:]
    chanlocs = chanlocs[chanlocs != reference_chan]

    return eeg_reref, chanlocs


def exclude_non_eeg_channels(eeg, chanlocs):

    eog_channels = ['HEO', 'VEO']
    mastoid_channels = ['M1', 'M2']

    eeg_channels = [1 if chan not in eog_channels + mastoid_channels else 0 for chan in chanlocs]
    eeg_new = np.squeeze(eeg[np.where(eeg_channels),:])
    chanlocs_new = chanlocs[np.where(eeg_channels)]

    eog_channels = [1 if chan in eog_channels else 0 for chan in chanlocs]
    eog = np.squeeze(eeg[np.where(eog_channels),:])
    eog_chanlocs = chanlocs[np.where(eog_channels)]

    mastoid_channels = [1 if chan in mastoid_channels else 0 for chan in chanlocs]
    mastoid = np.squeeze(eeg[np.where(mastoid_channels),:])
    mastoid_chanlocs = chanlocs[np.where(mastoid_channels)]

    return eeg_new, chanlocs_new, eog, eog_chanlocs, mastoid, mastoid_chanlocs


def remove_irrelevant_segments(eeg, fs, event_latency):

    n_stim = len(event_latency)
    n_block_stim = 16

    pre_stim_len = int(2*fs)
    post_stim_len = int(2*fs)
    
    event_latency_new = np.zeros((n_stim,), dtype=np.int64)

    for i in range(n_stim):
        
        if i % n_block_stim == 0:
            eeg_block = eeg[:, event_latency[i] - pre_stim_len : event_latency[i+1]]   # eeg block start: 2 s before the first stimulus in block
            next_latency_shift = eeg_block.shape[1] - pre_stim_len

        elif i % n_block_stim == n_block_stim-1:
            eeg_block = eeg[:, event_latency[i] : event_latency[i]+post_stim_len]    # eeg block end: 2 s after the last stimulus in block
            next_latency_shift = post_stim_len + pre_stim_len

        else:
            eeg_block = eeg[:, event_latency[i] : event_latency[i+1]]
            next_latency_shift = eeg_block.shape[1]
            
        if i == 0:
            eeg_new = eeg_block
            event_latency_new[i] = pre_stim_len
        else:
            eeg_new = np.concatenate((eeg_new, eeg_block),axis=1)
            event_latency_new[i] = event_latency_new[i-1] + latency_shift

        latency_shift = next_latency_shift

    return eeg_new, event_latency_new

