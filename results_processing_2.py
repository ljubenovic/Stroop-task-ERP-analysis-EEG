from utility_functions import *
from erp_functions import *
from visualisation_functions import *
import hyperparameters as hp
from peak_analysis import paired_t_test
import pandas as pd


if __name__ == '__main__':


    # Adjust ERP component and time window: 
    type = '200'
    time_window = (100,260)

    fs = 1000
    chanlocs = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3','FCZ',
                'FC4', 'FT8', 'T7', 'C3', 'C4', 'T8', 'TP7', 'CP3','CPZ', 'CP4',
                'TP8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']
    
    n_chan = len(chanlocs)
    n_sub = 20
    n_samp = 200

    epoch_duration = hp.epoch_duration - hp.pre_stim_duration

    ### Loading epochs from .csv

    # Epochs averaged without previous aligning to the peak
    global_avgs_1_whole = load_csv(0, 'global_avgs_1', os.path.join('data','N200'))
    global_avgs_2_whole = load_csv(0, 'global_avgs_2', os.path.join('data','N200'))

    # Epochs averaged independently for each subject
    epochs_1 = np.zeros((n_chan, epoch_duration, n_sub))
    epochs_2 = np.zeros((n_chan, epoch_duration, n_sub))
    S_ids = []
    for i in range(n_sub):
        if i < 4:
            S_id = i+1
        else:
            S_id = i+2
        S_ids.append(S_id)
        epoch_1 = load_csv(S_id, 'epoch_avg_1'.format(S_id), os.path.join('data','N200'))
        epoch_2 = load_csv(S_id, 'epoch_avg_2'.format(S_id), os.path.join('data','N200'))
        epochs_1[:,:,i] = epoch_1
        epochs_2[:,:,i] = epoch_2

    ### Epochs averaging with aligning to the ERP peak

    global_avgs_1, amps_1, lats_1 = epoch_aligned_averaging(epochs_1, fs, time_window, n_samp)
    global_avgs_2, amps_2, lats_2 = epoch_aligned_averaging(epochs_2, fs, time_window, n_samp)

    # Visualisation
    plot_avg_epochs_all_channels(global_avgs_1, global_avgs_2, fs, chanlocs, 0, filename='global_averages_{}_aligned'.format(type))
    plot_avg_epochs_all_channels(global_avgs_1_whole, global_avgs_2_whole, fs, chanlocs, 0, filename='global_averages_without_aligning')

    ### Paired t-test

    p_vals = np.zeros((n_chan,))
    for i in range(n_chan):
        p_vals[i] = paired_t_test(amps_1[i,:], amps_2[i,:], chanlocs[i])

    p_df = pd.DataFrame(np.array(p_vals).T, columns=['p_val'], index=chanlocs)
    plot_table(p_df, 'p_values_df_n{}'.format(type), 1)

    indicies = np.where(p_vals < 0.05)[0]
    for idx in indicies:
        channel = chanlocs[idx]
        plot_samples([amps_1[idx,:], amps_2[idx,:]], [lats_1[idx,:], lats_2[idx,:]], channel, type)
        plot_boxplots([[amps_1[idx,:], amps_2[idx,:]]], ['neutral', 'incongruent'], ['amplitude [a.u.]'], [''], 'n{}_amps_{}'.format(type,channel))

    ### ERP amplitude extraction (with aligning)

    (amps_1, _) = erp_extraction_all_channels(global_avgs_1, fs, (0,n_samp))
    (amps_2, _) = erp_extraction_all_channels(global_avgs_2, fs, (0,n_samp))
    amps = np.array([amps_1, amps_2]).T
    print(amps.shape)

    amps_df = pd.DataFrame(amps, columns=['neutral','incongruent'], index=chanlocs)
    plot_table(amps_df, 'n{}_abs_amps'.format(type), 2)

    ### ERP amplitude and latency extraction (without aligning)

    (amps_1, lats_1) = erp_extraction_all_channels(global_avgs_1_whole, fs, time_window)
    (amps_2, lats_2) = erp_extraction_all_channels(global_avgs_2_whole, fs, time_window)
    amps = np.array([amps_1, amps_2]).T
    lats = np.array([lats_1, lats_2]).T

    amps_df = pd.DataFrame(amps, columns=['neutral','incongruent'], index=chanlocs)
    plot_table(amps_df, 'n{}_abs_amps_whole'.format(type), 2)

    lats_df = pd.DataFrame(lats, columns=['neutral','incongruent'], index=chanlocs)
    plot_table(lats_df, 'n{}_lats_whole'.format(type), 2)

    