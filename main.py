"""
main.py

This is the main script for analyzing N200 and N400 event-related potentials (ERPs) from EEG data.

The dataset used in this project was obtained from the following source:

Chen, Z., Gao, C., Li, T. et al. (2023). Open access dataset integrating EEG and fNIRS during Stroop tasks. 
Scientific Data, 10, 618. 
Available at: https://doi.org/10.1038/s41597-023-02524-1

The dataset contains EEG and fNIRS data recorded during Stroop tasks.

The script utilizes hyperparameters defined in the `hyperparameters.py` script to ensure consistent and customizable processing.

Functionality:
--------------
1. Processes raw EEG data.
2. Processes raw EOG data.
3. Removes blink-related artifacts using the Picard ICA algorithm.
4. Extracts and analyzes N200 event-related potentials

ERP analysis is also performed in the following scripts:
- 'peak_analysis.py'
- 'results_processing.py'
- 'results_processing_2.py'

Notes:
------
- Ensure that the raw EEG data is preloaded and in the correct format before running the script.
- Customize the behavior of the pipeline by modifying hyperparameters.py script.
"""

from pipelines import *
from visualisation_functions import *
import hyperparameters as hp

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

if __name__ == '__main__':

    avgs_1 = np.zeros((29,900,20))
    avgs_2 = np.zeros((29,900,20))

    j = 0
    for i in range(0,21):

        S_id = i+1

        if S_id == 5:
            # There's a problem with the raw EEG data for the subject with ID 5 (signal is only 8 s long)
            continue
        
        #eeg_preprocessing_pipeline(S_id, hp.reference_channel, hp.cutoff_low_eeg, hp.cutoff_high_eeg, hp.filt_order)

        #blink_epochs, blink_peaks = eog_processing_pipeline(S_id, hp.cutoff_low_eog, hp.cutoff_high_eog, hp.numtaps, hp.window, hp.blink_epoch_duration)
        #ica_denoising_pipeline(S_id, blink_epochs, blink_peaks, hp.ortho, hp.extended, hp.blink_epoch_duration, hp.threshold_z)

        (epochs_avg_1, epochs_avg_2) = erp_extraction_pipeline(S_id, hp.epoch_duration, hp.pre_stim_duration, hp.time_window, hp.focused_channel, hp.relevant_channels)
        avgs_1[:,:,j] = epochs_avg_1
        avgs_2[:,:,j] = epochs_avg_2

        j += 1

    fs = 1000
    chanlocs = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3','FCZ',
                'FC4', 'FT8', 'T7', 'C3', 'C4', 'T8', 'TP7', 'CP3','CPZ', 'CP4',
                'TP8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']
    
    plot_epochs_with_average(avgs_1, avgs_2, fs, chanlocs, 'FC3', 0)
    plot_epochs_with_average(avgs_1, avgs_2, fs, chanlocs, 'C3', 0)

    global_avg_1 = epoch_averaging(avgs_1)
    global_avg_2 = epoch_averaging(avgs_2)

    save_to_csv(global_avg_1, 0, 'global_avgs_1', os.path.join('data','N200'))
    save_to_csv(global_avg_2, 0, 'global_avgs_2', os.path.join('data','N200'))

    plot_avg_epochs_on_subplots('FC3')
    plot_avg_epochs_on_subplots('C3')

