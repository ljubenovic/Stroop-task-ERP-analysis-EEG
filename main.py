"""
main.py

This is the main script for analyzing N200 event-related potentials (ERPs) from EEG and fNIRS data.

The dataset used in this project was obtained from the following source:

Chen, Z., Gao, C., Li, T. et al. (2023). Open access dataset integrating EEG and fNIRS during Stroop tasks. 
Scientific Data, 10, 618. 
Available at: https://doi.org/10.1038/s41597-023-02524-1

The dataset contains EEG and fNIRS data recorded during Stroop tasks.

The script leverages hyperparameters defined in the `hyperparameters.py` script to ensure consistent and customizable processing.

Functionality:
--------------
1. Processes raw EEG data.
2. Processes raw EOG data.
3. Removes blink-related artifacts using the Picard ICA algorithm.
4. Extracts and analyzes N200 event-related potentials.

Notes:
------
- Ensure that the raw EEG data is preloaded and in the correct format before running the script.
- Customize the behavior of the pipeline by modifying hyperparameters.py script.
"""

from pipelines import *
import hyperparameters as hp

if __name__ == '__main__':

    for i in range(0,1):

        S_id = i+1

        if S_id == 5:
            # There's a problem with the raw EEG data for the subject with ID 5
            continue
        
        #eeg_preprocessing_pipeline(S_id, hp.reference_channel, hp.cutoff_low_eeg, hp.cutoff_high_eeg, hp.filt_order)

        #blink_epochs, blink_peaks = eog_processing_pipeline(S_id, hp.cutoff_low_eog, hp.cutoff_high_eog, hp.numtaps, hp.window, hp.blink_epoch_duration)
        #ica_denoising_pipeline(S_id, blink_epochs, blink_peaks, hp.ortho, hp.extended, hp.blink_epoch_duration, hp.threshold_z)

        erp_extraction_pipeline(S_id, hp.epoch_duration, hp.pre_stim_duration, hp.time_window, hp.focused_channel, hp.relevant_channels)
    