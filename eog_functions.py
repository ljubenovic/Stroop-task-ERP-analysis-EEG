import os
import csv
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import zscore, pearsonr

############################################################################################################
def detect_blinks(eog, fs, epoch_duration, percentile=1):
    """
    Detect significant blink-related peaks in the multi-channel EOG signal.

    Parameters:
    - eog: Input EOG signal (2D numpy array, shape: n_channels x n_samples)
    - fs: Sampling frequency (in Hz)
    - epoch_duration: Duration of the epoch around each peak (in milliseconds, default is 500 ms)
    - percentile: Percentile value to reject extreme outliers (default is 5)

    Returns:
    - blink_epochs: List of lists where each sublist contains EOG signal fragments (epochs) around detected peaks for each channel
    - blink_peaks: List of lists where each sublist contains indices of detected peaks for each channel
    """

    n_channels, n_samples = eog.shape
    epoch_samples = int(fs * (epoch_duration / 1000.0))
    blink_epochs = []
    blink_peaks = []

    for channel in range(n_channels):
        signal = eog[channel]
        
        # Determine the threshold for peak detection
        threshold = (np.quantile(signal, 0.99) - np.quantile(signal, 0.01)) / 4

        # Find peaks with the specified threshold and minimum distance
        peaks, _ = find_peaks(np.abs(signal), height=threshold, distance=fs * 0.2)  # Minimum distance of 200 ms between neighboring peaks

        # Extract epochs around detected peaks
        channel_epochs = [signal[max(0, peak - epoch_samples // 2): min(n_samples, peak + epoch_samples // 2)]
                          for peak in peaks]
        
        blink_peaks.append(peaks)
        blink_epochs.append(channel_epochs)

    return blink_epochs, blink_peaks

############################################################################################################
def find_blink_related_components(S_id, components, blink_epochs, blink_peaks, blink_epoch_duration, threshold_z=2):
    """
    Reject independent components highly correlated with blinks.
    
    Parameters:
    - S_id: 
    - components: Matrix of independent components (n_components x n_samples)
    - blink_epochs:
    - blink_peaks:
    - threshold_z: Z-score threshold to detect blink-related components (default = 2)
    
    Returns:
    - artifact_indices: Indices of components identified as blink-related
    """

    csv_file = os.path.join('data','ICA components','S{}_components_to_reject.csv'.format(S_id))
    if os.path.exists(csv_file):
        os.remove(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
    
        n_comp, n_samp = components.shape

        artifact_indices = []

        writer.writerow(['comp idx','average Pearson corr'])

        for ch in range(len(blink_peaks)):

            writer.writerow(['EOG ch {}:'.format(ch)])

            blink_peaks_ch = blink_peaks[ch]
            blink_epochs_ch = blink_epochs[ch]
            
            n_blinks = len(blink_peaks_ch)

            # Compute the correlation between each independent component and the blink signal
            correlations = []
            for i in range(n_comp):

                component = components[i,:]
                
                corr_arr = []
                for j in range(n_blinks):

                    peak = blink_peaks_ch[j]
                    blink_epoch = blink_epochs_ch[j]
                    
                    epoch_samp = blink_epoch_duration
                    comp_epoch = component[max(0, peak - epoch_samp // 2): min(n_samp, peak + epoch_samp // 2 + epoch_samp % 2)]

                    corr, p_val = pearsonr(np.abs(blink_epoch), np.abs(comp_epoch))
                    corr_arr.append(corr)

                corr_mean = np.mean(np.abs(corr_arr))
                correlations.append(corr_mean)

            correlations = np.array(correlations)

            # Iteratively reject components with a correlation higher than the Z-score threshold
            while True:
                z_scores = zscore(correlations)
                high_corr_idx = np.where(z_scores > threshold_z)[0]

                # Stop if no more blink-related components are detected
                if len(high_corr_idx) == 0:
                    break

                # Add the indices of the high-correlation components to the artifact list
                artifact_indices.extend(high_corr_idx.astype(int))

                for idx in high_corr_idx:
                    writer.writerow([idx,correlations[idx]])

                # Set those components' correlations to zero for the next iteration
                correlations[high_corr_idx] = 0

    return artifact_indices
