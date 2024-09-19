import numpy as np
from scipy.signal import butter, filtfilt, freqz, firwin


def butter_filter(data, fs, order = None, cutoff_low = None, cutoff_high = None):
    """
    Apply a Butterworth filter (high-pass, low-pass, or band-pass) to multi-channel data.

    Parameters:
    - data: Input data (2D numpy array, shape: n_channels x n_samples)
    - fs: Sampling frequency (in Hz)
    - order: Order of the filter (optional, automatically determined if not provided)
    - cutoff_low: Low cut-off frequency for high-pass or band-pass filter (in Hz, optional)
    - cutoff_high: High cut-off frequency for low-pass or band-pass filter (in Hz, optional)
    
    Returns:
    - data_filt: Filtered data (same shape as input)
    - freq: Frequency response of the filter (for visualization, in Hz)
    - h: Amplitude response of the filter (for visualization)
    """

    f_nyquist = fs / 2  # Nyquist frequency (half of the sampling rate)
    min_order = 3       # Minimum filter order to ensure adequate filtering
    
    # Automatically determine the filter order if not provided
    if not order:
        minfac = 3  # Minimum factor to ensure enough cutoff-freq cycles in the filter
        
        if cutoff_low:
            order = minfac * int(fs / cutoff_low)
        elif cutoff_high:
            order = minfac * int(fs / cutoff_high)
        else:
            raise ValueError('Filter cutoff frequencies are not specified!')
    
    # Ensure the filter order is at least the minimum value
    if order < min_order:
        order = min_order

    # Design the appropriate filter based on the provided cutoff frequencies
    if cutoff_low and cutoff_high:
        # Band-pass filter (both low and high cut-off frequencies specified)
        cutoff_low_norm = cutoff_low / f_nyquist
        cutoff_high_norm = cutoff_high / f_nyquist
        b, a = butter(order, [cutoff_low_norm, cutoff_high_norm], btype='bandpass', analog=False)

    elif cutoff_low:
        # High-pass filter (only low cut-off frequency specified)
        cutoff_low_norm = cutoff_low / f_nyquist
        b, a = butter(order, cutoff_low_norm, btype='highpass', analog=False)

    elif cutoff_high:
        # Low-pass filter (only high cut-off frequency specified)
        cutoff_high_norm = cutoff_high / f_nyquist
        b, a = butter(order, cutoff_high_norm, btype='lowpass', analog=False)
    else:
        raise ValueError('Either lowcut or highcut must be specified!')

    # Apply zero-phase forward-backward channel-wise filtering to avoid phase distortion
    data_filt = filtfilt(b, a, data, axis=1)

    # Compute the frequency response of the filter
    freq, h = freqz(b, a, worN= 8000, fs=fs)
    
    return data_filt, freq, h


def bandpass_fir_filter(data, fs, cutoff_low, cutoff_high, numtaps=101, window='hann'):
    """
    Apply a bandpass filter to a multi-channel signal using an FIR filter with a specified window.

    Parameters:
    - data: Input signal (2D numpy array, shape: n_channels x n_samples)
    - fs: Sampling frequency (in Hz)
    - lowcut: Low cut-off frequency for the filter (in Hz)
    - highcut: High cut-off frequency for the filter (in Hz)
    - numtaps: Number of filter taps (default 101, higher value = sharper filter)
    - window: Window function to apply when designing the filter (default 'hann')
    
    Returns:
    - data_filt: The band-pass filtered data (same shape as input)
    """

    # Create FIR filter coefficients
    fir_coeff = firwin(numtaps=numtaps, cutoff=[cutoff_low, cutoff_high], window=window, pass_zero=False, fs=fs)
    
    # Apply zero-phase forward-backward channel-wise filtering to avoid phase distortion
    data_filt = filtfilt(fir_coeff, 1.0, data, axis=1)

    # Compute the frequency response of the filter
    freq, h = freqz(fir_coeff, worN=8000, fs=fs)

    return data_filt, freq, h