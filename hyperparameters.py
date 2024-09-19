"""
hyperparameters.py

This script contains all the key hyperparameters used throughout the project.

Variables:
-----------

1) Hyperparameters used for the preprocessing of raw EEG data:

    - reference_channel: str
    - cutoff_low: float (in Hz)
    - cutoff_high: float (in Hz)
    - filt_order: int (in Hz)

2) Hyperparameters used for the preprocessing of raw EOG data (from HEO and VEO channels):

    - cutoff_low_eog: float (in Hz)
    - cutoff_high_eog: float (in Hz)
    - numtaps: int
    - window: str
    - blink_epoch_duration: float (in miliseconds)

3) Hyperparameters used for removing the blink-related artifacts:

    - ortho: bool
    - extended: bool
    - threshold_z: float

    Notes:
    ------

    The parameter ortho choses whether to work under orthogonal constraint (i.e. enforce the decorrelation of the output) or not.
    It also comes with an extended version just like extended-infomax, which makes separation of both sub and super-Gaussian signals possible.
    It is chosen with the parameter extended.

        ortho=False, extended=False: same solution as Infomax
        ortho=False, extended=True: same solution as extended-Infomax
        ortho=True, extended=True: same solution as FastICA
        ortho=True, extended=False: finds the same solutions as Infomax under orthogonal constraint.

4) Hyperparameters used for the extraction of the N200 event-related potential:
        
    - epoch_duration: float (in miliseconds)
    - pre_stim_duration: float (in miliseconds)
    - time_window: tuple (in miliseconds)
    - focused_channel: str
    - relevant_channels: list of str

"""

reference_channel = 'CZ'
cutoff_low_eeg = 2
cutoff_high_eeg = 40
filt_order = 3

cutoff_low_eog = 1
cutoff_high_eog = 10
numtaps = 101
window = 'hann'
blink_epoch_duration = 1000

ortho = False
extended = False
threshold_z = 1

epoch_duration = 1000
pre_stim_duration = 100
time_window = (250, 320)    # expected averaged latency ~ 281 ms
focused_channel = 'FCZ' # expected maximal amplitude at FCz
#relevant_channels = ['FC3','FC1','FCZ','FC2','FC4','C3','C1','CZ','C2','C4']    # Electrodes of the fronto-central cortex
relevant_channels = ['FZ','FCZ']