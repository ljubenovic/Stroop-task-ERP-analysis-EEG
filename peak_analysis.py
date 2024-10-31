import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from utility_functions import *
from erp_functions import *
from visualisation_functions import *
from scipy.stats import ttest_rel

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

############################################################################################################
def paired_t_test(samples_1, samples_2, channel):

    t_stat, p_value = ttest_rel(samples_1, samples_2)

    #print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    alpha = 0.05  # nivo značajnosti
    
    if p_value < alpha:
        print("Postoji statistički značajna razlika za kanal {} (p = {})".format(channel, p_value))

    return np.round(p_value,3)

############################################################################################################
def hist_most_important_peaks(chanlocs, channel, percentage = 5):

    peaks_arr_1 = []
    peaks_arr_2 = []
    plt.figure(figsize=(10,4))
    for idx in range(2):
        plt.subplot(1,2,idx+1)

        for i in range(0,21):

            S_id = i+1

            if S_id == 5:
                continue

            type = str(idx+1)
            epoch_avg = load_csv(S_id, 'epoch_avg_' + type, os.path.join('data','N200'))
            epoch_avg_ch = epoch_avg[chanlocs.index(channel),:]
            signal = -epoch_avg_ch

            threshold = np.percentile(signal, 100-percentage)
            peaks, _ = find_peaks(signal, height=threshold)
            # Na ovaj nacin su odredjeni najznacajniji pikovi u signalu, osnosno oni pikovi koji upadaju u 5 % najmanjih odbiraka po amplitudi
            
            if idx == 0:
                peaks_arr_1.extend(peaks)
            else:
                peaks_arr_2.extend(peaks)

        if idx == 0:    
            plt.hist(peaks_arr_1, density=True, bins=np.arange(100,900,20), color='k', zorder=3, alpha=0.9)
            plt.title('Histogram za neutralne stimuluse')
        else:
            plt.hist(peaks_arr_2, density=True, bins=np.arange(100,900,20), color='k', zorder=3, alpha=0.9)
            plt.title('Histogram za nekongruentne stimuluse')
        plt.xlabel('latenca pika [ms]'); plt.ylabel('gustina verovatnoće')
        plt.grid(which='both',zorder=0, alpha=0.5)
    plt.tight_layout()
    path = os.path.join('data','Results','hist_peaks_{}.png'.format(channel))
    plt.savefig(path)
    plt.close()

    return

############################################################################################################
def erp_analysis_avg_epochs(chanlocs, channel, fs, time_window):

    ch_idx = chanlocs.index(channel)

    amp_arr = [[],[]]
    lat_arr = [[],[]]
    for idx in range(2):
        for i in range(0,21):

                S_id = i+1

                if S_id == 5:
                    continue

                type = str(idx+1)
                epoch_avg = load_csv(S_id, 'epoch_avg_' + type, os.path.join('data','N200'))
                epoch_avg_ch = epoch_avg[ch_idx,:]

                (erp_amp, erp_latency) = erp_extraction(epoch_avg_ch, fs, time_window)
                amp_arr[idx].append(erp_amp)
                lat_arr[idx].append(erp_latency)

    ### Vizuelizacija
    erp_bar_plots(amp_arr, lat_arr, channel)
    plot_samples(amp_arr, lat_arr, channel)

    ### Ispisivanje srednjih vrednosti i standardnih devijacija na stdout

    print('\n')
    print('---neutral---')
    print('amplitude:')
    print('mean: ', np.mean(amp_arr[0]))
    print('std: ', np.std(amp_arr[0]))
    print('latency:')
    print('mean: ', np.mean(lat_arr[0]))
    print('std: ', np.std(lat_arr[0]))
    print('\n')
    print('---incongruent---')
    print('amplitude:')
    print('mean: ', np.mean(amp_arr[1]))
    print('std: ', np.std(amp_arr[1]))
    print('latency:')
    print('mean: ', np.mean(lat_arr[1]))
    print('std: ', np.std(lat_arr[1]))

    ### Paired t-test

    p_vals_amps = paired_t_test(amp_arr[0], amp_arr[1], channel)
    p_vals_lats = paired_t_test(lat_arr[0], lat_arr[1], channel)

    return (p_vals_amps, p_vals_lats)



if __name__ == '__main__':    

    _, fs, chanlocs, _, event_latency, _ = load_mat(1, 'EEG_denoised', os.path.join('data','Processed data', 'Processed EEG data'))

    hist_most_important_peaks(chanlocs, 'FZ', 5)
    hist_most_important_peaks(chanlocs, 'FCZ', 5)
    hist_most_important_peaks(chanlocs, 'C4', 5)

    channel = 'FZ'
    time_window_200 = (120, 260)
    time_window_400 = (320, 480)

    erp_analysis_avg_epochs(chanlocs, channel, fs, time_window_200)   
    erp_analysis_avg_epochs(chanlocs, channel, fs, time_window_400)   

    channel = 'FCZ'
    time_window_200 = (100, 200)
    time_window_400 = (280, 420)

    erp_analysis_avg_epochs(chanlocs, channel, fs, time_window_200)   
    erp_analysis_avg_epochs(chanlocs, channel, fs, time_window_400)   

    channel = 'C4'
    time_window_200 = (180, 300)
    time_window_400 = (420, 600)

    erp_analysis_avg_epochs(chanlocs, channel, fs, time_window_200)   
    erp_analysis_avg_epochs(chanlocs, channel, fs, time_window_400)    

    """
    time_window_200 = (100,200)
    p_amp_arr = []
    p_lat_arr = []
    for channel in chanlocs:
        (p_amp, p_lat) = erp_analysis_avg_epochs(chanlocs, channel, fs, time_window_200)
        p_amp_arr.append(p_amp)
        p_lat_arr.append(p_lat)

    idx_amp = np.nanargmin(p_amp_arr)
    print('Statisticki najznacajnija razlika u apmplitudama postoji za {} (p = {})'.format(chanlocs[idx_amp], p_amp_arr[idx_amp]))
    idx_lat = np.nanargmin(p_lat_arr)
    print('Statisticki najznacajnija razlika u latencama postoji za {} (p = {})'.format(chanlocs[idx_lat], p_lat_arr[idx_lat]))

    # Postoji statistiscki znacajna razlika u amplitudama za:
    # (100,300): FCZ, F8, FT8 
    # (150,300): FCZ, FT8
    # (100,250): FCZ, F3, FC3
    # Postoji statistiscki znacajna razlika u latencama za:
    # (100,300): FT8, CPZ
    # (150,300): OZ, T7

    # (120,350)
    # Statisticki najznacajnija razlika u apmplitudama postoji za FCZ (p = 0.020676663912460976)
    # Statisticki najznacajnija razlika u latencama postoji za FZ (p = 0.011069972922074789)
    # (200,300)
    # Statisticki najznacajnija razlika u apmplitudama postoji za FT8 (p = 0.009683662914345243)
    # Statisticki najznacajnija razlika u latencama postoji za FP2 (p = 0.0562390487374549)
    """
    