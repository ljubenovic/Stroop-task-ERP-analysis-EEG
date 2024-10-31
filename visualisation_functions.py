import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utility_functions import *
from erp_functions import epoch_averaging
import hyperparameters as hp

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

############################################################################################################
def plot_multichannel(data, fs, chanlocs, event_type, event_latency, S_id, filename=None, time_window=None):

    '''Prikazuje vremeske oblike višekanalnog signala (različite kanale na različitim subplot-ovima)'''

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
        plt.plot(t,data[i,:],'k', linewidth=0.6)
        if np.any(event_latency):
            for j in range(len(event_latency)):
                if event_type[j] == 1:
                    plt.axvline(x=t[event_latency[j]], color='g', alpha=0.3)
                else:
                    plt.axvline(x=t[event_latency[j]], color='r', alpha=0.3)
        plt.xlabel('t [s]')
        plt.ylabel('amplituda [a.u.]')
        plt.title('ch: {}'.format(chanlocs[i]))
        plt.grid(True, which='both', alpha=0.5)
    plt.tight_layout()

    if filename:
        path = os.path.join("data","Graphs", "S{}".format(S_id), filename + ".png")
        plt.savefig(path)

    plt.close()

    # Zoomed plot
    if n_chan <= 3:
        plt.figure(figsize=(10,8))
    else:
        plt.figure(figsize=(10,50))
    for i in range(n_chan):
        plt.subplot(n_chan,1,i+1)
        plt.plot(t,data[i,:],'k', linewidth=0.6)
        if np.any(event_latency):
            for j in range(len(event_latency)):
                if event_type[j] == 1:
                    plt.axvline(x=t[event_latency[j]], color='g', alpha=0.3)
                else:
                    plt.axvline(x=t[event_latency[j]], color='r', alpha=0.3)
        plt.xlabel('t [s]')
        plt.ylabel('amplituda [a.u.]')
        plt.title('ch: {}'.format(chanlocs[i]))
        plt.xlim(0,30)
        plt.grid(True, which='both', alpha=0.5)
    plt.tight_layout()

    if filename:
        path = os.path.join("data","Graphs", "S{}".format(S_id), filename + "_zoomed.png")
        plt.savefig(path)

    plt.close()

############################################################################################################
def plot_multichannel_with_peaks(data, peaks, fs, chanlocs, S_id, filename=None):

    '''Prikazuje vremeske oblike višekanalnog signala zajedno sa prosleđenim pikovima (različite kanale na različitim subplot-ovima)'''

    (n_chan, n_samp) = data.shape    # (34, 243080)
    t = np.arange(0, n_samp/fs, 1/fs)
    if np.squeeze(t.shape) > data.shape[1]:
        t = t[:data.shape[1]]

    if n_chan <= 3:
        plt.figure(figsize=(12,6))
    else:
        plt.figure(figsize=(10,50))
    for i in range(n_chan):
        plt.subplot(n_chan,1,i+1)
        plt.plot(t,data[i,:],'k', linewidth=0.6, alpha=0.8)
        plt.scatter(t[peaks[i]],data[i,peaks[i]], c='r', edgecolors='r', s=6)
        plt.xlabel('t [s]')
        plt.ylabel('amplituda [a.u.]')
        plt.title('{}'.format(chanlocs[i]))
        plt.grid(True, which='both', alpha=0.5)
    plt.tight_layout()

    if filename:
        path = os.path.join("data","Graphs", "S{}".format(S_id), filename + ".png")
        plt.savefig(path)

    plt.close()

    return

############################################################################################################
def plot_multichannel_fft(data, fs, chanlocs, S_id, filename=None):

    '''Prikazuje brze Furijeove transformacije (FFT) višekanalnog signala (različite kanale na različitim subplot-ovima)'''

    (n_chan, n_samp) = data.shape    # (34, 243080)
    n_samp = np.max([n_samp,100000])
    freqs = np.fft.fftfreq(n_samp, 1/fs)[:n_samp // 2]

    if n_chan <= 3:
        plt.figure(figsize=(10,8))
    else:
        plt.figure(figsize=(10,50))
    for i in range(n_chan):
        signal = data[i,:]
        fft_signal = np.fft.fft(signal, n_samp)
        fft_signal = np.abs(fft_signal[:n_samp // 2])

        plt.subplot(n_chan,1,i+1)
        plt.plot(freqs, fft_signal,'k', linewidth=0.6)
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

############################################################################################################
def plot_ampl_response(h, freq, filt_type = ""):

    '''Prikazuje amplitudski odziv filtra'''

    plt.figure(figsize=(12,4))

    plt.plot(freq, np.abs(h),'k',linewidth=0.8,alpha=0.8)
    plt.title('Amplitudska frekvencijska karakteristika {} filtra'.format(filt_type))
    plt.xlabel('f [Hz]')
    plt.ylabel('magnituda [a.u.]')
    plt.grid(which='both')
    plt.xlim(0,100)

    plt.tight_layout()
    plt.savefig(os.path.join('data','Results','frekv_karakteristika_{}.png'.format(filt_type)))
    plt.close()

def eeg_processing_visualisation(eeg_raw, eeg_cz, eeg, eeg_filt_1, eeg_filt, fs, chanlocs, channel, S_id):

    '''Prikazuje međukorake u predobradi EEG signala'''

    chanlocs = [x for x in chanlocs]

    x0 = eeg_raw[chanlocs.index(channel),:]
    x1 = eeg_cz[chanlocs.index(channel),:]
    t0 = np.arange(0, len(x0)/fs, 1/fs)

    x2 = eeg[chanlocs.index(channel),:]
    x3 = eeg_filt_1[chanlocs.index(channel),:]
    x4 = eeg_filt[chanlocs.index(channel),:]
    t = np.arange(0, len(x2)/fs, 1/fs)
    
    plt.figure(figsize=(12,12))

    plt.subplot(5,1,1)
    plt.plot(t0,x0,'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('Sirovi EEG signal')
    plt.grid(which='both')
    
    plt.subplot(5,1,2)
    plt.plot(t0,x1,'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('EEG signal rereferenciran u odnosu na Cz')
    plt.grid(which='both')

    plt.subplot(5,1,3)
    plt.plot(t,x2,'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('EEG signal nakon segmentacije')
    plt.grid(which='both')

    plt.subplot(5,1,4)
    plt.plot(t,x3,'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('EEG signal nakon visokopropusnog filtriranja')
    plt.grid(which='both')

    plt.subplot(5,1,5)
    plt.plot(t,x4,'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('EEG signal nakon niskopropusnog filtriranja')
    plt.grid(which='both')

    plt.tight_layout()
    plt.savefig(os.path.join('data','Graphs','S{}'.format(S_id),'predobrada_0.png'))
    plt.close()

    return

############################################################################################################
def eog_processing_visualisation(eog_raw, eog, eog_filt, fs, S_id):
    
    '''Prikazuje međukorake u obradi EOG signala'''

    t0 = np.arange(0, eog_raw.shape[1]/fs, 1/fs)
    t = np.arange(0, eog.shape[1]/fs, 1/fs)
    
    plt.figure(figsize=(12,7.2))

    plt.subplot(3,1,1)
    plt.plot(t0,eog_raw[1,:],'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('Sirovi E0G signal')
    plt.grid(which='both')

    plt.subplot(3,1,2)
    plt.plot(t,eog[1,:],'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('EOG signal nakon segmentacije')
    plt.grid(which='both')

    plt.subplot(3,1,3)
    plt.plot(t,eog_filt[1,:],'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('E0G signal nakon filtriranja propusnikom opsega učestanosti')
    plt.grid(which='both')

    plt.tight_layout()
    plt.savefig(os.path.join('data','Graphs','S{}'.format(S_id),'predobrada_1.png'))
    plt.close()

    return

############################################################################################################
def eeg_denoising_visualisation(eeg, eeg_denoised, fs, chanlocs, channel, S_id):
    
    '''Prikazuje izgled EEG signala pre i nakon uklanjanja artefakata analizom nezavisnih komponenti'''

    chanlocs = [x for x in chanlocs]

    x0 = eeg[chanlocs.index(channel),:]
    x1 = eeg_denoised[chanlocs.index(channel),:]
    t0 = np.arange(0, len(x0)/fs, 1/fs)
    
    plt.figure(figsize=(12,6))

    plt.subplot(2,1,1)
    plt.plot(t0,x0,'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('EEG signal pre odšumljivanja')
    plt.grid(which='both')
    
    plt.subplot(2,1,2)
    plt.plot(t0,x1,'k',linewidth=0.4,alpha=0.8)
    plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
    plt.title('EEG signal nakon odšumljivanja')
    plt.grid(which='both')

    plt.tight_layout()
    plt.savefig(os.path.join('data','Graphs','S{}'.format(S_id),'predobrada_2.png'))
    plt.close()

    return

############################################################################################################
def ica_components_visualisation(components, fs, S_id):
    
    '''Prikazuje vremenske oblike komponenti dobijenih analizom nezavisnih komponenti na različitim subplot-ovima'''

    n_components, n_samp = components.shape
    t = np.arange(0, n_samp/fs, 1/fs)

    plt.figure(figsize=(14,20))
    for i in range(n_components):
        ic = components[i,:]

        plt.subplot(n_components,1,i+1)
        plt.plot(t,ic,'k',linewidth=0.6,alpha=0.8)
        #plt.ylabel('amplituda [a.u.]')
        plt.ylabel('{}'.format(i+1))
        plt.yticks([])
        if i == 0:
            plt.title('Vremenski oblici izdvojenih komponenti')
        if i == 28:
            plt.xlabel('t [s]')
        else:
            plt.xticks([])
        plt.grid(True, which='both', alpha=0.5)
        plt.xlim((0,30))

    plt.subplots_adjust(wspace=0, hspace=0)

    path = os.path.join('data','Graphs','S{}'.format(S_id),'ICA_components_zoomed.png')
    plt.savefig(path)
    plt.close()

    plt.figure(figsize=(14,20))
    for i in range(n_components):
        ic = components[i,:]

        plt.subplot(n_components,1,i+1)
        plt.plot(t,ic,'k',linewidth=0.6,alpha=0.8)
        #plt.ylabel('amplituda [a.u.]')
        plt.ylabel('{}'.format(i+1))
        plt.yticks([])
        if i == 0:
            plt.title('Vremenski oblici izdvojenih komponenti')
        if i == 28:
            plt.xlabel('t [s]')
        else:
            plt.xticks([])
        plt.grid(True, which='both', alpha=0.5)
    plt.subplots_adjust(wspace=0, hspace=0)

    path = os.path.join('data','Graphs','S{}'.format(S_id),'ICA_components.png')
    plt.savefig(path)
    plt.close()

    return

############################################################################################################
def plot_block_epochs_with_averages(epochs_1, epochs_2, fs, chanlocs, channel, S_id):
    
    '''Prikazuje originalne epohe EEG signala zajedno sa njihovom srednjom vrednošću za sve blokove pojedinačno'''

    ch_idx = chanlocs.index(channel)
    epochs_ch_1 = epochs_1[ch_idx,:,:]
    epochs_ch_2 = epochs_2[ch_idx,:,:]

    n_samp = epochs_ch_1.shape[0]
    n_epochs = 16

    t = np.arange(0, n_samp/fs, 1/fs)*1000

    plt.figure(figsize=(12,8))

    for idx in range(4):
        if idx == 0:
            block = epochs_ch_1[:,:16]
        elif idx == 1:
            block = epochs_ch_1[:,16:]
        elif idx == 2:
            block = epochs_ch_2[:,:16]
        else:
            block = epochs_ch_2[:,16:]

        block_avg = epoch_averaging(block)

        plt.subplot(2,2,idx+1)
        for i in range(n_epochs):
            plt.plot(t,block[:,i],'--k',linewidth=0.8,alpha=0.4)
        plt.plot(t,block_avg,'k',linewidth=1.6,alpha=0.8)
        plt.grid(which='both',alpha=0.5)
        plt.xlabel('t [ms]'); plt.ylabel('amplituda [a.u.]')
        if idx < 2:
            plt.title('Blok {} (neutralni stimulusi)'.format(idx+1))
        else:
            plt.title('Blok {} (nekongruentni stimulusi)'.format(idx+1))
    plt.tight_layout()
    if S_id != 0:
        path = os.path.join('data','Graphs','S{}'.format(S_id), 'block_avgs_{}.png'.format(channel))
    else:
        path = os.path.join('data','Results', 'block_avgs_{}.png'.format(channel))
    plt.savefig(path)
    plt.close()

    # Ponovljeni postupak za parove blokova

    n_epochs = 32

    plot_epochs_with_average(epochs_1, epochs_2, fs, chanlocs, channel, S_id)

    return

############################################################################################################
def plot_epochs_with_average(epochs_1, epochs_2, fs, chanlocs, channel, S_id):
    
    '''Prikazuje epohe EEG signala sa zadatog kanala zajedno sa njihovom srednjom vrednošću za obe vrste stimulusa'''
    
    (_, n_samp, n_epochs) = epochs_1.shape
    ch_idx = chanlocs.index(channel)
   
    t = np.arange(0, n_samp/fs, 1/fs)*1000

    plt.figure(figsize=(12,4))
    for idx in range(2):
        plt.subplot(1,2,idx+1)

        if idx == 0:
            epochs = epochs_1
        else:
            epochs = epochs_2

        epochs_ch = epochs[ch_idx,:,:]
        epoch_avg = epoch_averaging(epochs_ch)        

        for i in range(n_epochs):
            plt.plot(t,epochs_ch[:,i],'--k',linewidth=0.6,alpha=0.4)

        plt.plot(t,epoch_avg,'k',linewidth=1.6,alpha=0.8)
        plt.grid(which='both',alpha=0.5)
        plt.xlabel('t [ms]'); plt.ylabel('amplituda [a.u.]')
        if idx == 0:
            plt.title('Neutralni stimulusi'.format(idx+1))
        else:
            plt.title('Nekongruentni stimulusi'.format(idx+1))
    plt.tight_layout()
    if S_id != 0:
        path = os.path.join('data','Graphs','S{}'.format(S_id), 'epochs_avgs_{}.png'.format(channel))
    else:
        path = os.path.join('data','Results','epochs_avgs_{}.png'.format(channel))
    plt.savefig(path)
    plt.close()

############################################################################################################
def plot_boxplots(data, tick_labels, ylabels, titles, filename):

    '''Prikazuje boxplot prosleđenih podataka'''

    n_subplots = len(data)
    
    if n_subplots != 1:
        fig, axs = plt.subplots(1, n_subplots, figsize=(14, 8))
    else:
        fig, axs = plt.subplots(1, n_subplots, figsize=(8, 4))

    for i in range(n_subplots):

        data_i_1 = [x for x in data[i][0] if np.isfinite(x)]
        data_i_2 = [x for x in data[i][1] if np.isfinite(x)]
        data_i = [data_i_1, data_i_2]
        
        if isinstance(axs, np.ndarray):
            ax = axs[i]
        else:
            ax = axs
        ax.boxplot(data_i, tick_labels=tick_labels,
                notch=True, patch_artist=False, showmeans=True,
                medianprops=dict(color='red', linewidth=3, linestyle='-'),
                meanprops=dict(color='green',marker='v'),
                boxprops = dict(linestyle='-', linewidth=2, color='black'),
                flierprops = dict(marker='o', markerfacecolor='black', markersize=8, linestyle='none', markeredgecolor='black'))
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.grid(True)

    legend_labels = ['median', 'mean', 'outlier', 'box']
    legend_handles = [
        plt.Line2D([0], [0], color='red', linewidth=3),   # median
        plt.Line2D([0], [0], color='green', marker='v', linestyle='none'),  # mean
        plt.Line2D([0], [0], color='black', marker='o', markersize=8, linestyle='none', markeredgecolor='black'),     # Outliers
        plt.Line2D([0], [0], color='black', linewidth=2)   # box edges
    ]
    fig.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(0.92, 0.92), borderaxespad=0.)

    plt_path = os.path.join('data','Results', filename)
    plt.savefig(plt_path, bbox_inches='tight')

    plt.close()

############################################################################################################
def plot_avg_epochs_on_subplots(channel):
    
    '''Prikazuje usrednjene epohe za pojedinačne ispitanike (za zadati kanal)'''

    # 5x4 subplot za prikaz usrednjenih epoha za sve ispitanike
    
    fs = 1000
    n_samp = int((hp.epoch_duration - hp.pre_stim_duration)/1000*fs)
    t = np.arange(0, n_samp/fs, 1/fs)*1000

    chanlocs = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ',
                 'FC4', 'FT8', 'T7', 'C3', 'C4', 'T8', 'TP7', 'CP3', 'CPZ', 'CP4',
                   'TP8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']
    ch_idx = chanlocs.index(channel)

    j = 0
    plt.figure(figsize=(14,20))
    for i in range(0,21):

        S_id = i+1

        if S_id == 5:
            continue

        epoch_avg_1 = load_csv(S_id, 'epoch_avg_1', os.path.join('data','N200'))
        epoch_avg_2 = load_csv(S_id, 'epoch_avg_2', os.path.join('data','N200'))

        plt.subplot(10,2,j+1)
        plt.plot(t, epoch_avg_1[ch_idx,:], 'k', linewidth=1.2, alpha=0.8, label='neutralni')
        plt.plot(t, epoch_avg_2[ch_idx,:], '--k', linewidth=0.8, alpha=0.8, label='nekongruentni')
        plt.xlabel('t [ms]'); plt.ylabel('amplituda [a.u.]')
        plt.title('S{}'.format(S_id))
        plt.grid(which='both')
        plt.legend(loc='upper right')
        plt.grid(which='both',alpha=0.5)

        j+=1

    plt.tight_layout()
    path = os.path.join('data','Results','all_subjects_averages_{}.png'.format(channel))
    plt.savefig(path)
    plt.close()

############################################################################################################
def plot_block_avgs_on_subplots(type, channel):
    
    '''Prikazuje usrednjene epohe za svakog ispitanika pojedinačno i izabrani kanal (za neutralne i nekongruentne stimuluse na istom subplot-u)'''

    chanlocs = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ',
                'FC4', 'FT8', 'T7', 'C3', 'C4', 'T8', 'TP7', 'CP3', 'CPZ', 'CP4',
                'TP8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']
    
    fs = 1000
    n_samp = int((hp.epoch_duration - hp.pre_stim_duration)/1000*fs)
    t = np.arange(0, n_samp/fs, 1/fs)*1000
    
    j = 0
    plt.figure(figsize=(14,8))
    for i in range(0,21):

        S_id = i+1
        if S_id == 5:
            continue

        epochs_st = load_csv(S_id, 'epochs_'+type, os.path.join('data','N200'))
        epochs_st = epochs_st.reshape(29, 900, 32)

        block_avg_1 = epoch_averaging(epochs_st[chanlocs.index(channel),:,:16])
        block_avg_2 = epoch_averaging(epochs_st[chanlocs.index(channel),:,16:])

        plt.subplot(5,4,j+1)
        plt.plot(t,block_avg_1,'k',linewidth=1.6, label='blok 1')
        plt.plot(t,block_avg_2,'--k',linewidth=1, label='blok 2')
        plt.title('S{}'.format(S_id, channel))
        plt.xlabel('t [ms]'); plt.ylabel('amplituda [a.u.]')
        plt.legend(loc='upper right')
        plt.grid(which='both',alpha=0.5)

        j+=1

    plt.tight_layout()
    path = os.path.join('data','Results','all_subjects_block_avgs_{}_{}.png'.format(type,channel))
    plt.savefig(path)
    plt.close()

############################################################################################################
def plot_avg_epochs_all_channels(epochs_avg_1, epochs_avg_2, fs, chanlocs, S_id=0, amps = None, lats=None, filename='global_averages'):
    
    '''Prikazuje globalno usrednjene epohe za svaki od kanala'''

    # Prikaz globalno usrednjenih potencijala (za sve ispitanike), na 10x3 subplotu

    (n_ch, n_samp) = epochs_avg_1.shape
    t = np.arange(0, n_samp/fs, 1/fs)

    plt.figure(figsize=(14,20))
    for i in range(n_ch):
        channel = chanlocs[i]

        plt.subplot(10,3,i+1)
        plt.plot(t, epochs_avg_1[i,:], 'k', linewidth=1.2, alpha=0.8, label='neutralni')
        plt.plot(t, epochs_avg_2[i,:], '--k', linewidth=0.8, alpha=0.8, label='nekongruentni')
        if np.any(amps) and np.any(lats):
            plt.scatter(lats[i,0]/1000,-1*amps[i,0],c='r',s=10)
            plt.scatter(lats[i,1]/1000,-1*amps[i,1],c='r',s=10)
        plt.xlabel('t [s]'); plt.ylabel('amplituda [a.u.]')
        plt.title('{}'.format(channel))
        plt.grid(which='both')
        plt.legend(loc='upper right')
    plt.tight_layout()
    if S_id:
        path = os.path.join('data','Graphs','S{}'.format(S_id),filename+'.png')
    else:
        path = os.path.join('data','Results',filename+'.png')
    plt.savefig(path)
    #plt.show()
    plt.close()

    return

############################################################################################################
def erp_bar_plots(amp_arr, lat_arr, channel):
    
    '''Prikazuje razlike između amplituda i latenci između neutralnih i nekongruentnih stimulusa'''

    x_ticks = [x+1 if x>=5 else x for x in np.arange(1,21)]
    x_ticks_1 = [x-0.15 for x in x_ticks]
    x_ticks_2 = [x+0.15 for x in x_ticks]

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.bar(x_ticks_1, amp_arr[0], width=0.3, label='neutralni', color='k', alpha=1, zorder=3)
    plt.bar(x_ticks_2, amp_arr[1], width=0.3, label='nekongruentni', color='k', alpha=0.5, hatch='//', zorder=2)
    plt.xlabel('ID ispitanika'); plt.xticks(x_ticks)
    plt.ylabel('amplituda [a.u.]')
    plt.title('Amplituda N200')
    plt.grid(which='both',zorder=0,alpha=0.5)
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.bar(x_ticks_1, lat_arr[0], width=0.3, label='neutralni', color='k', alpha=1, zorder=3)
    plt.bar(x_ticks_2, lat_arr[1], width=0.3, label='nekongruentni', color='k', alpha=0.5, hatch='//', zorder=2)
    plt.xlabel('ID ispitanika'); plt.xticks(x_ticks)
    plt.ylabel('latenca [ms]')
    plt.title('Latenca N200')
    plt.grid(which='both',zorder=0,alpha=0.5)
    plt.legend(loc='upper right')

    plt.tight_layout()
    path = os.path.join('data','Results','n200_bars_{}.png'.format(channel))    
    plt.savefig(path)
    plt.close()

    return

############################################################################################################
def plot_samples(amp_arr, lat_arr, channel, type):
    
    '''Prikazuje izdvojene komponente ERP-a u prostoru obeležja (amplituda i latenci)'''

    plt.figure(figsize=(10,6))
    plt.scatter(amp_arr[0],lat_arr[0], c='k',label='nautralni',s=30)
    plt.scatter(amp_arr[1],lat_arr[1], c='r',label='nekongruentni',s=30)
    plt.xlabel('amplituda [a.u.]'); plt.ylabel('latenca [ms]')
    plt.title('Prikaz odbiraka detektovanih N200 komponenti')
    plt.grid(which='both',alpha=0.5)
    plt.legend(loc='upper right')
    path = os.path.join('data','Results','n{}_samples_{}.png'.format(type,channel))
    plt.savefig(path)
    plt.close()

############################################################################################################
def plot_table(df, filename, flag=None):

    '''Prikazuje tabelu koja odgovara prosleđenom dataframe-u'''

    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns.to_list(), cellLoc = 'center', loc='center', rowLabels=df.index.to_list())
    table.auto_set_column_width([0, 1])
    table.scale(1, 2)
    if flag == 1:
        # u slucaju da se crta tabela za p-vrednsti, boji u svetlo sivo sva polja manja od 0.05
        for i in range(len(df)):
            if df.values[i] < 0.05:
                table[(i+1),0].set_facecolor('grey')
    if flag == 2:
        # u slucaju da se crta tabela za amplitude, boji u svetlo sivo sva polja gde je amplituda nekongruentnog veca
        for i in range(len(df)):
                if df.values[i, 1] > df.values[i,0]:
                    table[(i+1),0].set_facecolor('grey')
                    table[(i+1),1].set_facecolor('grey')
    
    plt.tight_layout()
    plt.savefig(os.path.join('data','Results',filename+'.png'))
    plt.close()

    return

############################################################################################################
def plot_all_channels(eeg, fs, S_id, chanlocs):

    n_components, n_samp = eeg.shape
    t = np.arange(0, n_samp/fs, 1/fs)

    path = os.path.join('data','Graphs','S{}'.format(S_id),'eeg_raw_0.png')

    plt.figure(figsize=(14,20))
    for i in range(n_components):
        ic = eeg[i,:]

        plt.subplot(n_components,1,i+1)
        plt.plot(t,ic,'k',linewidth=0.6,alpha=0.8)
        #plt.ylabel('amplituda [a.u.]')
        plt.ylabel('{}'.format(chanlocs[i]))
        plt.yticks([])
        if i == 0:
            if os.path.exists(path):
                plt.title('Vremenski oblici EEG signala nakon predobrade')
            else:
                plt.title('Vremenski oblici sirovih EEG signala')
        if os.path.exists(path) and i == 28:
            plt.xlabel('t [s]')
        elif i == 33:
            plt.xlabel('t [s]')
        else:
            plt.xticks([])
        plt.grid(True, which='both', alpha=0.5)
        #plt.xlim((0,30))

    plt.subplots_adjust(wspace=0, hspace=0)

    if os.path.exists(path):
        path = os.path.join('data','Graphs','S{}'.format(S_id),'eeg_denoised_0.png')
    plt.savefig(path)
    plt.close()

    """plt.figure(figsize=(14,20))
    for i in range(n_components):
        ic = eeg[i,:]

        plt.subplot(n_components,1,i+1)
        plt.plot(t,ic,'k',linewidth=0.6,alpha=0.8)
        #plt.ylabel('amplituda [a.u.]')
        plt.ylabel('{}'.format(i+1))
        plt.yticks([])
        if i == 0:
            plt.title('Vremenski oblici izdvojenih komponenti')
        if i == 28:
            plt.xlabel('t [s]')
        else:
            plt.xticks([])
        plt.grid(True, which='both', alpha=0.5)
    plt.subplots_adjust(wspace=0, hspace=0)

    path = os.path.join('data','Graphs','S{}'.format(S_id),'ICA_components.png')
    plt.savefig(path)
    plt.close()"""

    return

