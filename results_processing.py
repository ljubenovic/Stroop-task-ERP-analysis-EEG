import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def read_erp_data(S_id, type, channel = "FZ"):

    path = os.path.join('data','N200','S{}_'.format(S_id)+type+'.csv')

    df = pd.read_csv(path, header=None, skiprows=1)
    df.columns = ['channel', 'amplitude', 'latency']
    df['channel'] = [ch[:-1] for ch in df['channel'].astype(str)]

    amp = df.loc[df['channel'] == channel, 'amplitude'].values
    latency = df.loc[df['channel'] == channel, 'latency'].values

    return (amp, latency)


if __name__ == '__main__':

    channel = "FZ"

    amp_neutral = []
    amp_incongruent = []
    latency_neutral = []
    latency_incongruent = []

    id_start = 0
    id_end = 14

    S_ids_neutral = []
    S_ids_incongruent = []

    for i in range(id_start, id_end):

        if i==4:
            continue

        S_id = i+1

        type = 'neutral'
        (amp, latency) = read_erp_data(S_id, type, channel)
        if amp and latency:
            amp_neutral.append(amp)
            latency_neutral.append(latency)
            S_ids_neutral.append(S_id)
            
        type = 'incongruent'
        (amp, latency) = read_erp_data(S_id, type, channel)
        if amp and latency:
            amp_incongruent.append(amp)
            latency_incongruent.append(latency)
            S_ids_incongruent.append(S_id)


    amp_neutral = np.array(amp_neutral)
    amp_incongruent = np.array(amp_incongruent)
    latency_neutral = np.array(latency_neutral)
    latency_incongruent = np.array(latency_incongruent)
    S_ids_neutral = np.array(S_ids_neutral)
    S_ids_incongruent = np.array(S_ids_incongruent)

    # Plots for neutral stimulus
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    ax = axs[0]
    ax.scatter(S_ids_neutral, amp_neutral)
    ax.set_xlabel('S_id'); ax.set_ylabel('N200 amplitude [a.u.]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('N200 amplitude (neutral stimulus)')

    ax = axs[1]
    ax.scatter(S_ids_neutral, latency_neutral)
    ax.set_xlabel('S_id'); ax.set_ylabel('N200 latency [ms]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('N200 latency (neutral stimulus)')

    for ax in axs.flat:
        ax.grid(True)

    plt.tight_layout()

    plt_path = os.path.join('data','Results','N200_scatter_neutral.png')
    plt.savefig(plt_path)

    plt.close()

    # Plots for incongruent stimulus
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    ax = axs[0]
    ax.scatter(S_ids_incongruent, amp_incongruent)
    ax.set_xlabel('S_id'); ax.set_ylabel('N200 amplitude [a.u.]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('N200 amplitude (incongruent stimulus)')

    ax = axs[1]
    ax.scatter(S_ids_incongruent, latency_incongruent)
    ax.set_xlabel('S_id'); ax.set_ylabel('N200 latency [ms]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('N200 latency (incongruent stimulus)')

    for ax in axs.flat:
        ax.grid(True)

    plt.tight_layout()

    plt_path = os.path.join('data','Results','N200_scatter_incongruent.png')
    plt.savefig(plt_path)

    plt.close()

    # Neutral and incongruent stimuluses on the same plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    ax = axs[0,0]
    ax.scatter(S_ids_neutral, amp_neutral)
    ax.set_xlabel('S_id'); ax.set_ylabel('N200 amplitude [a.u.]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('N200 amplitude (neutral stimulus)')

    ax = axs[1,0]
    ax.scatter(S_ids_neutral, latency_neutral)
    ax.set_xlabel('S_id'); ax.set_ylabel('N200 latency [ms]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('N200 latency (neutral stimulus)')

    ax = axs[0,1]
    ax.scatter(S_ids_incongruent, amp_incongruent)
    ax.set_xlabel('S_id'); ax.set_ylabel('N200 amplitude [a.u.]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('N200 amplitude (incongruent stimulus)')

    ax = axs[1,1]
    ax.scatter(S_ids_incongruent, latency_incongruent)
    ax.set_xlabel('S_id'); ax.set_ylabel('N200 latency [ms]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('N200 latency (inconguent stimulus)')

    for ax in axs.flat:
        ax.grid(True)

    plt.tight_layout()

    plt_path = os.path.join('data','Results','N200_scatter.png')
    plt.savefig(plt_path)

    plt.close()
