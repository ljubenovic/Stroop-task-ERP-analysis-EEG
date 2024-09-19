import numpy as np
import matplotlib.pyplot as plt
import os
from picard import picard
from utils import *


def ICA_get_components(data, fs, S_id, ortho, extended, plotting=True):
    
    W_mat, M_mat, _ = picard(data, ortho=ortho, extended=extended, max_iter=1000)
    components = np.dot(M_mat, np.dot(W_mat, data))

    """
    W_mat, M_mat, components = picard(data, ortho, extended)  

    W_mat: Whitening matrix (data_whitened = W_mat*data)
    M_mat: Estimated unmixing matrix (components = unmixing_matrix* X_whitened)
    components: Estimated independent components

    """

    if plotting:
        n_components, n_samp = components.shape
        t = np.arange(0, n_samp/fs, 1/fs)
        freqs = np.fft.fftfreq(n_samp, 1/fs)[:n_samp // 2]

        plt.figure(figsize=(10,100))
        for i in range(n_components):
            ic = components[i,:]

            plt.subplot(2*n_components,1,2*i+1)
            plt.plot(t,ic,linewidth=0.8)
            plt.xlabel('t [s]')
            plt.ylabel('amplituda [a.u.]')
            plt.title('Vremenski oblik IC {}'.format(i))
            plt.grid(True, which='both', alpha=0.5)
            plt.xlim((0,33))
            
            fft_ic = np.fft.fft(ic)
            fft_ic = np.abs(fft_ic[:n_samp // 2])

            plt.subplot(2*n_components,1,2*i+2)
            plt.plot(freqs,fft_ic, linewidth=0.8)
            plt.xlabel('f [Hz]')
            plt.ylabel('magnituda [a.u.]')
            plt.title('Amplitudski spektar IC {}'.format(i))
            plt.grid(True, which='both', alpha=0.5)
            plt.xlim((0,50))
        plt.tight_layout()
        path = os.path.join('data','Graphs','S{}'.format(S_id),'ICA_components_zoomed.png')
        plt.savefig(path)

        plt.close()

        plt.figure(figsize=(10,100))
        for i in range(n_components):
            ic = components[i,:]

            plt.subplot(2*n_components,1,2*i+1)
            plt.plot(t,ic, linewidth=0.8)
            plt.xlabel('t [s]')
            plt.ylabel('amplituda [a.u.]')
            plt.title('Vremenski oblik IC {}'.format(i))
            plt.grid(True, which='both', alpha=0.5)

            fft_ic = np.fft.fft(ic)
            fft_ic = np.abs(fft_ic[:n_samp // 2])

            plt.subplot(2*n_components,1,2*i+2)
            plt.plot(freqs,fft_ic, linewidth=0.8)
            plt.xlabel('f [Hz]')
            plt.ylabel('magnituda [a.u.]')
            plt.title('Amplitudski spektar IC {}'.format(i))
            plt.grid(True, which='both', alpha=0.5)
            plt.xlim((0,50))
        plt.tight_layout()
        path = os.path.join('data','Graphs','S{}'.format(S_id),'ICA_components.png')
        plt.savefig(path)

        plt.close()

    save_to_csv(components, S_id, 'components', os.path.join('data','ICA components'))

    return W_mat, M_mat, components


def ICA_denoising(W_mat, M_mat, components, components_to_remove = None):

    components_filtered = components.copy()

    if np.any(components_to_remove):
        components_filtered[components_to_remove,:] = 0

    data_denoised = np.dot(np.linalg.pinv(W_mat), np.dot(np.linalg.pinv(M_mat), components_filtered))

    return data_denoised