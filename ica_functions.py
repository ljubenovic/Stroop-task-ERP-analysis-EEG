import numpy as np
import matplotlib.pyplot as plt
import os
from picard import picard
from utility_functions import *

############################################################################################################
def ICA_get_components(data, fs, S_id, ortho, extended, plotting=True):
    
    W_mat, M_mat, _ = picard(data, ortho=ortho, extended=extended, max_iter=1000)
    components = np.dot(M_mat, np.dot(W_mat, data))

    if plotting:
        from visualisation_functions import ica_components_visualisation
        ica_components_visualisation(components, fs, S_id)

    save_to_csv(components, S_id, 'components', os.path.join('data','ICA components'))

    return W_mat, M_mat, components


############################################################################################################
def ICA_denoising(W_mat, M_mat, components, components_to_remove = None):

    components_filtered = components.copy()

    if np.any(components_to_remove):
        components_filtered[components_to_remove,:] = 0

    data_denoised = np.dot(np.linalg.pinv(W_mat), np.dot(np.linalg.pinv(M_mat), components_filtered))

    return data_denoised