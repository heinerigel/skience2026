
import os, sys
import numpy as np
import obspy
from scipy.signal import periodogram, hilbert
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import pyproj
import glob
# import NonLinLocPy
from numba import jit
import SeisSrcInv
from obspy.imaging.beachball import beachball
from pathlib import Path
import pickle
import time
import gc


@jit(nopython=True)
def ncc(data, template):
    """Function performing cross-correlation between long waveform data (data) and template.
    Performs normalized cross-correlation in fourier domain (since it is faster).
    Returns normallised correlation coefficients."""
    n_samp_template = len(template)
    n_iters = len(data) - n_samp_template + 1
    ncc = np.zeros(n_iters)
    for i in range(len(ncc)):
        ncc[i] = np.sum(data[i:n_samp_template+i] * template / (np.std(data[i:n_samp_template+i]) * np.std(template))) / n_samp_template
    return(ncc)

def mt_ned2use(mt_in):
    """Function to convert mt from ned coordinates to use."""
    mt_out = np.array([mt_in[2], mt_in[0], mt_in[1], -mt_in[5], -mt_in[3], mt_in[4]])
    return mt_out

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_stf_and_attn_starting_model_params(real_data_array, recs, receivers_df, fs, nlloc_hyp_data, allowable_f_range=[25,150], vp=3600.):
    """Function to get source-time-function (dominant frequency) 
    and attenuation starting FWI model parameters.
    Based on spectral ratios method."""
    
    # Using spectral ratios and anelastic attenuation equation:

    N_rec = real_data_array.shape[0]

    # 0.i. Define reflection, radiation and geometrical spreading components:
    Rf_arr = np.ones(N_rec) # Reflection coefficients
    Rp_arr = np.ones(N_rec) # Radiation pattern correction factors
    Gs_arr = np.ones(N_rec) # Geometrical spreading factors
    # 0.ii. Find hypocentral distances and travel-times based on linear paths:
    trav_times = np.zeros(N_rec)
    hyp_dists = np.zeros(N_rec)
    for i in range(len(recs)):
        rec_row = receivers_df.loc[receivers_df['Name'] == recs[i]]
        hyp_dists[i] = np.sqrt(((nlloc_hyp_data.max_prob_hypocenter['x']*1000 - rec_row['x_m'])**2) + ((nlloc_hyp_data.max_prob_hypocenter['y']*1000 - rec_row['y_m'])**2) + ((nlloc_hyp_data.max_prob_hypocenter['z']*-1000 - rec_row['Elevation']*1000)**2))
        trav_times[i] = hyp_dists[i] / vp
    # 0.iii. And update geometrical spreading factors to be 1/R dependent:
    Gs_arr = 1 / hyp_dists

    # 1.i. Calculate spectral amplitudes (amplitude at peak-frequency:
    # (i.e. assuming constant frequency Q)):
    A_all_recs = np.zeros(real_data_array.shape[0])
    fmax_all_recs = np.zeros(real_data_array.shape[0])
    for i in range(real_data_array.shape[0]):
        f, Pxx = periodogram(real_data_array[i,:], fs=fs)
        val, f_begin_idx = find_nearest(f, allowable_f_range[0])
        val, f_end_idx = find_nearest(f, allowable_f_range[1])
        A_all_recs[i] = np.sqrt(np.max(Pxx[f_begin_idx:f_end_idx]))
        fmax_all_recs[i] = f[f_begin_idx:f_end_idx][np.argmax(Pxx[f_begin_idx:f_end_idx])]
    mean_fmax = np.mean(fmax_all_recs)

    # 1.ii. And populate obs-vector and X-matrix:
    y_vec_list = []
    X_matrix_list = []
    for i in range(len(A_all_recs)):
        for j in range(len(A_all_recs)):
            # Add condition to avoid duplication (i.e. only half of symetric matrix):
            if i < j:
                # Get y obs vector:
                y_vec_list.append( np.log(A_all_recs[i] / A_all_recs[j]) - np.log( (Rf_arr[i]*Rp_arr[i]*Gs_arr[i]) / (Rf_arr[j]*Rp_arr[j]*Gs_arr[j]) ) )
                # Get X model matrix:
                X_matrix_curr_row = np.zeros(N_rec)
                X_matrix_curr_row[i] = -np.pi * fmax_all_recs[i]
                X_matrix_curr_row[j] = np.pi * fmax_all_recs[j]
                X_matrix_list.append(X_matrix_curr_row)   
    y_vec = np.array(y_vec_list)
    X_matrix = np.array(X_matrix_list)

    # 2. Perform inversion to find t* values:
    # Solve without positive definite constraint on tstar:
    #tstar_vec, res, rank, aa = np.linalg.lstsq(X_matrix, y_vec)
    # Solve with positive definite constraint on tstar:
    n_cpu=-1
    model = linear_model.LassoCV(alphas=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], max_iter=10000000, n_jobs=n_cpu, selection='cyclic', positive=True)#, eps=1e-6, selection='random', tol=1e-6) # (Varies the regularisation value, alpha)
    reg = model.fit(X_matrix, y_vec)
    tstar_vec = model.coef_

    # 3. Find (mean) source amplitude, dominant frequency and Q:
    # Find Qs:
    Q_vec = trav_times / tstar_vec
    mean_Q = np.mean(Q_vec[Q_vec<np.inf])
    # 3.iii. Find A0:
    A0 = (1/N_rec) * np.sum(A_all_recs / (Rf_arr * Rp_arr * Gs_arr * np.exp(-np.pi * fmax_all_recs * tstar_vec))) # (1/N * sum( Ai/(Ri*Gi*e^{-pi f tstari}) ))
    
    return mean_fmax, mean_Q, A0


def find_das_archive_fname(das_archive_dir, event_origin_time):
    archive_das_fnames = glob.glob(os.path.join(das_archive_dir, "*.sgy"))
    time_diffs = np.zeros(len(archive_das_fnames))
    for i in range(len(archive_das_fnames)):
        fname = os.path.split(archive_das_fnames[i])[-1]
        fname_split = fname.split('_')
        fname_split[2] = fname_split[2].replace(".", ":")        
        fname_utcdatetime = obspy.UTCDateTime(fname_split[1]+"T"+fname_split[2]+".00")
        time_diffs[i] = event_origin_time - fname_utcdatetime
    time_diffs[time_diffs<0] = 1e6 # (to ensure that only positive archive files are selected)
    opt_archive_das_fname = archive_das_fnames[np.argmin(time_diffs)]
    return opt_archive_das_fname
    

