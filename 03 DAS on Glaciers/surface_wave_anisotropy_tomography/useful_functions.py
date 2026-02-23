import numpy as np
import os
import glob
import obspy 
from numba import jit


# Define some useful functions:

@jit(nopython=True, parallel=True)
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

@jit(nopython=True)
def ncc_max_tshift(data_1, data_2, max_tshift_idx):
    """Function performing cross-correlation between long waveform data (data) and template.
    Performs normalized cross-correlation in fourier domain (since it is faster).
    Returns normallised correlation coefficients.
    Note: both data_1 and data_2 should be the same length.
    max_tshift_idx is the maximum number of shifts to be applied to data_1.
    Returns the correlation coefficients for all shifts from -max_tshift_idx to max_tshift_idx.
    Note: Also needs to do receiver 1 - receiver 2 time shift.
    """
    n_tshifts = 2*max_tshift_idx + 1
    n_samp_template = len(data_2)
    ncc = np.zeros(n_tshifts)
    for i in range(len(ncc)):
        tshift_curr = i - max_tshift_idx
        ncc[i] = np.sum(data_1 * np.roll(data_2,tshift_curr) / (np.std(data_1) * np.std(data_2))) / n_samp_template
    return(ncc)


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
