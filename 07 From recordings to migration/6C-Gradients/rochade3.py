#-------------------------------------------------------------------------
# Filename: rolode.py
#  Purpose: Classes for ROtational Love wave Dispersion curve Estimation
#   Author: Alexander Wietek
#    Email: alexander.wietek@mytum.de
#
# Copyright (C) 2013 Alexander Wietek
#-------------------------------------------------------------------------
# I did some code cleanup and changed mainly the BAZ estimation so it gives
# now the to be expected values. Also changed is the error estimation 
# which is now representing the covariance estimate used for the kde_gaussian
# modelling 
# Joachim Wassermann 2014
#-------------------------------------------------------------------------
#
# Again, using the code extention from Esteban I changed the error estimation,
# applied a masked array for removing possible sensor noise values from the 
# FOG and did a bit of code polishing
# JoWa 2017
#-------------------------------------------------------------------------

"""
Functions for Rotational Love and Rayleigh wave Dispersion curve Estimation
@package rochade
@copyright:
    Alexander Wietek (alexwie@gmx.net)
    Jowachim Wassermann (j.wassermann@lmu.de)
@license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
import scipy as sp
from scipy.signal import hilbert
from scipy import stats
import math
import scipy.odr as odr
from obspy.core import UTCDateTime, read, Trace, Stream, Stats
from obspy.signal.cross_correlation import correlate,xcorr_max
import matplotlib
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import warnings
from datetime import datetime
from obspy.signal.trigger import trigger_onset,recursive_sta_lta,plot_trigger
from obspy.signal.array_analysis import array_processing


class window_estimator:
    """ Window_estimator class: class which stores all estimations
        for one time window

    @type phi: float
    @param phi: estimation of backazimuth
    @type cw: float
    @param cw: estimation of phase velocity
    @type err: float
    @param err: estimation of error 
    @type wght: float
    @param wght: weight for time window
    @type rotsq: float
    @param rotsq: sum of squares of vertical rotation rate 
    @type rotsq: float
    @param rotsq: sum of squares of transversal acceleration
    """

    def __init__(self,phi = 0, cw = 0, err = 0,wght = 0,rotsq = 0,accsq = 0,hov=False):
        self.phi = phi
        self.cw = cw
        self.err = err
        self.wght = wght
        self.rotsq = rotsq
        self.accsq = accsq
        self.hov = hov

class rolode_estimator:
    """ Rolode_estimator class. Stores all the calculations and
        estimations. Methods to calculate estimations and to do
        statistical evaluations are implemented here. Plots can 
        be produced, logfiles written, the results can be written 
        to a file, a method to calculate different weights is implemented.
        main method is window_estimation which does the estimations
        for the time windows

    
    @type f_lower: float array
    @param f_lower: array of highpass frequencies for calculations
    @type f_higher: float array
    @param f_higher: array of lowpass frequencies for calculations
    @type f_n: int
    @param f_n: number of frequencies
    @type windows: array of window_estimator lists
    @param windows: for each frequency there is a lists storing estimations
                    of the time windows. These estimations must be of type 
                    window_estimator.
    @type v_min: float
    @param v_min: minimum velocity to be displayed in histogram plots
    @type periods_per_window: float
    @type v_max: float
    @param v_max: max velocity to be displayed in histogram plots
    @param periods_per_window: specifies how many periods there are in one time
              window, window length = periods_per_window/ f
    """

    def __init__(self, f_lower = np.zeros((1,)), f_higher = np.zeros((1,)), 
                  periods_per_window = 15., v_min=40., v_max= 3000.,phi_0 = 0., cw_0 = 500.,trigger_params=None):
        if len(f_lower) != len(f_higher):
            msg = 'Bandpass frequency arrays must have same length'
            raise ValueError(msg)
        if (f_lower > f_higher).any():
            msg = 'Highpass frequencies must be lower than lowpass frequencies'
            raise ValueError(msg)

        self.f_lower = f_lower
        self.f_higher = f_higher 
        self.f_n = len(f_lower)

        self.windows =  [[] for i in range(self.f_n)]
        self.periods_per_window = periods_per_window
        if not trigger_params:
            self.tlta = 10.
            self.tsta = 0.5
            self.thres1 = 3.0
            self.thres2 = 0.5
            self.post = 20.
            self.pre = 1.
        else:
            self.tlta = trigger_params["lta"]
            self.tsta = trigger_params["sta"]
            self.thres1 = trigger_params["thres_1"]
            self.thres2 = trigger_params["thres_2"]
            self.post = trigger_params["post_t"]
            self.pre = trigger_params["pre_t"]
        self.calctime = 0
        self.start = 0
        self.end = 0
        self.phi_0 = phi_0
        self.cw_0 = cw_0
        self.sl_0 = 1/cw_0


        self.v_max = v_max
        self.v_min = v_min
        self.v_scale = v_min+(v_max-v_min)/2.
        self.means = np.zeros(self.f_n)
        self.medians = np.zeros(self.f_n)
        self.modes = np.zeros(self.f_n)
        self.modeheights = np.zeros(self.f_n)
        self.stds = np.zeros(self.f_n)
        self.means_hv = np.zeros(self.f_n)
        self.medians_hv = np.zeros(self.f_n)
        self.modes_hv = np.zeros(self.f_n)
        self.modeheights_hv = np.zeros(self.f_n)
        self.stds_hv = np.zeros(self.f_n)
        self.hov = False
        self.wineval = False
        self.mean_std_eval = False
        self.median_eval = False
        self.kde_mode_eval = False
        self.hist_arr =  [[] for i in range(self.f_n)]
        self.bins_arr =  [[] for i in range(self.f_n)]
        self.hist_nw_arr =  [[] for i in range(self.f_n)]
        self.bins_nw_arr =  [[] for i in range(self.f_n)]
        self.xkde_arr =  [[] for i in range(self.f_n)]
        self.ykde_arr =  [[] for i in range(self.f_n)]
        self.xkde_wghted_arr =  [[] for i in range(self.f_n)]
        self.ykde_wghted_arr =  [[] for i in range(self.f_n)]
        self.hist_arr_hv =  [[] for i in range(self.f_n)]
        self.bins_arr_hv =  [[] for i in range(self.f_n)]
        self.hist_nw_arr_hv =  [[] for i in range(self.f_n)]
        self.bins_nw_arr_hv =  [[] for i in range(self.f_n)]
        self.xkde_arr_hv =  [[] for i in range(self.f_n)]
        self.ykde_arr_hv =  [[] for i in range(self.f_n)]
        self.xkde_wghted_arr_hv =  [[] for i in range(self.f_n)]
        self.ykde_wghted_arr_hv =  [[] for i in range(self.f_n)]


    def append(self,window,i):
        """ Append an estimation of backazimuth and phase velocity
            
        @type window: window_estimator
        @param window: an estimation of a single time window
        @type i: int
        @param i: number of frequency to append the estimation to
        """
        if not isinstance(window,window_estimator):
            msg = 'Cannot append to rolode_estimator (wrong data type)'
            raise ValueError(msg)
        self.windows[i].append(window)

    def write(self,filename,firstRun=True):
        """ Write the rolode_estimator class to a file. uses pickle 
            package

        @type filename: string
        @param filename: filename of the pickle file
        """

        if firstRun:
            winfile = open(filename,"wb")
            pickle.dump(self,winfile)
            winfile.close()
        else:
            winfile = open(filename,"ab")
            pickle.dump(self,winfile)
            winfile.close()
        

    def read(self,filename):
        """ Read the rolode_estimator class tfrom a file. uses pickle 
            package

        @type filename: string
        @param filename: filename of the pickle file
        """
        winfile = open(filename,"rb")
        self = pickle.load(winfile)
        winfile.close()

    def printlog(self):
        """ Print calculations and specifications of 
            the rolode_estimator class.
        """
        linestring = "--------------------------------------------------------"
        print(linestring)
        print("ROLODE estimator class:")
        print("Parameters:")
        if self.start != 0 and self.end != 0:
            print("  from: " + self.start.ctime())
            print("  to:   " + self.end.ctime() + "\n")
        print("  highpass frequencies:" )
        print("    " + str(self.f_lower) ) 
        print("  lowpass frequencies:" )
        print("    " + str(self.f_higher) ) 
        print("  periods per window: %0.2f" %self.periods_per_window)
        if self.calctime != 0:
            print("Calculation time: %0.2f seconds" %self.calctime)
        print(linestring+ "\n")

    def writelog(self,filename,firstRun):
        """ Write a logfile of the calculations and specifications of 
            the rolode_estimator class.

        @type filename: string
        @param filename: filename of the logfile
        """
        if firstRun:
            logfile = open(filename,"w")
        else:
            logfile = open(filename,"a")
        linestring = "--------------------------------------------------------"
        logfile.write(linestring + "\n")
        logfile.write("ROLODE estimator class:\n")
        logfile.write("Parameters:\n")
        if self.start != 0 and self.end != 0:
            logfile.write("  from: " + self.start.ctime() + "\n")
            logfile.write("  to:   " + self.end.ctime() + "\n\n")
        logfile.write("  highpass frequencies:\n" )
        logfile.write("    " + str(self.f_lower)  + "\n") 
        logfile.write("  lowpass frequencies:"  + "\n")
        logfile.write("    " + str(self.f_higher)  + "\n") 
        logfile.write("  periods per window: %0.2f\n" %self.periods_per_window)
        if self.calctime != 0:
            logfile.write("Calculation time: %0.2f seconds\n" %self.calctime)
        logfile.write(linestring+ "\n")
        logfile.close()


    def writeresults(self,savepath):
        """ Write results for dispersion curve to file. The format is
            Frequency/Mean/Standard Deviation/Mode/Median. Results are
            written to a file called "results.csv" :

        @type savepath: string
        @param savepath: path to save file in
        """
        if not self.mean_std_eval: self.calc_mean_std()
        if not self.median_eval: self.calc_median()
        if not self.kde_mode_eval: self.calc_kde_mode()

        resultfile = open(savepath+"results.csv","w")
        resultfile.write("frequency, mean, std, mode, median\n")
        for f_num,f in enumerate((self.f_lower + self.f_higher)/2):
            resultfile.write("%0.2f, %0.3f, %0.3f, %0.2f, %0.2f\n"\
                              %(f,self.means[f_num],self.stds[f_num],
                                self.modes[f_num], self.medians[f_num]))
   
        resultfile.close()

    def writeresults2(self,savepath):
        """ Write results for H/V (ellipticity) curve to file. The format is
            Frequency/Mean/Standard Deviation/Mode/Median. Results are
            written to a file called "results.csv" :

        @type savepath: string
        @param savepath: path to save file in
        """
        if not self.mean_std_eval: self.calc_mean_std()
        if not self.median_eval: self.calc_median()
        if not self.kde_mode_eval: self.calc_kde_mode()

        resultfile = open(savepath+"results_hov.csv","w")
        resultfile.write("frequency, mean, std, mode, median\n")
        for f_num,f in enumerate((self.f_lower + self.f_higher)/2):
            resultfile.write("%0.2f, %0.3f, %0.3f, %0.2f, %0.2f\n"
                              %(f,self.means_hv[f_num],self.stds_hv[f_num],
                                self.modes_hv[f_num], self.medians_hv[f_num]))
   
        resultfile.close()

    def f_phi_cw (self,phi_sl,acc_ne):
        """ Model function for Orthogonal Distance regression:

        @type phi_sl: 2 component list or array where 
        @param phi_sl: phi_sl[0] is the backazimuth, phi_sl[1] is the slowness
        @type acc_ne: 2d numpy array of shape (2,n)
        @param acc_ne: acc_ne[0] is the North component of acceleration
                      acc_ne[1] is the East component of acceleration
        @return: numpy array , transversal acceleration in direction of 
                the backazimuth multiplied by lowness divided by 2
        """
        acc_n = acc_ne[0]
        acc_e = acc_ne[1]
        acc_t = np.zeros(len(acc_n))
        phi = phi_sl[0]
        sl = phi_sl[1]
        #This is the correct transform matrix of a right handed coordinate system with X=E,Y=N and phi pointing 
        #from the station to the source (backazimuth). The rotated coordinate R is pointing away from source-receiver and T is
        #perpendicular to it
        acc_t = acc_n * np.sin(phi) - acc_e * np.cos(phi)

        return  acc_t*sl/2.

    def f_phi_cw_ray (self, phi_cw, rotrate_ne):
        """ Model function for orthogonal distance regression: vertical acceleration,
            horizontal components of rotation according to relation of 
            vertical acceleration - horizontal rotation for rayleigh waves:
        
        @type phi_cw: 2 component list or array where 
        @param phi_cw: phi_cw[0] is the backazimuth, phi_cw[1] is the velocity
        @type rot_ne: 2d numpy array of shape (2,n)
        @param rot_ne: rot_ne[0] is the North component of rotation
                      rot_ne[1] is the East component of rotation
        
        """
        phi = phi_cw[0]
        cw = phi_cw[1]
        rotrate_n = rotrate_ne[0]
        rotrate_e = rotrate_ne[1]

        # The rotated coordinate or rotation will be TRANSVERSAL to the vector pointing away from
        # source - receiver
        # the rotation components are always + 90 to the translation components thus the Transverse 
        # rotation is out of phase with the vertical translation
        rot_t = rotrate_n * np.sin(phi) - rotrate_e * np.cos( phi) 
        return rot_t * cw


    def fit_azimuth(self,b, x):
        return b[0] * x + b[1]

    def f_cw_ray (self, cw, rotrate_t):
        """ 
        Very simple relationship between velocity and rotation rate (vector sum)
        
        @type cw: float
        @param cw:  is the velocity
        @type rot_ne: 1d numpy array
        @param rotrate_t: transverse component of rotation rate
        
        """
        return rotrate_t * cw

    def f_phi_rot (self,phi,acc_ne):
        """ Model function for Orthogonal Distance regression:

        @type phi_sl: 2 component list or array where 
        @param phi_sl: phi_sl[0] is the backazimuth, phi_sl[1] is the slowness
        @type acc_ne: 2d numpy array of shape (2,n)
        @param acc_ne: acc_ne[0] is the North component of acceleration
                      acc_ne[1] is the East component of acceleration
        @return: numpy array , transversal acceleration in direction of 
                the backazimuth multiplied by lowness divided by 2
        """
        acc_n = acc_ne[0]
        acc_e = acc_ne[1]
        acc_t = np.zeros(len(acc_n))
        #This is the correct transform matrix of a right handed coordinate system with X=E,Y=N and phi pointing 
        #from the station to the source (backazimuth). The rotated coordinate R is pointing away from source-receiver and T is
        #perpendicular to it
        acc_t = acc_n * np.sin(phi) - acc_e * np.cos(phi)

        return  acc_t




    def set_weights(self, method = "normed", exp = 1.): 
        """ Method to set different weights for window estimations
            for statistical analysis
        
        @type method: string
        @param method: Characterizes the weighting method, must be one of 
                       the following: 
                       "standard" ... wght = np.sum(rotrate**2)/(err)
                       "normed"   ... wght = (1 - err/np.sum(rotrate**2))^exp
                       "uniform"  ... wght = 1
        @type exp: float
        @param exp: if the weighing method "normed" is chosen this is the
                    exponent
        """
        if method == "unbounded":
            for f_num in range(0,self.f_n):
                win_n = len(self.windows[f_num])
                for i in range(0,win_n):
                    try:
                        self.windows[f_num][i].wght = 1./self.windows[f_num][i].wght
                    except:
                        msg ="Not possible for %d-%d ..."%(f_num,i)
                        warnings.warn(msg)
                        continue
        elif method == "normed":
            for f_num in range(0,self.f_n):
                win_n = len(self.windows[f_num])
                for i in range(0,win_n):
                    try:
                        if self.windows[f_num][i].wght < 1.:
                            self.windows[f_num][i].wght = (1. - self.windows[f_num][i].wght)**exp
                        else: self.windows[f_num][i].wght = 1e-4
                    except:
                        msg = "Not possible for %d-%d ..."%(f_num,i)
                        warnings.warn(msg)
                        continue
        elif method == "uniform":
            for f_num in range(0,self.f_n):
                win_n = len(self.windows[f_num])
                for i in range(0,win_n):
                    self.windows[f_num][i].wght = 1


    def calc_mean_std(self,verbose = True,hov=False):
        """ Calculates weighted mean and weighted standard deviation
 
        @type verbose: boolean 
        @param verbose: if True text is printed to check the progress of 
                      calculations 
        """
        if verbose :
                print("Calculating means and standard deviations...\n")
        if self.wineval:
            for f_num in range(0,self.f_n):
                try:
                    win_n = len(self.windows[f_num])
                    print("  Now at frequency f = %0.2f" 
                           %( (self.f_lower[f_num]*(np.sqrt(np.sqrt(2.))))))
                    phi_arr  = np.zeros((win_n,))
                    hv_arr  = np.zeros((win_n,))
                    cw_arr   =  np.zeros((win_n,))
                    err_arr  =  np.zeros((win_n,))
                    wght_arr = np.zeros((win_n,))

                    for i in range(0,win_n):
                        if hov == True:
                            hv_arr[i]  = self.windows[f_num][i].phi
                        else:
                            phi_arr[i]  = self.windows[f_num][i].phi
                        cw_arr[i]   = self.windows[f_num][i].cw
                        err_arr[i]  = self.windows[f_num][i].err
                        wght_arr[i] = self.windows[f_num][i].wght

                    # Calculate weighted means, median and standard deviation
                    self.means[f_num] = np.average(cw_arr,weights = wght_arr)
                    if hov == True:
                        self.means_hv[f_num] = np.average(hv_arr,weights = wght_arr)
                except:
                    msg = "not able to calculate... continuing"
                    warnings.warn(msg)
                    continue
                self.mean_std_eval = True
        else: 
            msg ="Not able to calculate means: use window_estimation() first!"
            warnings.warn(msg)

    def calc_median(self, verbose = True,hov=False):
        """ Calculates weighted median of phase velocities
        
        @type verbose: boolean 
        @param verbose: if True text is printed to check the progress of 
                      calculations 
        """
        if verbose :
                print("Calculating medians...\n")
        if self.wineval:
            for f_num in range(0,self.f_n):
                try:
                    win_n = len(self.windows[f_num])
                    print("  Now at frequency f = %0.2f" 
                           %( (self.f_lower[f_num] + self.f_higher[f_num])/2 ))
                    phi_arr  = np.zeros((win_n,))
                    hv_arr  = np.zeros((win_n,))
                    cw_arr   =  np.zeros((win_n,))
                    err_arr  =  np.zeros((win_n,))
                    wght_arr = np.zeros((win_n,))

                    for i in range(0,win_n):
                        if hov == True:
                            hv_arr[i]  = self.windows[f_num][i].phi
                        else:
                            phi_arr[i]  = self.windows[f_num][i].phi
                        cw_arr[i]   = self.windows[f_num][i].cw
                        err_arr[i]  = self.windows[f_num][i].err
                        wght_arr[i] = self.windows[f_num][i].wght

                    # Calculate histogram for weighted median
                    hist_kde, bins_kde = np.histogram(cw_arr, int((self.v_max - self.v_min)/self.v_min),
                                                  range = (self.v_min,self.v_max),
                                                  weights = wght_arr,
                                                  normed = False, 
                                                  density = False)
                    width_kde = 0.9*( bins_kde[1] - bins_kde[0] )
                    center_kde = ( bins_kde[:-1] + bins_kde[1:] ) / 2
                    cw_kde_wghted = []
                    kde_count = 0
                    for val in center_kde:
                        cw_kde_wghted += [val]*(int(hist_kde[kde_count]*100))
                        kde_count += 1
                    self.medians[f_num] = np.median(cw_kde_wghted)
                    if hov == True:
                        h_min = np.min(hv_arr)
                        h_max = np.max(hv_arr)
                        # Calculate histogram for weighted median
                        hist_kde, bins_kde = np.histogram(hv_arr, int(h_max - h_min),
                                                  range = (h_min,h_max),
                                                  weights = wght_arr,
                                                  normed = False, 
                                                  density = False)
                        width_kde = 0.9*( bins_kde[1] - bins_kde[0] )
                        center_kde = ( bins_kde[:-1] + bins_kde[1:] ) / 2
                        hv_kde_wghted = []
                        kde_count = 0
                        for val in center_kde:
                            hv_kde_wghted += [val]*(int(hist_kde[kde_count]*100))
                            kde_count += 1
                        self.medians_hv[f_num] = np.median(hv_kde_wghted)

                except:
                    msg = "Not able to calculate median.... continuing"
                    warnings.warn(msg)
                    continue
               
            self.median_eval = True
        else: 
            msg ="Not able to calculate medians: use window_estimation() first!"
            warnings.warn(msg)

    def calc_kde_mode(self,
                         bin_n = 100, kde_sigma = 0.25,
                         kde_res = 2000.,verbose = True,hov=False):
        """ Calculates weighted kernel density estimation and weighted mode 
        
        @type bin_n: int 
        @param bin_n: number of bins for histograms
        @type kde_sigma: float 
        @param kde_sigma: parameter for width of gaussian filter
        @type kde_res: float
        @param kde_res: resolution of histograms for calculating the
                        weighted mode
        @type verbose: boolean 
        @param verbose: if True text is printed to check the progress of 
                      calculations 
        """
        if self.wineval:
            kde_step = (self.v_max-self.v_min)/kde_res
            if verbose :
                print("Calculating KDEs and modes:\n")
            for f_num in range(0,self.f_n):
                if verbose:
                    print("  Now at frequency f = %0.2f" 
                           %( (self.f_lower[f_num] + self.f_higher[f_num])/2 ))
                win_n = len(self.windows[f_num])

                try:
                    phi_arr  = np.zeros((win_n,))
                    hv_arr  = np.zeros((win_n,))
                    cw_arr   =  np.zeros((win_n,))
                    err_arr  =  np.zeros((win_n,))
                    wght_arr = np.zeros((win_n,))

                    for i in range(0,win_n):
                        if hov == True:
                            hv_arr[i]  = self.windows[f_num][i].phi
                        else:
                            phi_arr[i]  = self.windows[f_num][i].phi
                        cw_arr[i]   = self.windows[f_num][i].cw
                        err_arr[i]  = self.windows[f_num][i].err
                        wght_arr[i] = self.windows[f_num][i].wght

                    # Perform kernel density estimation
                    density = stats.gaussian_kde(cw_arr)
                    density.covariance_factor = lambda : kde_sigma/10
                    density._compute_covariance()
                    xkde = np.arange(self.v_min,self.v_max,kde_step)
                    ykde = density.evaluate(xkde)
                    mode = xkde[np.argmax(ykde)]

                    # Calculate histogram with weights
                    hist, bins = np.histogram(cw_arr, bin_n, range = (self.v_min,self.v_max), 
                                        weights = wght_arr, normed = True,
                                        density = True)
                    width = 0.9 * ( bins[1] - bins[0] )
                    center = ( bins[:-1] + bins[1:] ) / 2

                    # Calculate histogram with no weights
                    hist_nw, bins_nw = np.histogram(cw_arr, bin_n, 
                                                range = (self.v_min,self.v_max), 
                                                normed = True, density = True)
             
                    # Calculate histogram for weighted KDE
                    hist_kde, bins_kde = np.histogram(cw_arr, int(self.v_max - self.v_min),
                                                  range = (self.v_min,self.v_max),
                                                  weights = wght_arr,
                                                  normed = False, 
                                                  density = False)
                    center_kde = ( bins_kde[:-1] + bins_kde[1:] ) / 2
                    cw_kde_wghted = []
                    kde_count = 0
                    for val in center_kde:
                        cw_kde_wghted += [val]*(int(hist_kde[kde_count]*100))
                        kde_count += 1
                    self.medians[f_num] = np.median(cw_kde_wghted)

                    # Perform weighted kernel density estimation
                    density_wghted = stats.gaussian_kde(cw_kde_wghted)
                    density_wghted.covariance_factor = lambda : kde_sigma
                    density_wghted._compute_covariance()
                    xkde_wghted = np.arange(self.v_min,self.v_max,kde_step)
                    ykde_wghted = density_wghted.evaluate(xkde)

                    # Storing data 
                    self.modes[f_num] = xkde_wghted[np.argmax(ykde_wghted)]
                    self.modeheights[f_num] = np.amax(ykde_wghted)
                    self.stds[f_num] = np.sqrt(density_wghted.covariance[0][0])
                    if self.stds[f_num] > self.modes[f_num]:
                        self.stds[f_num] = self.modes[f_num]
                    self.hist_arr[f_num] = hist
                    self.bins_arr[f_num] = bins
                    self.hist_nw_arr[f_num] = hist_nw
                    self.bins_nw_arr[f_num] = bins_nw
                    self.xkde_arr[f_num] = xkde
                    self.ykde_arr[f_num] = ykde
                    self.xkde_wghted_arr[f_num] = xkde_wghted
                    self.ykde_wghted_arr[f_num] = ykde_wghted

                    if hov == True:
                    # Perform kernel density estimation
                        h_min = np.min(hv_arr)
                        h_max = np.max(hv_arr)
                        h_step = (h_max-h_min)/kde_res
                        density = stats.gaussian_kde(hv_arr)
                        density.covariance_factor = lambda : kde_sigma/10
                        density._compute_covariance()
                        xkde = np.arange(h_min,h_max,h_step)
                        ykde = density.evaluate(xkde)
                        mode = xkde[np.argmax(ykde)]

                    # Calculate histogram with weights
                        hist, bins = np.histogram(hv_arr, bin_n, range = (h_min,h_max),
                                        weights = wght_arr, normed = True,
                                        density = True)
                        width = 0.9 * ( bins[1] - bins[0] )
                        center = ( bins[:-1] + bins[1:] ) / 2

                    # Calculate histogram with no weights
                        hist_nw, bins_nw = np.histogram(hv_arr, bin_n,
                                                range = (h_min,h_max),
                                                normed = True, density = True)

                    # Calculate histogram for weighted KDE
                        hist_kde, bins_kde = np.histogram(hv_arr, int(h_max - h_min),
                                                  range = (h_min,h_max),
                                                  weights = wght_arr,
                                                  normed = False,
                                                  density = False)
                        center_kde = ( bins_kde[:-1] + bins_kde[1:] ) / 2
                        hv_kde_wghted = []
                        kde_count = 0
                        for val in center_kde:
                            hv_kde_wghted += [val]*(int(hist_kde[kde_count]*100))
                            kde_count += 1
                        self.medians_hv[f_num] = np.median(hv_kde_wghted)

                    # Perform weighted kernel density estimation
                        density_wghted = stats.gaussian_kde(hv_kde_wghted)
                        density_wghted.covariance_factor = lambda : kde_sigma
                        density_wghted._compute_covariance()
                        xkde_wghted = np.arange(h_min,h_max,h_step)
                        ykde_wghted = density_wghted.evaluate(xkde)

                    # Storing data 
                        self.modes_hv[f_num] = xkde_wghted[np.argmax(ykde_wghted)]
                        self.modeheights_hv[f_num] = np.amax(ykde_wghted)
                        self.stds_hv[f_num] = np.sqrt(density_wghted.covariance[0][0])
                        if self.stds_hv[f_num] > self.modes_hv[f_num]:
                            self.stds_hv[f_num] = self.modes_hv[f_num]
                        self.hist_arr_hv[f_num] = hist
                        self.bins_arr_hv[f_num] = bins
                        self.hist_nw_arr_hv[f_num] = hist_nw
                        self.bins_nw_arr_hv[f_num] = bins_nw
                        self.xkde_arr_hv[f_num] = xkde
                        self.ykde_arr_hv[f_num] = ykde
                        self.xkde_wghted_arr_hv[f_num] = xkde_wghted
                        self.ykde_wghted_arr_hv[f_num] = ykde_wghted

                except:
                    msg = "Not able to calculate kde ... continuing"
                    warnings.warn(msg)
                    continue

            self.kde_mode_eval = True
            print("")
        else: 
            msg = "Not able to calculate KDE: use window_estimation() first!"
            warnings.warn(msg)

    def plot_polar(self,bins_n = 36, showfigs = False, savefigs = True,
                   savepath = "", figformat = [".ps",".eps",".jpg",".pdf"],dpi_=100):
        """ Plots distribution of slowness and backazimuth on a polar plot

        @type bins_n: int 
        @param bins_n: number of bins for the backazimuth
        @type showfigs: boolean 
        @param showfigs: if True the figures are shown
        @type savefigs: boolean 
        @param savefigs: if True the histograms are saved to file
        @type savepath: string
        @param savepath: path to save plots in
        @type figformat: list of strings 
        @param figformat: specify the file extensions for the file
                          format of the plot
        """



        N = int(360./5.)
        S = int(90./5.)
        cmap = cm.viridis

        abins = np.arange(N + 1) * 2*np.pi / N
        sbins = np.linspace(0, 5.0, S + 1)

        for f_num in range(0,self.f_n):
            try:
                f = (self.f_lower[f_num] + self.f_higher[f_num])/2 
                win_n = len(self.windows[f_num])

                phi_arr  = np.zeros((win_n,))
                cw_arr   =  np.zeros((win_n,))
                wght_arr = np.zeros((win_n,))
                for i in range(0,win_n):
                    phi_arr[i]  = self.windows[f_num][i].phi
                    cw_arr[i]   = self.windows[f_num][i].cw
                    wght_arr[i] = self.windows[f_num][i].wght
        
                pi = np.pi
                baz = phi_arr
                slow = 1000./cw_arr
                if self.v_min > 0.:
                    slowmax = 1000./self.v_min
                else:
                    slowmax = 1.0
                slowmax = 5.
                baz = np.radians(baz)
                baz[baz < 0.0] += 2*np.pi
                baz[baz > 2*np.pi] -= 2*np.pi

                # sum rel power in bins given by abins and sbins
                hist, baz_edges, sl_edges = np.histogram2d(baz, slow,
                    bins=[abins, sbins], weights=wght_arr,
                    normed= True)

                # transform to gradient
                #baz_edges = baz_edges / 180 * np.pi

                # add polar and colorbar axes
                fig = plt.figure(figsize=(8, 8))
                cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
                ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
                ax.set_theta_direction(-1)
                ax.set_theta_zero_location("N")

                dh = abs(sl_edges[1] - sl_edges[0])
                dw = abs(baz_edges[1] - baz_edges[0])
            # circle through backazimuth
                for i, row in enumerate(hist):
                    bars = ax.bar(((i + 1) * dw) * np.ones(N),
                      dh * np.ones(N),
                      width=dw, bottom=dh * np.arange(N),align='edge',
                      color=cmap(row / hist.max()),linewidth=0)

                #ax.set_xticks([0., pi/2., pi, 3./2.*pi])
                #ax.set_xticklabels(['N', 'E', 'S', 'W'])
                ax.set_title("Backazimuth / Slowness distribution (ROLODE)")
                i,j = np.unravel_index(hist.argmax(), hist.shape)
                ax.set_xlabel("Maximum at: %f for f = %0.2f [Hz]"%(baz_edges[i]*180./pi,f))
                # set slowness limits
                ax.set_ylim(0., slowmax)

                ColorbarBase(cax, cmap=cmap,
                         norm=Normalize(vmin=hist.min(), vmax=hist.max()))
                if savefigs: 
        #        for postfix in figformat:
        #            plt.savefig(savepath + "polar_f_%0.2f" %f + postfix)
                    plt.savefig(savepath + "polar_f_%0.2f" %f + ".png",dpi=dpi_,transparent=True)
                if showfigs: plt.show()

                plt.clf()

                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)

                hist_baz, bins_baz = np.histogram(baz, bins=int(360/2.),
                                                        range = (0,2*np.pi),
                                                        weights = wght_arr,
                                                        density = False)

                bins_number = 180  # the [0, 360) interval will be subdivided into this
                                # number of equal bins
                width = 2 * np.pi / bins_number

                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_yticklabels([])

                bars = ax.bar(bins_baz[:-1], hist_baz, width=width,bottom=0.0,alpha=.7,color="orange")

                plt.savefig(savepath + "polar_%02f_hist"%(f) + '.png', format="png", dpi=dpi_,transparent=True)
                plt.clf()

            except:
                msg = "Nothing to plot ... continuing"
                warnings.warn(msg)
                continue


    def plot_errorbar(self, showfigs = False, 
                        savefigs = True, savepath = "",
                        figformat = [".ps",".eps",".jpg",".pdf"]):
        """ Plots dispersion curve with errorbars (the errorbars are the
            standard deviations of the distributions of phase velocities)

        @type showfigs: boolean 
        @param showfigs: if True the figures are shown
        @type savefigs: boolean 
        @param savefigs: if True the histograms are saved to file
        @type savepath: string
        @param savepath: path to save plots in
        @type figformat: list of strings 
        @param figformat: specify the file extensions for the file
                          format of the plot
        """
        if not self.mean_std_eval: self.calc_mean_std()
        if not self.median_eval: self.calc_median()
        if not self.kde_mode_eval: self.calc_kde_mode()
        print("Plotting Dispersion curve with errorbars")
        plt.figure()
#        plt.errorbar((self.f_lower + self.f_higher)/2, self.means, 
#                      yerr = self.stds,
#                      label= "Means (weighted) with std. deviation", 
#                      linewidth = 2.0, color = "c")
#        plt.errorbar((self.f_lower + self.f_higher)/2,self.medians,
#                 yerr = self.stds,
#                 label= "Medians (weighted)", 
#                 linewidth = 2.0, color = '#FF6633')
        plt.errorbar((self.f_lower + self.f_higher)/2,self.modes,
                 yerr = self.stds,
                 label= "Modes (weighted)", 
                 linewidth = 2.0, color = "r")
        plt.ylim(self.v_min,self.v_max)
        plt.legend()
        plt.title("Estimation of dispersion curve c(f)")
        plt.xlabel('f [Hz]')
        plt.ylabel('c(f) [m/s]')
#        plt.semilogx()
        if savefigs: 
#            for postfix in figformat:
#                plt.savefig(savepath + "errorbar"+ postfix) 
                plt.savefig(savepath + "errorbar"+ ".png",dpi=600) 
        if showfigs: plt.show()
        plt.close('all')

    def plot_errorbar1(self, showfigs = False, 
                        savefigs = True, savepath = "",
                        figformat = [".ps",".eps",".jpg",".pdf"]):
        """ Plots dispersion curve with errorbars (the errorbars are the
            standard deviations of the distributions of phase velocities)

        @type showfigs: boolean 
        @param showfigs: if True the figures are shown
        @type savefigs: boolean 
        @param savefigs: if True the histograms are saved to file
        @type savepath: string
        @param savepath: path to save plots in
        @type figformat: list of strings 
        @param figformat: specify the file extensions for the file
                          format of the plot
        """
        if not self.mean_std_eval: self.calc_mean_std()
        if not self.median_eval: self.calc_median()
        if not self.kde_mode_eval: self.calc_kde_mode()
        print("Plotting H/V curve with errorbars")
        plt.figure()
        plt.errorbar((self.f_lower + self.f_higher)/2,self.modes_hv,
                 yerr = self.stds_hv,
                 label= "Modes (weighted)", 
                 linewidth = 2.0, color = "r")
        plt.ylim(self.v_min,self.v_max)
        plt.legend()
        plt.title("Estimation of H/V curve")
        plt.xlabel('f [Hz]')
        plt.ylabel('H/V')
#        plt.semilogx()
        if savefigs: 
#            for postfix in figformat:
#                plt.savefig(savepath + "errorbar"+ postfix) 
                plt.savefig(savepath + "hov_error"+ ".png",dpi=600) 
        if showfigs: plt.show()
        plt.close('all')

    def plot_errorbar2(self, showfigs = False, 
                        savefigs = True, savepath = "",
                        figformat = [".ps",".eps",".jpg",".pdf"]):
        """ Plots dispersion curve corrected by the H/V with errorbars (the errorbars are the
            standard deviations of the distributions of phase velocities)

        @type showfigs: boolean 
        @param showfigs: if True the figures are shown
        @type savefigs: boolean 
        @param savefigs: if True the histograms are saved to file
        @type savepath: string
        @param savepath: path to save plots in
        @type figformat: list of strings 
        @param figformat: specify the file extensions for the file
                          format of the plot
        """
        if not self.mean_std_eval: self.calc_mean_std()
        if not self.median_eval: self.calc_median()
        if not self.kde_mode_eval: self.calc_kde_mode()
        print("Plotting Dispersion curve with errorbars")
        plt.figure()
        plt.errorbar((self.f_lower + self.f_higher)/2,self.modes/self.modes_hv,
                 yerr = self.stds,
                 label= "Modes (weighted) and corrected by H/V", 
                 linewidth = 2.0, color = "r")
        plt.ylim(self.v_min,self.v_max)
        plt.legend()
        plt.title("Estimation of dispersion curve c(f)/tan(e)")
        plt.xlabel('f [Hz]')
        plt.ylabel('c(f) [m/s]')
#        plt.semilogx()
        if savefigs: 
#            for postfix in figformat:
#                plt.savefig(savepath + "errorbar"+ postfix) 
                plt.savefig(savepath + "errorbar_hvcorr"+ ".png",dpi=600) 
        if showfigs: plt.show()
        plt.close('all')

    def plot_histograms(self, showfigs = False, 
                        savefigs = True, savepath = "",
                        figformat = [".png"]):
                       # figformat = [".ps",".eps",".jpg",".pdf"]):
        """ Plots histograms for every frequency 

        @type showfigs: boolean 
        @param showfigs: if True the figures are shown
        @type savefigs: boolean 
        @param savefigs: if True the histograms are saved to file
        @type savepath: string
        @param savepath: path to save plots in
        @type figformat: list of strings 
        @param figformat: specify the file extensions for the file
                          format of the plot
        """
        if not self.mean_std_eval: self.calc_mean_std()
        if not self.median_eval: self.calc_median()
        if not self.kde_mode_eval: self.calc_kde_mode()
        print("Plotting histograms ...")
        for f_num in range(0,self.f_n):
            try:
                f = (self.f_lower[f_num] + self.f_higher[f_num])/2 
                hist = self.hist_arr[f_num] 
                bins = self.bins_arr[f_num] 
                hist_nw = self.hist_nw_arr[f_num] 
                bins_nw = self.bins_nw_arr[f_num] 
                xkde = self.xkde_arr[f_num]
                ykde = self.ykde_arr[f_num]
                xkde_wghted = self.xkde_wghted_arr[f_num] 
                ykde_wghted = self.ykde_wghted_arr[f_num] 

                width = 0.9 * ( bins[1] - bins[0] )
                center = ( bins[:-1] + bins[1:] ) / 2
                width_nw = 0.9*( bins_nw[1] - bins_nw[0] )
                center_nw = ( bins_nw[:-1] + bins_nw[1:] ) / 2
            
                plt.figure()
                plt.plot(xkde,ykde,label= "KDE (no weights)", color = "y",
                     linestyle='dashed',linewidth = 3.0)
                plt.plot(xkde_wghted,ykde_wghted,label= "KDE (with weights)",
                     color = 'b', linewidth = 3.0)
                plt.bar(center_nw, hist_nw, align = 'center', width = width_nw ,edgecolor='black',
                    hatch='/',color = 'y', alpha = 0.5, 
                    label = "no weights")
                plt.bar(center, hist, align = 'center', width = width, alpha = 0.5,hatch='*',edgecolor='black',
                    label ="with weights" )
                plt.axvline(x = self.modes[f_num],label = "Mode (weighted)", 
                        color = "r", linestyle='dotted',linewidth = 2.0)
#            plt.axvline(x = self.means[f_num],label = "Mean (weighted)", 
#                         color = "c", linewidth = 2.0)
#            plt.axvline(self.medians[f_num], label = "Median (weighted)",
#                        color = '#FF6633', linewidth = 2.0)
                plt.legend(loc='upper left')
                plt.title("Distribution of velocity for f = %0.2f [Hz]" %f ,
                      fontsize=14 )
                plt.xlabel('Velocity [m/s]' %f)
                plt.ylabel(r'Density')
                if savefigs: 
                    plt.savefig(savepath + "hist_f_%0.2f" %f + ".png",dpi=600) 
                if showfigs: plt.show()
                plt.close('all')

            except:
                msg = "Nothing to plot..."
                warnings.warn(msg)
                continue

    def plot_histograms2(self, showfigs = False,
                        savefigs = True, savepath = "",
                        figformat = [".png"]):
                       # figformat = [".ps",".eps",".jpg",".pdf"]):
        """ Plots histograms for every frequency 

        @type showfigs: boolean 
        @param showfigs: if True the figures are shown
        @type savefigs: boolean 
        @param savefigs: if True the histograms are saved to file
        @type savepath: string
        @param savepath: path to save plots in
        @type figformat: list of strings 
        @param figformat: specify the file extensions for the file
                          format of the plot
        """
        if not self.mean_std_eval: self.calc_mean_std()
        if not self.median_eval: self.calc_median()
        if not self.kde_mode_eval: self.calc_kde_mode()
        print("Plotting histograms ...")
        for f_num in range(0,self.f_n):
            f = (self.f_lower[f_num] + self.f_higher[f_num])/2
            hist = self.hist_arr[f_num]
            bins = self.bins_arr[f_num]
            hist_nw = self.hist_nw_arr[f_num]
            bins_nw = self.bins_nw_arr[f_num]
            xkde = self.xkde_arr_hv[f_num]
            ykde = self.ykde_arr_hv[f_num]
            xkde_wghted = self.xkde_wghted_arr_hv[f_num]
            ykde_wghted = self.ykde_wghted_arr_hv[f_num]

            width = 0.9 * ( bins[1] - bins[0] )
            center = ( bins[:-1] + bins[1:] ) / 2
            width_nw = 0.9*( bins_nw[1] - bins_nw[0] )
            center_nw = ( bins_nw[:-1] + bins_nw[1:] ) / 2

            plt.figure()
            plt.plot(xkde,ykde,label= "KDE (no weights)", color = "y",
                     linestyle='dashed',linewidth = 3.0)
            plt.plot(xkde_wghted,ykde_wghted,label= "KDE (with weights)",
                     color = 'b', linewidth = 3.0)
            plt.bar(center_nw, hist_nw, align = 'center', width = width_nw ,edgecolor='black',
                    hatch='/',color = 'y', alpha = 0.5,
                    label = "no weights")
            plt.bar(center, hist, align = 'center', width = width, alpha = 0.5,hatch='*',edgecolor='black',
                    label ="with weights" )
            plt.axvline(x = self.modes_hv[f_num],label = "Mode (weighted)",
                        color = "r", linestyle='dotted',linewidth = 2.0)
            plt.legend(loc='upper left')
            plt.title("Distribution of H/V for f = %0.2f [Hz]" %f ,
                      fontsize=14 )
            plt.xlabel('H/V [a.u.]' %f)
            plt.ylabel(r'Density')
            plt.xlim(0.,10.)

            if savefigs:
#                for postfix in figformat:
#              plt.savefig(savepath + "hist_f_%0.2f" %f + postfix) 
              plt.savefig(savepath + "hovhist_f_%0.2f" %f + ".png",dpi=600)
            if showfigs: plt.show()
            plt.close('all')


    def window_estimation_love(self,rotrate_z, accel_n, accel_e, xcorr = False, verbose = True,mask_value=0,body=False,trigger=False,pro=False):
        """ Calculates estimations of the phase velocity, backazimuth of
        signals, for time windows.

        @type rotrate_z: obspy Trace 
        @param rotrate_z: Vertical rotation rate in rad/s
        @type accel_n:  obspy Trace 
        @param accel_n: Acceleration in the North direction in m/s^2
        @type accel_e:  obspy Trace 
        @param accel_e: Acceleration in the East direction in m/s^2
        @type periods_per_window: float
        @param periods_per_window: Number of periods of the signal per
              time window (Note: the window size is adapted to the 
              respective frequency)
        @return: List of Window datatypes:
        """
        tstart = datetime.now()

        #Input checks
        if not (len(accel_n.data) == len(accel_e.data) == len(rotrate_z.data)):
            msg = 'Input signals must have same length'
            raise ValueError(msg)
     
        self.start = accel_n.stats['starttime']
        self.end = accel_n.stats['endtime']

        df = rotrate_z.stats.sampling_rate
        # we trigger in the (broad)band in which we do the anlysis 
        tdata = rotrate_z.copy()
        tdata.filter("bandpass",freqmin=self.f_higher[-1],freqmax=self.f_lower[0],zerophase=True)
        if trigger or pro:
            cft = recursive_sta_lta(tdata.data,int(self.tsta*df),int(self.tlta*df))
            on_off = trigger_onset(cft,self.thres1,self.thres2)
            #plot_trigger(tdata,cft,self.thres1,self.thres2)
        
        if verbose:
            print("ROLODE parameter estimation for seperate time windows\n")
            print("  from: " + self.start.ctime())
            print("  to:   " + self.end.ctime() + "\n")
        # Loop over frequencies
        for f_num in range(0,self.f_n):
            if self.f_lower[f_num] < (self.f_higher[f_num] - self.f_lower[f_num]):
                       bandwidth = self.f_lower[f_num];
            else:
                       bandwidth = self.f_higher[f_num] - self.f_lower[f_num];
            win_len = (self.periods_per_window/bandwidth)
            win_samp = int(win_len*accel_n.stats.sampling_rate)
            win_len = float(win_samp/accel_n.stats.sampling_rate)
            win_n = int( (self.end - self.start) / win_len )
            fcenter = self.f_lower[f_num]*(np.sqrt(np.sqrt(2.)))
            if verbose:
                print("  Now at frequency f = %0.2f with bandwidth df = %0.2f" 
                      %( fcenter, bandwidth) ) 
            rot_trace = rotrate_z.copy()
            # we scale the rotation seismograms with a trial velocity....
            #rot_trace.data *= self.v_scale
            acc_n_trace = accel_n.copy()
            acc_e_trace = accel_e.copy()

            rot_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            rot_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)
            acc_n_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            acc_n_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)
            acc_e_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            acc_e_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)
 

            if trigger:
                for trig in on_off:
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_trace.stats.npts-1

                    rot_trace.data[t1:t2]=mask_value/10.

            if pro:
                for i,trig in enumerate (on_off):
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_trace.stats.npts-1
                on_off[i,0] = t1
                on_off[i,1] = t2
                if i == 0:
                    rot_trace.data[0:t1]=mask_value/10.
                elif i == len(on_off)-1:
                    rot_trace.data[t2:]=mask_value/10.
                else:
                    rot_trace.data[on_off[i-1,1]:t1]=mask_value/10.

            # Loop over time windows
            gaps = 0
            for i in range(0,win_n):
                # trim data for timewindow
                win_start = self.start + i*win_len
                win_end = self.start + (i+1)*win_len
                rot = rot_trace.copy()
                rot.trim(win_start, win_end)
                rotdata = rot.data
                acc_n = acc_n_trace.copy()
                acc_n.trim(win_start,win_end)
                acc_e = acc_e_trace.copy()
                acc_e.trim(win_start,win_end)
                try:
                    if mask_value > 0.:
                        condition1 = (np.abs(rotdata[:]) < mask_value) 
                        rot_z = np.ma.masked_array(rotdata,mask=condition1)
                        #rot_z=rot_z.compressed()
                        acc_ne = np.empty((2,rot_z.shape[0]))
                        acc_ne[0] = np.ma.masked_array(acc_n.data,mask=condition1)
                        acc_ne[1] = np.ma.masked_array(acc_e.data,mask=condition1)
                       
                    else:
                        rot_z = rotdata
                        acc_ne = np.empty((2,rot_z.shape[0]))
                        acc_ne[0] = acc_n.data
                        acc_ne[1] = acc_e.data

                    if acc_ne.shape[1] <2:
                        gaps+=1
                    else:
                        if xcorr == False:
                            # Calculate optimal direction with ODR ... fast but not precise
                            phi_0 =  np.random.uniform(-1.,1.)*np.pi
                            sl_0 = np.random.random_sample()*self.sl_0

                            odr_data = odr.Data(acc_ne,rot_z)
                            odr_model = odr.Model(fcn = self.f_phi_cw)
                            odr_opti = odr.ODR(odr_data,odr_model, np.array([phi_0,sl_0]),maxit=20) 
                            odr_opti.set_job(fit_type = 0, deriv = 1, var_calc = 2)

                            odr_output = odr_opti.run()
                            phi_opt = odr_output.beta[0]
                            if body == True:
                                cw_opt = 2./odr_output.beta[1]
                            else:
                                cw_opt = 1/odr_output.beta[1]

                            #scale it back
                            #cw_opt *= (self.v_scale)
                            err = odr_output.sum_square
                            delta = odr_output.sum_square_delta
                            eps = odr_output.sum_square_eps
                            rotsq = np.sum(rot.data**2)
                            inv_err = delta/np.sum(odr_output.y**2) + eps/rotsq
                            wght = np.sqrt(inv_err)
                            acc_t = 2*self.f_phi_cw([phi_opt,cw_opt],acc_ne)*cw_opt
                            accsq = np.sum(acc_t**2)

                            #cross check if the quadrant is correct 
                            t_acc = acc_ne[0]*np.sin(phi_opt) - acc_ne[1]*np.cos(phi_opt)
                            cc = correlate(t_acc,rot_z,shift=0)
                            shift,mcor = xcorr_max(cc)
                            if np.sign(mcor)<0:
                                phi_opt += np.pi
                                if (phi_opt > 2*np.pi): phi_opt -= 2*np.pi
                            
                            phi_opt = np.degrees(phi_opt)
                            window = window_estimator(phi_opt,np.abs(cw_opt),err,wght,rotsq,accsq)
                            self.append(window,f_num) 
                        else:
                            max_mcorr = -1.
                            t_baz = np.arange(0,np.pi,np.pi/45.)
                            for tb in t_baz:
                                t_acc = acc_ne[0]*np.sin(tb) - acc_ne[1]*np.cos(tb)
                                cc = correlate(t_acc,rot_z,shift=0)
                                mcor = cc[0]
                                #_,mcor = xcorr_max(cc)
                                if np.abs(mcor) > max_mcorr:
                                    max_mcorr = np.abs(mcor)
                                    phi_opt = tb
                                if np.sign(mcor)<0:
                                    phi_opt += np.pi
                                if (phi_opt > 2*np.pi): phi_opt -= 2*np.pi
                            t_acc = acc_ne[0]*np.sin(phi_opt) - acc_ne[1]*np.cos(phi_opt)
                            try:
                                cw_0 = np.random.random_sample() *self.cw_0
                                odr_data = odr.Data(rot_z,t_acc)
                                odr_model = odr.Model(fcn = self.f_cw_ray)
                                odr_opti = odr.ODR(odr_data,odr_model,\
                                        beta0=[cw_0])
                                odr_opti.set_job(fit_type = 0, deriv = 1, var_calc = 2)
                                odr_output = odr_opti.run()
                                if body == True:
                                    cw_opt = odr_output.beta[0]
                                else:
                                    cw_opt = odr_output.beta[0]/2.
                                err = odr_output.sum_square
                                delta = odr_output.sum_square_delta
                                eps = odr_output.sum_square_eps
                                rotzsq = np.sum(rot_z**2)
                                inv_err = delta/np.sum(odr_output.y**2) + eps/rotzsq
                                wght = np.sqrt(inv_err)
        
                                acc_t = self.f_cw_ray([cw_opt],rot_z)/(2*cw_opt)
                                accsq = np.sum(acc_t**2)
        
                                bazimuth = np.degrees(phi_opt)
        
                                window = window_estimator(bazimuth,cw_opt,err,wght,rotzsq,accsq)
                                self.append(window,f_num)
                            except:
                                msg + "something went wrong here"
                                warnings.warn(msg)
                                continue
                                
                except:
                    print("-", end="",flush=True)
                
        self.wineval = True
        tend = datetime.now()
        self.calctime = (tend-tstart).seconds
        if verbose:
            print("\nROLODE parameter estimation done.")
            print("Elapsed time: %0.2f seconds" %(tend-tstart).seconds)


    def window_estimation_rayleigh(self,accel_z, rotrate_n, rotrate_e, mask_value=0.,verbose = True,body=False,trigger=False,pro=False):
        """ Calculates estimations of the phase velocity, backazimuth of
        signals, for time windows.

        @type accel_z: obspy Trace 
        @param accel_z: Vertical accelertion in m/s^2
        @type rotrate_n:  obspy Trace 
        @param rotrate_n: rotational rate in the North direction in rad/s
        @type rotrate_e:  obspy Trace 
        @param rotrate_e: rotation rate in the East direction in rad/s
        @type periods_per_window: float
        @param periods_per_window: Number of periods of the signal per
              time window (Note: the window size is adapted to the 
              respective frequency)
        @return: List of Window datatypes:
        """

        tstart = datetime.now()

        #Input checks
        if not (len(rotrate_n.data) == len(rotrate_e.data) == len(accel_z.data)):
            msg = 'Input signals must have same length'
            raise ValueError(msg)

        self.start = rotrate_n.stats['starttime']
        self.end = rotrate_n.stats['endtime']

        df = rotrate_n.stats.sampling_rate
        # we trigger in the (broad)band in which we do the anlysis 
        tdata1 = rotrate_n.copy()
        tdata2 = rotrate_e.copy()
        tdata1.filter("bandpass",freqmin=self.f_higher[-1],freqmax=self.f_lower[0],zerophase=True)
        tdata2.filter("bandpass",freqmin=self.f_higher[-1],freqmax=self.f_lower[0],zerophase=True)
        if trigger or pro:
            cft = recursive_sta_lta(tdata1.data,int(self.tsta*df),int(self.tlta*df))
            on_off = trigger_onset(cft,self.thres1,self.thres2)
            cft = recursive_sta_lta(tdata2.data,int(self.tsta*df),int(self.tlta*df))
            on_off2 = trigger_onset(cft,self.thres1,self.thres2)

        if verbose:
            print("ROLODE parameter estimation for seperate time windows (Version 2) ")
            print('1st Rayleigh version (H[acc_Z]  VS. Rot_rate_T) \n' )
            print("  from: " + self.start.ctime())
            print("  to:   " + self.end.ctime() + "\n")

        #%% Loop over frequencies    
        for f_num in range(0,self.f_n):
            if self.f_lower[f_num] < (self.f_higher[f_num] - self.f_lower[f_num]):
                       bandwidth = self.f_lower[f_num];
            else:
                       bandwidth = self.f_higher[f_num] - self.f_lower[f_num];
            win_len = (self.periods_per_window/bandwidth)
            win_samp = int(win_len*rotrate_n.stats.sampling_rate)
            win_len = float(win_samp/rotrate_n.stats.sampling_rate)
            win_n = int( (self.end - self.start) / win_len )
            fcenter = self.f_lower[f_num]*(np.sqrt(np.sqrt(2.)))
            if verbose:
                print("  Now at frequency f = %0.2f with bandwidth df = %0.2f"
                      %( fcenter, bandwidth) )

            acc_z_trace = accel_z.copy()
            rot_n_trace = rotrate_n.copy()
            rot_e_trace = rotrate_e.copy()

            # again we scale the rotation seismograms
            #rot_n_trace.data *= self.v_scale
            #rot_e_trace.data *= self.v_scale

            acc_z_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            acc_z_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)
            rot_n_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            rot_n_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)
            rot_e_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            rot_e_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)

            if trigger:
                for trig in on_off:
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_n_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_n_trace.stats.npts-1

                    rot_n_trace.data[t1:t2]=mask_value/10.
                    rot_e_trace.data[t1:t2]=mask_value/10.

                for trig in on_off2:
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_e_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_e_trace.stats.npts-1

                    rot_n_trace.data[t1:t2]=mask_value/10.
                    rot_e_trace.data[t1:t2]=mask_value/10.

            if pro:
                for i,trig in enumerate(on_off):
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_n_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_n_trace.stats.npts-1
                    on_off[i,0] = t1
                    on_off[i,1] = t2

                    if i == 0:
                        rot_n_trace.data[0:t1]=mask_value/10.
                        rot_e_trace.data[0:t1]=mask_value/10.
                    elif i == len(on_off)-1:
                        rot_n_trace.data[t2:]=mask_value/10.
                        rot_e_trace.data[t2:]=mask_value/10.
                    else:
                        rot_n_trace.data[on_off[i-1,1]:t1]=mask_value/10.
                        rot_e_trace.data[on_off[i-1,1]:t1]=mask_value/10.

                for i,trig in enumerate(on_off2):
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_e_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_e_trace.stats.npts-1
                    on_off2[i,0] = t1
                    on_off2[i,1] = t2

                    if i == 0:
                        rot_n_trace.data[0:t1]=mask_value/10.
                        rot_e_trace.data[0:t1]=mask_value/10.
                    elif i == len(on_off)-1:
                        rot_n_trace.data[t2:]=mask_value/10.
                        rot_e_trace.data[t2:]=mask_value/10.
                    else:
                        rot_n_trace.data[on_off2[i-1,1]:t1]=mask_value/10.
                        rot_e_trace.data[on_off2[i-1,1]:t1]=mask_value/10.



            gap = 0
            # Loop over time windows
            for i in range(0,win_n):
                # trim data for timewindow
                win_start = self.start + i*win_len
                win_end = self.start + (i+1)*win_len
                accz = acc_z_trace.copy()
                accz.trim(win_start, win_end)
                rot_n = rot_n_trace.copy()
                rot_n.trim(win_start,win_end)
                rot_e = rot_e_trace.copy()
                rot_e.trim(win_start,win_end)
                vec_rot = np.sqrt(rot_n.data**2 + rot_e.data**2)/2.
                try:
                    if mask_value > 0.:
                        vec_rot_d = np.ma.masked_inside(vec_rot,-mask_value,mask_value)
                        try:
                            if vec_rot_d.mask == np.ma.nomask:
                                rot_n_d = rot_n.data
                                rot_e_d = rot_e.data
                                accz_d = accz.data
                        except:
                            masked = vec_rot_d.mask
                            vec_rot_d = vec_rot_d.compressed()
                            rot_n_d = rot_n.data[~masked]
                            rot_e_d = rot_e.data[~masked]
                            accz_d = accz.data[~masked]
                    else:
                        rot_n_d = rot_n.data
                        rot_e_d = rot_e.data
                        accz_d = accz.data

                    if len(accz_d) <2:
                        gap += 1
                    else:
                        # Calculate optimal direction with ODR
                        rotrate_ne = np.empty((2,len(rot_n_d)))
                        rotrate_ne[0]=rot_n_d
                        rotrate_ne[1]=-rot_e_d


                        phi_0 =  np.random.uniform(-1.,1.)*np.pi
                        cw_0 = np.random.random_sample()*self.cw_0

# **************** initial guess cw_0, finding cw ****************************
                        odr_data = odr.Data(rotrate_ne,accz_d)
                        odr_model = odr.Model(fcn = self.f_phi_cw_ray)
                        odr_opti = odr.ODR(odr_data,odr_model, np.array([phi_0,cw_0]),maxit=10) 
                        odr_opti.set_job(fit_type = 0, deriv = 1, var_calc = 2)
                        odr_output = odr_opti.run()
                        phi_opt = -1.0*odr_output.beta[0]
                        if body == True:
                            cw_opt = odr_output.beta[1]/2.
                        else:
                            cw_opt = odr_output.beta[1]

                        # scale it back!
                        #cw_opt *= self.v_scale
                        err = odr_output.sum_square
                        delta = odr_output.sum_square_delta
                        eps = odr_output.sum_square_eps
                        acczsq = np.sum(accz.data**2)
                        inv_err = delta/np.sum(odr_output.y**2) + eps/acczsq
                        wght = np.sqrt(inv_err)

                        rot_t = self.f_phi_cw_ray([phi_opt,cw_opt],rotrate_ne)/cw_opt
                        t_rot = rotrate_ne[0]*np.sin(phi_opt) - rotrate_ne[1]*np.cos(phi_opt)
                        rotsq = np.sum(rot_t**2)

                        cc = correlate(accz_d,t_rot,shift=0)
                        shift,mcor = xcorr_max(cc)
                        if np.sign(mcor)>0:
                            phi_opt += np.pi

                        if phi_opt > 2*np.pi: phi_opt -= 2*np.pi
                        if phi_opt < 0: phi_opt += 2*np.pi

                        phi_opt = np.degrees(phi_opt)


                        window = window_estimator(phi_opt,np.abs(cw_opt),err,wght,rotsq,acczsq)
                        self.append(window,f_num)
                except:
                    print('-')

        self.wineval = True
        tend = datetime.now()
        self.calctime = (tend-tstart).seconds
        if verbose:
            print("\nROLODE parameter estimation done.")
            print("Elapsed time: %0.2f seconds" %(tend-tstart).seconds)


    def window_estimation_rayleigh2(self,acc, rotrate_n, rotrate_e, verbose = True,mask_value=0,body=False,trigger=False,pro=False,rad=False,flinn=False):
        """ Calculates estimations of the phase velocity, backazimuth of
        signals, for time windows.

        @type acc: obspy Trace 
        @param acc: Vertical accelertion in m/s^2
        @type rotrate_n:  obspy Trace 
        @param rotrate_n: rotational rate in the North direction in rad/s
        @type rotrate_e:  obspy Trace 
        @param rotrate_e: rotation rate in the East direction in rad/s
        @type periods_per_window: float
        @param periods_per_window: Number of periods of the signal per
              time window (Note: the window size is adapted to the 
              respective frequency)
        @return: List of Window datatypes:
        """

        tstart = datetime.now()

        #Input checks
        if not (len(rotrate_n.data) == len(rotrate_e.data) == acc[0].stats.npts == acc[1].stats.npts == acc[2].stats.npts):
            msg = 'Input signals must have same length'
            raise ValueError(msg)

        self.start = rotrate_n.stats['starttime']
        self.end = rotrate_n.stats['endtime']

        df = acc[0].stats.sampling_rate

        # we trigger in the (broad)band in which we do the anlysis 
        tdata1 = rotrate_n.copy()
        tdata2 = rotrate_e.copy()
        tdata1.filter("bandpass",freqmin=self.f_higher[-1],freqmax=self.f_lower[0],zerophase=True)
        tdata2.filter("bandpass",freqmin=self.f_higher[-1],freqmax=self.f_lower[0],zerophase=True)
        if trigger or pro:
            cft = recursive_sta_lta(tdata1.data,int(self.tsta*df),int(self.tlta*df))
            on_off = trigger_onset(cft,self.thres1,self.thres2)
            cft = recursive_sta_lta(tdata2.data,int(self.tsta*df),int(self.tlta*df))
            on_off2 = trigger_onset(cft,self.thres1,self.thres2)

        if verbose:
            print("ROLODE parameter estimation for seperate time windows (Version 3) ")
            print('2st Rayleigh version (R_rot_T   VS. acc_z) \n' )
            print("  from: " + self.start.ctime())
            print("  to:   " + self.end.ctime() + "\n")

        #%% Loop over frequencies    
        for f_num in range(0,self.f_n):
            if self.f_lower[f_num] < (self.f_higher[f_num] - self.f_lower[f_num]):
                       bandwidth = self.f_lower[f_num];
            else:
                       bandwidth = self.f_higher[f_num] - self.f_lower[f_num];
            win_len = (self.periods_per_window/bandwidth)
            win_samp = int(win_len*rotrate_n.stats.sampling_rate)
            win_len = float(win_samp/rotrate_n.stats.sampling_rate)
            win_n = int( (self.end - self.start) / win_len )
            fcenter = self.f_lower[f_num]*(np.sqrt(np.sqrt(2.)))
            if verbose:
                print("  Now at frequency f = %0.2f with bandwidth df = %0.2f"
                      %( fcenter, bandwidth) )

            acc_z_trace = acc.select(component="Z")[0].copy()
            acc_n_trace = acc.select(component="N")[0].copy()
            acc_e_trace = acc.select(component="E")[0].copy()
            rot_n_trace = rotrate_n.copy()
            rot_e_trace = rotrate_e.copy()

            if verbose:
                print("Applied filter %f - %f"%(self.f_lower[f_num],self.f_higher[f_num]))

            acc_z_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            acc_z_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)
            acc_n_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            acc_n_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)
            acc_e_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            acc_e_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)
            rot_n_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            rot_n_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)
            rot_e_trace.filter( 'lowpass', freq = self.f_higher[f_num] ,zerophase=True)
            rot_e_trace.filter( 'highpass', freq = self.f_lower[f_num] ,zerophase=True)


            if trigger:
                for trig in on_off:
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_n_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_n_trace.stats.npts-1

                    rot_n_trace.data[t1:t2]=mask_value/10.
                    rot_e_trace.data[t1:t2]=mask_value/10.

                for trig in on_off2:
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_e_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_e_trace.stats.npts-1

                    rot_n_trace.data[t1:t2]=mask_value/10.
                    rot_e_trace.data[t1:t2]=mask_value/10.

            if pro:
                for i,trig in enumerate(on_off):
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_n_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_n_trace.stats.npts-1
                    on_off[i,0] = t1
                    on_off[i,1] = t2

                    if i == 0:
                        rot_n_trace.data[0:t1]=mask_value/10.
                        rot_e_trace.data[0:t1]=mask_value/10.
                    elif i == len(on_off)-1:
                        rot_n_trace.data[t2:]=mask_value/10.
                        rot_e_trace.data[t2:]=mask_value/10.
                    else:
                        rot_n_trace.data[on_off[i-1,1]:t1]=mask_value/10.
                        rot_e_trace.data[on_off[i-1,1]:t1]=mask_value/10.

                for i,trig in enumerate(on_off2):
                    t1 = trig[0]
                    t2 = trig[1]
                    if t1-int(self.pre*df) > 0:
                       t1 -= int(self.pre*df)
                    else:
                       t1 = 0
                    if t2+int(self.post*df) < rot_e_trace.stats.npts:
                       t2+=int(self.post*df)
                    else:
                       t2 = rot_e_trace.stats.npts-1
                    on_off2[i,0] = t1
                    on_off2[i,1] = t2

                    if i == 0:
                        rot_n_trace.data[0:t1]=mask_value/10.
                        rot_e_trace.data[0:t1]=mask_value/10.
                    elif i == len(on_off)-1:
                        rot_n_trace.data[t2:]=mask_value/10.
                        rot_e_trace.data[t2:]=mask_value/10.
                    else:
                        rot_n_trace.data[on_off2[i-1,1]:t1]=mask_value/10.
                        rot_e_trace.data[on_off2[i-1,1]:t1]=mask_value/10.


            # Loop over time windows
            for i in range(0,win_n):
                # trim data for timewindow
                win_start = self.start + i*win_len
                win_end = self.start + (i+1)*win_len
                
                accz = acc_z_trace.copy()
                accz.trim(win_start, win_end)
                accn = acc_n_trace.copy()
                accn.trim(win_start, win_end)
                acce = acc_e_trace.copy()
                acce.trim(win_start, win_end)
                rote = rot_e_trace.copy()
                rote.trim(win_start,win_end)
                rotn = rot_n_trace.copy()
                rotn.trim(win_start,win_end)

                if mask_value > 0.:
                    condition1 = (np.abs(rotn.data[:]) < mask_value) & (np.abs(rote.data[:]) < mask_value)
                    rot_n = np.ma.masked_array(rotn.data,mask=condition1)
                    rot_e = np.ma.masked_array(rote.data,mask=condition1)
                    rot_n=rot_n.compressed()
                    rot_e=rot_e.compressed()
                    acc_z=np.ma.masked_array(accz.data,mask=condition1)
                    acc_n=np.ma.masked_array(accn.data,mask=condition1)
                    acc_e=np.ma.masked_array(acce.data,mask=condition1)
                    acc_z=acc_z.compressed()
                    acc_n=acc_n.compressed()
                    acc_e=acc_e.compressed()
                else:
                    rot_n = rotn.data
                    rot_e = rote.data
                    acc_z = accz.data
                    acc_n = accn.data
                    acc_e = acce.data
                #movement of rotation rotN = -cos(baz) * rot; rotE = sin(baz) * rot
                # baz = -arctan(rotN/rotE)
                if len(rot_n) and len(rot_e)>2:
                    if flinn == False:
                        odr_data = odr.Data(rot_e,rot_n) 
                        odr_azi = odr.Model(fcn=self.fit_azimuth)
                        odr_opti = odr.ODR(odr_data,odr_azi, beta0=np.array([0,0]))
                        odr_output = odr_opti.run()


                        az_sl = odr_output.beta[0]
                        sl_error = odr_output.sd_beta[0]

                        #bazimuth = -np.arctan(az_sl)
                        bazimuth = -np.arctan2(az_sl,1)
                    else:
                        #movement of rotation rotN = sin(az) * rot; rotE = -cos(az) * rot
                        data = np.zeros((2, len(rot_n)), dtype=np.float64)
                        # East
                        data[0, :] = rot_e
                        # North
                        data[1, :] = rot_n
                        covmat = np.cov(data)
                        eigvec, eigenval, v = np.linalg.svd(covmat)
                        #bazimuth = -(np.arctan(eigvec[1][0]/eigvec[0][0]))
                        bazimuth = -(np.arctan2(eigvec[1][0],eigvec[0][0]))

                        # Rectilinearity defined after Montalbetti & Kanasewich, 1970
                        wght = (1.0 - np.sqrt(eigenval[1] / eigenval[0]))**0.5


            # sign of Xcorr should make the distinguish of correct quadrant possibel?!     
                    t_rot = rot_n*np.sin(bazimuth) - rot_e*np.cos(bazimuth)
                    if rad:
                        acc_r = -acc_n*np.cos(bazimuth) - acc_e*np.sin(bazimuth)
                        cc = correlate(acc_r,t_rot,shift=0)

                        shift,mcor = xcorr_max(cc)
                        if np.sign(mcor)<0:
                            bazimuth += np.pi
                    else:
                        cc = correlate(acc_z,t_rot,shift=0)
                        shift,mcor = xcorr_max(cc)
                        if np.sign(mcor)>0:
                            bazimuth += np.pi

                    if bazimuth > 2*np.pi:
                        bazimuth -= 2*np.pi
                    if bazimuth < 0.:
                        bazimuth += 2*np.pi

                    t_rot = rot_n*np.sin(bazimuth) - rot_e*np.cos(bazimuth)

# **************** initial guess cw_0, finding cw ****************************
                    try:
                        cw_0 = np.random.random_sample() *self.cw_0
                        odr_data = odr.Data(t_rot,acc_z)
                        odr_model = odr.Model(fcn = self.f_cw_ray)
                        odr_opti = odr.ODR(odr_data,odr_model,\
                                beta0=[cw_0])
                        odr_opti.set_job(fit_type = 0, deriv = 1, var_calc = 2)
                        odr_output = odr_opti.run()
                        if body == True:
                            cw_opt = odr_output.beta[0]*2.
                        else:
                            cw_opt = odr_output.beta[0]
                        err = odr_output.sum_square
                        delta = odr_output.sum_square_delta
                        eps = odr_output.sum_square_eps
                        acczsq = np.sum(acc_z**2)
                        inv_err = delta/np.sum(odr_output.y**2) + eps/acczsq
                        wght = np.sqrt(inv_err)

                        rot_t = self.f_cw_ray([cw_opt],acc_z)/cw_opt
                        rotsq = np.sum(rot_t**2)

                        phi_opt = np.degrees(bazimuth)

                        window = window_estimator(phi_opt,cw_opt,err,wght,rotsq,acczsq)
                        self.append(window,f_num)
                    except:
                        msg + "something went wrong here"
                        warnings.warn(msg)
                        continue


        self.wineval = True
        tend = datetime.now()
        self.calctime = (tend-tstart).seconds
        if verbose:
            print("\nROLODE parameter estimation done.")
            print("Elapsed time: %0.2f seconds" %(tend-tstart).seconds)


    def window_estimation_fk(self, vel_z, verbose = True):
        """ Calculates estimations of the phase velocity, backazimuth of
        signals, for time windows.

        @type vel: obspy Trace
        @param vel: array traces (vertical) in m/s
        @return: List of Window datatypes:
        """

        tstart = datetime.now()

        #Input checks
        self.start = vel_z[0].stats['starttime']
        self.end = vel_z[0].stats['endtime']

        if verbose:
            print("RoLoRaDe parameter estimation for seperate time windows (Version 4) ")
            print('fk based Rayleigh version \n' )
            print("  from: " + self.start.ctime())
            print("  to:   " + self.end.ctime() + "\n")

        #%% Loop over frequencies
        for f_num in range(0,self.f_n):
            if self.f_lower[f_num] < (self.f_higher[f_num] - self.f_lower[f_num]):
                       bandwidth = self.f_lower[f_num];
            else:
                       bandwidth = self.f_higher[f_num] - self.f_lower[f_num];
            win_len = (self.periods_per_window/bandwidth)
            win_samp = int(win_len*vel_z[0].stats.sampling_rate)
            win_n = int( (self.end - self.start) / win_len )
            fcenter = self.f_lower[f_num]*(np.sqrt(np.sqrt(2.)))
            if verbose:
                print("  Now at frequency f = %0.2f with bandwidth df = %0.2f"
                      %( fcenter, bandwidth) )

            vel_z_trace = vel_z.copy()
            slow_ = 1000./self.v_min
            ds_ = 1000./(self.v_max*2.)

            out = array_processing(vel_z_trace,
                     sll_x=-slow_,slm_x=slow_,sll_y=-slow_,slm_y=slow_,sl_s=ds_,
                     win_len=win_len,win_frac=0.5,
                     frqlow = self.f_lower[f_num],
                     frqhigh = self.f_higher[f_num],prewhiten=0,
                     semb_thres=0.001,vel_thres=-1e9,timestamp='mlabday',
                     stime=self.start,etime=self.end)

            for i in range(0,len(out[:,0])):
                wght = 1. - out[i,1]
                baz = out[i,3]
                apvel = (1./out[i,4])*1000.
                window = window_estimator(baz,apvel,1.,wght,out[i,2],1.)
                self.append(window,f_num)

        self.wineval = True
        tend = datetime.now()
        self.calctime = (tend-tstart).seconds
        if verbose:
            print("\nROLODE parameter estimation done.")
            print("Elapsed time: %0.2f seconds" %(tend-tstart).seconds)


