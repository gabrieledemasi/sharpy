import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress all Keras input_shape/input_dim warnings
warnings.filterwarnings(
    "ignore",
    message=".*Do not pass an `input_shape`/`input_dim` argument to a layer.*",
    category=UserWarning
)


import numpy as np
from mlgw.GW_generator import GW_generator
gwgen = GW_generator()
from utils import McQ2Masses, tukey_window
import warnings
import warnings



def mlgw_template(params, detector_dictionary):

    mc                      = params[6]
    q                       = params[7]
    m1_msun, m2_msun        = McQ2Masses(mc, q)
    chi1                    = params[9] # Dimensionless spin
    chi2                    = params[10]
    # Time of coalescence in seconds
    phic                    = params[4] # Time of coalescence
    dist_mpc                = np.exp(params[2]) # Distance to source in Mpc
    inclination             = params[3] # Inclination Angle
    theta                   = np.array([m1_msun, m2_msun, chi1, chi2, dist_mpc, inclination, phic])  


    sampling_frequency      = detector_dictionary["sampling_frequency"]
    duration                = detector_dictionary["duration"]  # Duration of the data segment in seconds
    ## to subsitute with the precomputed value 
    time_array_mlgw         = detector_dictionary["Times_MLGW"]
    segment_length          = detector_dictionary["segment_length"]
    n                       = detector_dictionary["n_tukey"]
    w                       = detector_dictionary["w_tukey"]


    waveform_generator      = GW_generator()
    h_p,h_c                 = waveform_generator.get_WF(theta = theta,t_grid = time_array_mlgw, modes=(2,2))




    padding                 = 0.4/duration
    window                  = tukey_window(segment_length,n, w, padding)
    h_p                     = h_p*window
    h_c                     = h_c*window
    # windowNorm     = duration/jnp.sum(window**2)
    # SQRTwindowNorm = jnp.sqrt(windowNorm)
    h_p_f                   = np.fft.rfft(h_p) / sampling_frequency
    h_c_f                   = np.fft.rfft(h_c) / sampling_frequency
    freq_array_mlgw         = np.fft.rfftfreq(len(h_p),1/sampling_frequency)

    h_p_f_interp            = np.interp(detector_dictionary.Frequency,freq_array_mlgw,h_p_f,left=0.0,right=0.0)
    h_c_f_interp            = np.interp(detector_dictionary.Frequency,freq_array_mlgw,h_c_f,left=0.0,right=0.0) 


    #to match LALsuite convention
    phase_shift             = np.exp(1j * 2 * np.pi * detector_dictionary.Frequency *-1 +1j* np.pi)
    

    
    h_p_f_shifted           = h_p_f_interp*phase_shift
    h_c_f_shifted           = h_c_f_interp*phase_shift
    h_plus                  = h_p_f_shifted
    h_cross                 = h_c_f_shifted

    return h_plus, h_cross







