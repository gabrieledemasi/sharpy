# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import warnings
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, decimate
from scipy.signal.windows import tukey
from sharpy import welch
from gwpy.timeseries import TimeSeries

def fetch_data(ifo, tstart, tend, channel=None, path=None, verbose=0, ):

    """
        Fetch data for a particular event

        ifo: IFO name ('H1' etc)
        tstart, tend: start and end time to find
        path: Local path to save file. If file exists it will be read from disk rather than fetched
        channel: Channel to read from frame data. If 'GWOSC' will fetch open data
        verbose: Print some info

        Returns a gwpy.TimeSeries object
    """

    # If file was previously saved, open it.
    if path is not None and os.path.exists(path):
        tseries = TimeSeries.read(path,start=tstart,end=tend)
        if verbose:
            sys.stdout.write('Reading from local file '+path)
        return tseries

    # If not, then see if it is on GWOSC.
    if channel=='GWOSC':
        #When downloading public data, fetch them with the highest possible sampling rate, then down-sample internally, if required. This is needed to avoid incompatibilities between GWOSC down-sampling and the internal one. The actual function used to down-sample is the same, but differences in things like the length of data stretch can affect filtering at the borders and hence the Bayes Factors.
        tseries = TimeSeries.fetch_open_data(ifo, tstart, tend, sample_rate = 16384, verbose=verbose, cache=True, )
    else:
        # Read from authenticated data.
        if channel is None:
            raise Exception('Channel not specified when fetching frame data.')
        tseries = TimeSeries.get(channel, tstart, tend, verbose=verbose)
    if path is not None:
        tseries.write(path)

    return tseries

def downsample(strain, old_sampling_rate, new_sampling_rate):
    """
    Downsamples the signal to the desired sampling rate
    
    :param strain: input signal to downsample
    :type strain: array
    :param old_sampling_rate: sampling rate of the input signal
    :type old_sampling_rate: float
    :param new_sampling_rate: desired sampling rate
    :type new_sampling_rate: float
    
    :return: input *strain* downsampled to the desired sampling rate
    :rtype: array
    """
    strain = decimate(strain, int(old_sampling_rate/new_sampling_rate), zero_phase=True)
    return strain

def resize_time_series(inarr, N, dt, starttime, desiredtc):
    """
    Resizes the components of *inarr* to length *N* and aligns their peaks to the desired time of coalescence *desiredtc* in the segment
    
    :param inarr: two-dimentional array containing the time domain polarizations to resize and align, :math:`inarr=\mathrm{array}(h_+(t),h_{\\times}(t))`
    :type inarr: two-dimentional array
    :param N: required length of the resized array
    :type N: int
    :param dt: required time interval between the samples of the resized array
    :type dt: float
    :param starttime: time of the first sample of :math:`h_+(t)` and :math:`h_{\\times}(t)`
    :type starttime: float
    :param desiredtc: required peak time of the resized array
    :type desiredtc: float
    
    :return: resized :math:`h_+(t)` and :math:`h_{\\times}(t)`, with peak time at *desiredtc* 
    :rtype: array, array
    """
    
    # length of the array before resizing  
    waveLength = inarr.shape[0]

    # sample of the desired tc 
    tcSample = int(np.floor((desiredtc-starttime)/dt))
    
    # and the corresponding time
    injTc = starttime + tcSample*dt

    # sample at which tc is located in the array before resizing, using the square amplitude as reference
    waveTcSample = np.argmax(inarr[:,0]**2+inarr[:,1]**2)
    
    # procedure: copy a chunk of the actual array into a buffer of the length N, aligning the actual tc to the desired one
    # and cutting the extra samples
    
    # determine the first index of the chunk to copy: it's the very first sample of the array if it must be moved ahead to 
    # perform the alignment (copy all the samples before the peak, but shifted ahead in the buffer), it's the number of
    # samples to cut at the beginning if it must be moved back (move the array back by excluding the extra samples at the beginning)
    if (tcSample >= waveTcSample):
        waveStartIndex = 0
    else:
        waveStartIndex = waveTcSample - tcSample

    # samples after the peak before resizing
    wavePostTc = waveLength - waveTcSample
       
    # determine the first index of the buffer where it will be copied: it's the amount of samples by which the array must to be moved
    # ahead in the first case (copy all the samples before the peak, but shifted ahead in the buffer), it's the very beginning of the
    # buffer in the second case (move back the array by cutting the extra initial samples)
    if tcSample >= waveTcSample:
        bufstartindex = tcSample - waveTcSample
    else:
        bufstartindex = 0
    
    # determine the last index of the chunk to copy: copy all the samples after the peak if the new length of the waveform after
    # the alignment, wavePostTc + tcSample, is shorter than the buffer; cut the extra samples after the peak otherwise
    
    # determine the last index of the buffer where it will be copied: it's the new length of the waveform wavePostTc + tcSample if
    # it is shorter than the buffer, it's the last index of the buffer otherwise (cut the extra samples)
    if (wavePostTc + tcSample <= N):
        bufendindex  = wavePostTc + tcSample
        waveEndIndex = waveLength
    else:
        bufendindex  = N
        waveEndIndex = waveStartIndex + bufendindex - bufstartindex

    # buffer length
    bufWaveLength = bufendindex - bufstartindex;

    # allocate the arrays of zeros which work as a buffer
    hp = np.zeros(N,dtype = np.float64)
    hc = np.zeros(N,dtype = np.float64)
    
    # copy the waveform over
    hp[bufstartindex:bufstartindex+bufWaveLength] = inarr[waveStartIndex:waveEndIndex,0]
    hc[bufstartindex:bufstartindex+bufWaveLength] = inarr[waveStartIndex:waveEndIndex,1]

    return hp,hc
    
def resize_nan_strain(strain, start_time, trig_time, chunk_size, sampling_rate):
    """
    Finds the first segment of *NaN* values in the strain and removes it
    
    :param strain: input data to check for *NaN* values
    :type strain: array
    :param start_time: time corresponding to the first sample in the input data
    :type start_time: float
    :param trig_time: peak time of the input data
    :type trig_time: float
    :param chunk_size: length in seconds of the input data
    :type chunk_size: float
    :param sampling_rate: sampling rate of the input data
    :type sampling_rate: float
    
    :return: data strain without the first segment of *NaNs* and boolean that flags if *NaNs* have been found before the trigger time 
    :rtype: array, bool
    """
    
    # from https://git.ligo.org/john-veitch/ringdown/blob/master/ringdown/noise.py#L13
    
    # indices of the on-source chunk bounds and of the trigger time
    dt                 = 1.0/sampling_rate
    trig_time_idx      = int((trig_time-start_time)*sampling_rate)
    first_idx_onsource = int((trig_time-(chunk_size-1.0)-start_time)*sampling_rate)
    last_idx_onsource  = int((trig_time+(1.0)-start_time)*sampling_rate)
    
    # check for NaNs on the whole length of the strain
    first_nan_index    = None
    last_nan_index     = len(strain)-1
    
    # first check for on-source NaNs. The on-surce train will be used for the analysis and cannot contain NaNs
    for j in range(first_idx_onsource, last_idx_onsource):
        if(np.isnan(strain[j])):
            raise Exception("NaNs present on the on-source chunk, resize the on-source chunk to a segment which does not contain NaNs.")
    
    
    # locate the first NaN segment in the data
    for i in range(len(strain)):
        if (first_nan_index == None):
            if(np.isnan(strain[i])):
                first_nan_index = i
        else:
            if not(np.isnan(strain[i])):
                last_nan_index = i-1
                break

    # since we raise an error in the case where the NaNs overlap with the on-source chunk, now the only possibility is that the last NaN idx is either to the left or to the right with respect to the trigger time
    if(last_nan_index < trig_time_idx):
        sys.stdout.write('Nans present in the [{0}, {1}]s interval (before signal).\nResizing data to the [{2}, {3}]s interval. Remaining length: {4}s\n'.format(int(first_nan_index*dt), int(last_nan_index*dt), int((last_nan_index+1)*dt), int((len(strain))*dt), int((len(strain)-(last_nan_index+1))*dt)))
        
        # if there are NaNs before the trigtime, need to adjust starttime. Flag this case.
        NaNAtTheBeginning = True
        
        return strain[last_nan_index+1:], NaNAtTheBeginning
    else:
        sys.stdout.write('Nans present in the [{0}, {1}]s interval (after signal).\nResizing data to the [{2}, {3}]s interval. Remaining length: {4}s\n'.format(int(first_nan_index*dt), int(last_nan_index*dt), 0, int((first_nan_index-1)*dt), int((first_nan_index-1)*dt)))
        NaNAtTheBeginning = False
        
        return strain[:first_nan_index-1], NaNAtTheBeginning


def load_data(fname,
              ifo,
              chunk_size       = 4.0,
              trigtime         = None,
              sampling_rate    = 4096,
              psd_file         = None,
              psd_method       = 'welch',
              download_data    = 0,
              datalen_download = 64,
              channel          = '',
              gwpy_tag         = None):
    """
    Processes real data and returns the necessary quantities for the analysis
    
    :param fname: path to the data strain, with file extension 
    :type fname: string
    :param chunk_size: length in seconds of the data segment to analyze
    :type chunk_size: float
    :param trigtime: trigger time of the input data (seconds)
    :type trigtime: float
    :param sampling_rate: desired sampling rate, *optional*. If specified, data are downsampled to the indicated value
    :type sampling_rate: float or *None*
    :param psd_file: path to the power spectral density of the detector, *optional*. If *None*, the PSD is estimated from the input data
     through the Welch method
    :type psd_file: string or *None*
    
    :return:
        - times of the data segment to analyze (seconds)
        - windowed time domain data segment (has length *chunk_size* and extends to one second after *trigtime*)
        - frequencies corresponding to the data segment and to its sampling rate
        - frequency domain data segment 
        - power spectral density of the detector over the relevant frequencies
    :rtype: array, array, array, array, array
    """
    
    if not(download_data):
        warnings.warn("The file name is expected to follow LIGO-Virgo conventions 'DET-FRAMETYPE-STARTTIME--DATALEN.txt', e.g.: 'H-H1_GWOSC_4_V1-1126259446-32.txt'. See https://www.gw-openscience.org for more infomation.")

        # extract some metadata from the file name
        file_name               = (fname.split('/')[-1])
        ifo,fr_type,starttime,T = (file_name.split('.')[0]).split('-')
        starttime               = float(starttime)
        T                       = float(T)
        
        sys.stdout.write('Loading {0} starting at {1} length {2}s\n'.format(fname,starttime,T))
        
        # strain information
        rawstrain = np.loadtxt(fname)
        Nfull     = len(rawstrain)
        dt        = T/Nfull

    else:
        sys.stdout.write('\nUsing GWPY to download data.\n')
        
        T         = float(datalen_download)
        starttime = int(trigtime)-(T/2.)
        endtime   = int(trigtime)+(T/2.)
        tseries   = fetch_data(ifo, starttime, endtime, channel=channel, path=None, verbose=2, )
        rawstrain = np.array(tseries.data)
        Nfull     = len(rawstrain)
        dt        = tseries.dt.to_value()
        sys.stdout.write('\nLoaded channel {} starting at {} length {}s.\n'.format(channel,starttime,T))

    # exclude NaN values from gwosc data: apply iteratively resize_nan_strain until no more NaNs segments are found and adjust the strain duration
    # from https://git.ligo.org/john-veitch/ringdown/blob/master/ringdown/noise.py#L152
    no_nans = 0

    while(no_nans == 0):
        if((np.isnan(rawstrain).any())):
            sys.stdout.write('Nans found in the data. Resizing strain.\n')
            rawstrain, NaNAtTheBeginning = resize_nan_strain(rawstrain, starttime, trigtime, chunk_size, 1./dt)
            N                            = len(rawstrain)
            deltaT                       = dt*(Nfull-N)
            T                            = T-deltaT
            
            if NaNAtTheBeginning: starttime += deltaT
            Nfull = N
            
        else:
            no_nans = 1

    # sampling rate (Hz)
    srate=1./dt
    
    # downsample the data segment if requested
    if sampling_rate is not None:
        strain = downsample(rawstrain, srate, sampling_rate)
        srate  = sampling_rate
        dt     = 1./srate
    else:
        strain = rawstrain
    
    # find the index corresponding to the trigger time
    index_trigtime = int((trigtime-starttime)*srate)
    
    # number of samples in the chunk
    chunksize = int(chunk_size*srate)
    
    # start time of the signal chunk
    # we want the trigger time 1s before the end of the segment
    index_chunk_start = index_trigtime - int(srate*(chunk_size-1))
    chunk_start       = starttime+dt*index_chunk_start

    # time-domain signal chunk
    signal_chunk = np.zeros(chunksize,dtype=np.float64)
    for i in range(chunksize):
        signal_chunk[i] = strain[index_chunk_start+i]
    
    # and corresponding frequencies
    frequencies = np.fft.rfftfreq(chunksize, dt)
    
    # compute the Tukey window, with a symmetric padding of 0.4/T
    padding = 0.4/chunk_size
    window  = tukey(chunksize,padding)
    
    # window the data
    signal_chunk    *= window
    windowNorm      = chunksize/np.sum(window**2)
    SQRTwindowNorm  = np.sqrt(windowNorm)
    
    # compute the frequency domain strain, accounting for the window normalisation that takes away some energy
    sf = np.fft.rfft(signal_chunk)*SQRTwindowNorm*dt
    
    mesa_object = None
    # power spectral density: either passed by the user or computed through the Welch or the MESA method
    if psd_file is None:
        
        if psd_method == 'welch':
            sys.stdout.write("Estimating power spectral density with the Welch method\n")
            
            # compute the PSD by removing the signal chunk
            freqs, psd = welch.psd(np.delete(strain,range(index_chunk_start,index_chunk_start+chunksize,chunksize)),
                                   srate,
                                   chunk_size,
                                   window_function  = window,
                                   overlap_fraction = 0.5)

        elif psd_method == 'mesa':
            sys.stdout.write("Estimating power spectral density with the Burg method\n")
            # compute the PSD by removing the signal chunk
            freqs, psd, mesa_object = mesa.psd(np.delete(strain,range(index_chunk_start,index_chunk_start+chunksize,chunksize)), srate, chunk_size)
        elif psd_method == 'mesa-on-source':
            sys.stdout.write("Estimating the on-source power spectral density with the Burg method\n")
            freqs, psd, mesa_object = mesa.psd_onsource(signal_chunk, srate, chunksize)
        elif psd_method == 'marginalise':
            sys.stdout.write("Estimating the on-source power spectral density with the Burg method\n")
            sys.stdout.write("Will marginalise over the noise power spectrum\n")
            freqs, psd, mesa_object = mesa.psd_onsource(signal_chunk, srate, chunksize)
        else:
            sys.stdout.write("PSD estimation method {} unknown!\n".format(psd_method))
            exit(-1)
        psd_int = interp1d(freqs, psd, bounds_error=False, fill_value=np.inf)
        powerspectrum = psd_int(frequencies)
    else:
        # load the PSD file, if given
        sys.stdout.write("Using precomputed power spectrum\n")
        f, psd = np.loadtxt(psd_file,unpack=True)
        
        # interpolate the PSD over the relevant frequencies
        powerspectrum = interp1d(f, psd, bounds_error=False, fill_value=np.inf)(frequencies)
        
    # compute times and frequencies for convenience    
    times = chunk_start+np.linspace(0,chunk_size,chunksize)

    return times, signal_chunk, frequencies, sf, powerspectrum, mesa_object

def generate_data(psd_file,
                  T             = 16.0,
                  starttime     = 1126259446.,
                  sampling_rate = 4096.,
                  fmin          = 10,
                  fmax          = None,
                  zero_noise    = False,
                  asd           = False):
    """
    Generates Gaussian noise realizations from the noise curve of the interferometer (to be summed to the waveform template to analyze simulated data)
    
    :param psd_file: path to the power spectral density to be used for the noise generation 
    :type psd_file: string 
    :param T: length in seconds of the noise segment to simulate
    :type T: float
    :param starttime: first time of the noise segment (seconds)
    :type starttime: float
    :param sampling_rate: sampling rate of the simulation
    :type sampling_rate: float 
    :param fmin: lower bound of the simulation frequency band (Hz)
    :type fmin: float
    :param fmax: upper bound of the simulation frequency band (Hz)
    :type fmax: float
    :param zero_noise: flag to enable/disable the simulation of detector noise. If *True*, the analysis is performed without noise
    :type zero_noise: bool
    :param asd: boolean flag: set *True* if *psd_file* is really an amplitude spectral density (ASD) - i.e., the square root of the power spectral density.
     Set *False* otherwise
    :type asd: bool
    
    :return:
        - times of the noise simulation (seconds)
        - time domain simulated noise
        - frequencies of the noise simulation (corresponding to the data length and to the sampling rate)
        - frequency domain simulated noise
        - power spectral density of the detector over the relevant frequencies
    :rtype: array, array, array, array, array
    """
    
    # unpack the noise datafiles and convert to power spectral densities if necessary
    f, psd = np.loadtxt(psd_file, unpack=True)
    if asd is True : psd *= psd
    
    # interpolate the PSD
    psd_int = interp1d(f, psd, bounds_error=False, fill_value='extrapolate')
    
    # compute the times of the simulation
    df    = 1./T
    N     = int(sampling_rate*T)
    times = np.linspace(starttime,starttime+T,N)
    
    # filter out the bad bits
    kmin = int(fmin/df)
    kmax = int(fmax/df)+1
    
    # generate the frequency domain noise: consider a zero-mean Gaussian process with standard deviation given by the PSD
    frequencies      = df*np.arange(0,N/2.+1)
    frequency_series = np.zeros(len(frequencies), dtype = np.complex128)
    
    if zero_noise is False:
        for i in range(kmin, kmax):
            sigma               = 0.5*np.sqrt(psd_int(frequencies[i])/df)
            frequency_series[i] = np.random.normal(0.0,sigma)+1j*np.random.normal(0.0,sigma)

    # anti Fourier transform to get the time domain simulated noise
    time_series = np.fft.irfft(frequency_series, n=N)*df*N
    
    return times, time_series, frequencies, frequency_series, psd_int(frequencies), None