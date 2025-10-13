
import os
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2000"
import os

import jax.numpy as np
#enable jax debugging
# os.environ["JAX_LOG_COMPILES"] = "1"
import jax



#uncomment these line if you want to run on cpu
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

# import jax
# import jax.numpy as np
 # Enable 64-bit precision
# jax.config.update("jax_enable_x64", True) 
import matplotlib.pyplot as plt
from functools import partial 

from functools import partial
import time 
import json



from jax.scipy.special import logsumexp
 


psd = "/home/gdemasi/SHARPy-GW/LIGO-P1200087-v18-aLIGO_DESIGN_psd.dat"


detector_settings = {
        "H1": {
            "psd_file"  : None, 
            "data_file" :'/home/gdemasi/gdemasi-work1/SHARPy-GW/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt',
            "channel"   : 'GWOSC',
            "download_data"  : True,
            
            
        },
        "L1": {
            "psd_file"  :None, 
            "data_file" : '/home/gdemasi/gdemasi-work1/SHARPy-GW/L-L1_GWOSC_4KHZ_R1-1126259447-32.txt',
            "channel"   :'GWOSC',
            "download_data"  : True,
          
        },
            
          
  


    }

from likelihood import GWNetwork, log_likelihood_det

gw_network = GWNetwork(detector_settings,
                       
                    #injection_parameters=truth,
    
                       )

# batched_detector = gw_network.batched_detector
detectors_list    = gw_network.detectors


log_likelihood = partial(log_likelihood_det, detector_list= detectors_list   )
import numpy as np 



# truth =  np.array([3.0, 1.0, 5.5, np.pi/2, np.pi, np.pi/2, 30.0, 0.7, 0.0, -1, 1])
truth =  np.array([3.0, 1.0, 5.5, np.pi/2, np.pi, np.pi/2, 31.0, 0.7, 0.0, -1, 1.])

# print(log_likelihood(truth))
# import sys 
# sys.exit()




import bilby 

class GW_likelihood(bilby.Likelihood):
    def __init__(self,):
        """
        A very simple Gaussian likelihood

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
        super().__init__(parameters={"ra": None,
                                        "dec": None,
                                        "logdistance": None,
                                        "theta_jn": None,
                                        "phi_ref": None,
                                        "pol": None,
                                        "mc": None,
                                        "q": None,
                                        "tc": None,
                                        "chi1": None,
                                        "chi2": None
                                    }
                                    )
        
    import numpy as np 

    def log_likelihood(self):
        ra          = self.parameters["ra"]
        dec         = self.parameters["dec"]
        logdistance = self.parameters["logdistance"]
        theta_jn    = self.parameters["theta_jn"]
        phi_ref     = self.parameters["phi_ref"]
        pol         = self.parameters["pol"]
        mc          = self.parameters["mc"]
        q           = self.parameters["q"]
        tc          = self.parameters["tc"]
        chi1        = self.parameters["chi1"]
        chi2        = self.parameters["chi2"]
        params      = np.array([ra, dec, logdistance, theta_jn, phi_ref, pol, mc, q, tc, chi1, chi2])
        
        log_likelihood_value = log_likelihood(params)
        
        return log_likelihood_value
            






likelihood = GW_likelihood()

priors = dict( 
    ra         = bilby.prior.Uniform(0, 2*np.pi, 'ra'),
    dec        = bilby.prior.Uniform(-np.pi/2, np.pi/2, 'dec'),
    logdistance= bilby.prior.Uniform(4.9, 8.7, 'logdistance'),
    theta_jn   = bilby.prior.Uniform(0, np.pi, 'theta_jn'),
    phi_ref    = bilby.prior.Uniform(0, 2*np.pi, 'phi_ref'),
    pol        = bilby.prior.Uniform(0, np.pi, 'pol'),
    mc         = bilby.prior.Uniform(25, 35, 'mc'),
    q          = bilby.prior.Uniform(0.4, 1., 'q'),
    tc         = bilby.prior.Uniform(-1e-1, 1e-1, 'tc' ),
    chi1       = bilby.prior.Uniform(-1., 1., 'chi1'),
    chi2       = bilby.prior.Uniform(-1., 1., 'chi2' )
    )



outdir = 'prova_dynesty_4000_correct_MLGW'
label  = 'test_run'
result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=4000,
        outdir=outdir,
        label=label,
        npool = 64, 
        
        
    )

result.plot_corner()

