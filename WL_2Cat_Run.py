import pandas as pd
import numpy as NP
import WL_emcee as WLE

def WL_2Cat_Run(input_filename, output_prefix, N_halos, z_halo, n_walkers, n_steps, n_burn, n_threads, halo_bounds='default', n_thin=10):
    '''
    '''
    #establish halo bounds for simulations if left as default
    if halo_bounds == 'default':
        bounds_halo = [[[0.2,0.3], [0.2,0.3], [0.0001,11.0]]]
    
    #loop through each of the inputs to run weak lensing code
    
    #read in the input catalog
    Cat = pd.read_csv(input_filename,sep=',',na_values='null')
    
    #sort data into appropriate array for WL code
    N_gal = NP.shape(Cat)[0]
    data = NP.zeros((N_gal, 7))
    data[:,0] = NP.array(Cat['ra'])
    data[:,1] = NP.array(Cat['dec'])
    data[:,2] = NP.array(Cat['z'])
    data[:,3] = NP.array(Cat['E1'])
    data[:,4] = NP.array(Cat['E2'])
    data[:,5] = NP.array(Cat['DE'])
    data[:,6] = NP.array(Cat['Catalog'])
    
    Cat_indicator = [0,1]
    
    WLE.WL_2Cat_Fit(data, Cat_indicator, bounds_halo, N_halos, z_halo, output_prefix, n_walkers, 
       n_steps, n_burn, n_threads=n_threads, n_thin=n_thin, verbose=False)

import sys
if len(sys.argv)<8:
    sys.stderr.write('Usage: blah input_filename, output_prefix, N_halos, z_halo, n_walkers, n_steps, n_burn, n_threads\n')
    print len(sys.argv)
    sys.exit(1)
(junk1, input_filename, output_prefix, N_halos, z_halo, n_walkers, n_steps, n_burn, n_threads) = sys.argv
WL_2Cat_Run(input_filename, output_prefix, int(N_halos), float(z_halo), int(n_walkers), int(n_steps), int(n_burn), int(n_threads))