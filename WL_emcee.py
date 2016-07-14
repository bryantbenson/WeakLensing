import WLTools as WLT
import emcee
import h5py
import numpy as NP
import time

def WL_Fit(Galaxy_input, bounds_halo, N_halos, z_halo, output_prefix, n_walkers, n_steps, n_burn, n_threads, n_thin=10, verbose=True):
    '''
    Galaxy_input: array of ra, dec, z, E1, E2, DE for all source galaxies
    Halo_input: array of initial values of ra, dec, z, M200 for all halos
    '''
    start_time = time.time()
    h_scale = 0.7
    Om = 0.3
        
    #Separate observed data from catalog for later use
    ra_gal = Galaxy_input[:,0]
    dec_gal = Galaxy_input[:,1]
    z_gal = Galaxy_input[:,2]
    e1_obs = Galaxy_input[:,3]
    e2_obs = Galaxy_input[:,4]
    DE = Galaxy_input[:,5]
    
    e1_obs_err = NP.std(e1_obs)
    e2_obs_err = NP.std(e2_obs)
    if verbose == True:
        print 'Checking e1 and e2 uncertainties'
        if e1_obs_err == 0:
            print 'E1 error is zero'
            return
        elif e2_obs_err == 0:
            print 'E2 error is zero'
            return
        else:
            print 'E1 and E2 stdevs are non zero'
            print e1_obs_err, e2_obs_err
    
    #setup output hdf5 file
    filename = output_prefix+'_result.hdf5'
    if verbose == True:
        print 'Setting up output file %s' %(filename)
    f = h5py.File(filename, 'w')
    
    # Save the user inputs.
    grp_input = f.create_group('/userinputs')
    grp_input.attrs['N_halos'] = N_halos
    grp_input.attrs['n_walkers'] = n_walkers
    grp_input.attrs['n_steps'] = n_steps
    grp_input.attrs['n_burn'] = n_burn
    grp_input['bounds_halo'] = bounds_halo
    grp_input.attrs['n_threads'] = n_threads
    
    # Save the raw data.
    grp_input['/data/ra_deg'] = ra_gal
    grp_input['/data/ra_deg'].attrs['units'] = 'degrees'
    grp_input['/data/dec_deg'] = dec_gal
    grp_input['/data/dec_deg'].attrs['units'] = 'degrees'
    grp_input['/data/z'] = z_gal
    grp_input['/data/z'].attrs['units'] = 'redshift'
    grp_input['/data/E1'] = e1_obs
    grp_input['/data/E2'] = e2_obs
    grp_input['/data/DE'] = DE
    
    # Concatinate the data structure.
    ra_gal = NP.reshape(ra_gal, (len(ra_gal),1))
    dec_gal = NP.reshape(dec_gal, (len(dec_gal),1))
    z_gal = NP.reshape(z_gal, (len(z_gal),))
    e1_obs = NP.reshape(e1_obs, (len(e1_obs),1))
    e2_obs = NP.reshape(e2_obs, (len(e2_obs),1))
    DE = NP.reshape(DE, (len(DE),1))
    
    # Number of dimensions (i.e. number of model parameters).
    # For each halo component there are 3 parameters (ra, dec, M200)
    n_dim = 3*N_halos
    if verbose == True:
        print 'Setting initial parameters for %i walkers' %(n_walkers)
    #find the mean and size of each prior boundary
    for i in range(N_halos):
        bounds_mid = []
        bounds_size = []
        bounds_mid.append(NP.mean(bounds_halo[i][0]))
        bounds_mid.append(NP.mean(bounds_halo[i][1]))
        bounds_mid.append(NP.mean(bounds_halo[i][2]))
    
        bounds_size.append(NP.max(bounds_halo[i][0])-NP.min(bounds_halo[i][0]))
        bounds_size.append(NP.max(bounds_halo[i][1])-NP.min(bounds_halo[i][1]))
        bounds_size.append(NP.max(bounds_halo[i][2])-NP.min(bounds_halo[i][2]))
    
    # Then estimate the initial starting locations of the halos as the mean of the prior range
    # Set the inital starting locations for each walker by choosing a random value inside the prior bounds
    p_0 = NP.random.uniform(-0.5, 0.5, (n_walkers,3))
    p_0[:,0] = p_0[:,0]*bounds_size[0] + bounds_mid[0]
    p_0[:,1] = p_0[:,1]*bounds_size[1] + bounds_mid[1]
    p_0[:,2] = p_0[:,2]*bounds_size[2] + bounds_mid[2]
    
    H0 = h_scale*100
    DF = WLT.DistanceFraction(H0, Om, z_gal, z_halo)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, WLT.lnprob, args=[ra_gal, dec_gal, z_gal, 
                            e1_obs, e2_obs, e1_obs_err, e2_obs_err, DE, DF, N_halos, z_halo, bounds_halo], threads=n_threads)
    if verbose == True:
        print 'Running the MCMC chains'
        
    sampler.run_mcmc(p_0, n_steps)
    
    if verbose == True:
        print 'Finished running MCMC chains'
    
    # Save the time series plots for each component and parameter.
    if verbose == True:
        print 'Creating time series figures'
    WLT.timeseries(sampler, N_halos, n_burn, output_prefix)

    # Compute the parameter medians and 68% quantiles.
    param_median = []
    param_1sigma = []
    for i in range(n_dim):
        param_median.append(NP.median(sampler.chain[:,n_burn:,i]))
        param_1sigma.append(NP.percentile(sampler.chain[:,n_burn:,i],
                            [16, 84]))
    
    # Locate most likely parameters
    mask = sampler.lnprobability[:,n_burn:] == NP.amax(sampler.lnprobability[:,n_burn:])
    param_maxlike = []
    for i in range(n_dim):
        param_maxlike.append(sampler.chain[:,n_burn:,i][mask][0])
        
    runtime = time.time() - start_time
    if verbose == True:
        print 'runtime was %f seconds' %(runtime)
        
    # Save the MCMC results to the hdf5 file.
    if verbose == True:
        print 'Saving results to hdf5 file'
    #Log Likelihood values
    f['/results/log_likelihoods'] = sampler.lnprobability
    # Parameter estimates.
    # Parameter labels
    param_label = ['RA', 'Dec', 'M200']
    param_units = ['degrees', 'degrees', '10^14 Solar Mass']
    
    grp_res = f.create_group('/results/parameters')
    grp_res.attrs['runtime'] = runtime
    
    for i in range(N_halos):
        # Create a group for each halo
        group_name = 'Halo{0}'.format(i)
        grp_comp = grp_res.create_group(group_name)
        for j in range(3):
            group_name = param_label[j]
            grp_param = grp_comp.create_group(group_name)
            grp_param.attrs['units'] = param_units[j]
            grp_param['median'] = param_median[i*3 + j]
            grp_param['max_likelihood'] = param_maxlike[i*3 + j]
            grp_param['quantiles'] = param_1sigma[i*3 +j]
            grp_param['quantiles'].attrs['quantile'] = '(16%, 84%)'
            grp_param['chain'] = sampler.chain[:, :, i*3 + j]
            grp_param['chain'].attrs['dimensions'] = '(n_step,n_walkers)'
    f.close()
    if verbose == True:
        print 'Finished MCMC analysis and results output'
    

def WL_2Cat_Fit(Galaxy_input, Cat_indicator, bounds_halo, N_halos, z_halo, output_prefix, n_walkers, n_steps, n_burn, n_threads=1, n_thin=10, verbose=True):
    '''
    Galaxy_input: array of ra, dec, z, E1, E2, DE, catalog # for all source galaxies
    Halo_input: array of initial values of ra, dec, z, M200 for all halos
    '''
    start_time = time.time()
    h_scale = 0.7
    Om = 0.3
    
    #Separate observed data from catalog for later use
    ra_gal = Galaxy_input[:,0]
    dec_gal = Galaxy_input[:,1]
    z_gal = Galaxy_input[:,2]
    e1_obs = Galaxy_input[:,3]
    e2_obs = Galaxy_input[:,4]
    DE = Galaxy_input[:,5]
    
    mask_Cat_1 = Galaxy_input[:,6] == Cat_indicator[0]
    mask_Cat_2 = Galaxy_input[:,6] == Cat_indicator[1]
        
    N_gal_1 = NP.sum(mask_Cat_1)
    N_gal_2 = NP.sum(mask_Cat_2)
    
    e1_obs_1_err = NP.std(e1_obs[mask_Cat_1])
    e2_obs_1_err = NP.std(e2_obs[mask_Cat_1])
    
    e1_obs_2_err = NP.std(e1_obs[mask_Cat_2])
    e2_obs_2_err = NP.std(e2_obs[mask_Cat_2])
    
    if verbose == True:
        print 'Checking E1 and E2 uncertainties'
        if e1_obs_1_err == 0:
            print 'E1_1 error is zero'
            return
        elif e2_obs_1_err == 0:
            print 'E2_1 error is zero'
            return
        if e1_obs_2_err == 0:
            print 'E1_2 error is zero'
            return
        elif e2_obs_2_err == 0:
            print 'E2_2 error is zero'
            return
        else:
            print 'E1 and E2 stdevs are non zero'
            print e1_obs_1_err, e2_obs_1_err, e1_obs_2_err, e2_obs_2_err
    
    #setup output hdf5 file
    filename = output_prefix+'_result.hdf5'
    if verbose == True:
        print 'Setting up output file %s' %(filename)
    f = h5py.File(filename, 'w')
    
    # Save the user inputs.
    grp_input = f.create_group('/userinputs')
    grp_input.attrs['N_halos'] = N_halos
    grp_input.attrs['n_walkers'] = n_walkers
    grp_input.attrs['n_steps'] = n_steps
    grp_input.attrs['n_burn'] = n_burn
    grp_input['bounds_halo'] = bounds_halo
    grp_input.attrs['n_threads'] = n_threads
    
    # Save the raw data.
    grp_input['/data/ra_deg'] = ra_gal
    grp_input['/data/ra_deg'].attrs['units'] = 'degrees'
    grp_input['/data/dec_deg'] = dec_gal
    grp_input['/data/dec_deg'].attrs['units'] = 'degrees'
    grp_input['/data/z'] = z_gal
    grp_input['/data/z'].attrs['units'] = 'redshift'
    grp_input['/data/E1'] = e1_obs
    grp_input['/data/E2'] = e2_obs
    grp_input['/data/DE'] = DE
    grp_input['/data/Catalog'] = Galaxy_input[:,6]
    
    # Concatinate the data structure.
    ra_gal = NP.reshape(ra_gal, (len(ra_gal),1))
    dec_gal = NP.reshape(dec_gal, (len(dec_gal),1))
    z_gal = NP.reshape(z_gal, (len(z_gal),))
    e1_obs = NP.reshape(e1_obs, (len(e1_obs),1))
    e2_obs = NP.reshape(e2_obs, (len(e2_obs),1))
    DE = NP.reshape(DE, (len(DE),1))
    
    # Number of dimensions (i.e. number of model parameters).
    # For each halo component there are 3 parameters (ra, dec, M200)
    n_dim = 3*N_halos
    if verbose == True:
        print 'Setting initial parameters for %i walkers' %(n_walkers)
    #find the mean and size of each prior boundary
    for i in range(N_halos):
        bounds_mid = []
        bounds_size = []
        bounds_mid.append(NP.mean(bounds_halo[i][0]))
        bounds_mid.append(NP.mean(bounds_halo[i][1]))
        bounds_mid.append(NP.mean(bounds_halo[i][2]))
    
        bounds_size.append(NP.max(bounds_halo[i][0])-NP.min(bounds_halo[i][0]))
        bounds_size.append(NP.max(bounds_halo[i][1])-NP.min(bounds_halo[i][1]))
        bounds_size.append(NP.max(bounds_halo[i][2])-NP.min(bounds_halo[i][2]))
    
    # Then estimate the initial starting locations of the halos as the mean of the prior range
    # Set the inital starting locations for each walker by choosing a random value inside the prior bounds
    p_0 = NP.random.uniform(-0.5, 0.5, (n_walkers,3))
    p_0[:,0] = p_0[:,0]*bounds_size[0] + bounds_mid[0]
    p_0[:,1] = p_0[:,1]*bounds_size[1] + bounds_mid[1]
    p_0[:,2] = p_0[:,2]*bounds_size[2] + bounds_mid[2]

    
    H0 = h_scale*100
    DF = WLT.DistanceFraction(H0, Om, z_gal, z_halo)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, WLT.lnprob_2Cat, args=[ra_gal, dec_gal, z_gal, e1_obs, e2_obs, mask_Cat_1, mask_Cat_2,
                                    e1_obs_1_err, e2_obs_1_err, e1_obs_2_err, e2_obs_2_err, DE, DF, N_halos, z_halo, bounds_halo], threads=n_threads)
    
    if verbose == True:
        print 'Running the MCMC chains'
        
    sampler.run_mcmc(p_0, n_steps)
    
    if verbose == True:
        print 'Finished running MCMC chains'
    
        # Save the time series plots for each component and parameter.
    if verbose == True:
        print 'creating time series figures'
    WLT.timeseries(sampler, N_halos, n_burn, output_prefix)

    # Compute the parameter medians and 68% quantiles.
    param_median = []
    param_1sigma = []
    for i in range(n_dim):
        param_median.append(NP.median(sampler.chain[:,n_burn:,i]))
        param_1sigma.append(NP.percentile(sampler.chain[:,n_burn:,i],
                            [16, 84]))
    
    # Locate most likely parameters
    mask = sampler.lnprobability[:,n_burn:] == NP.amax(sampler.lnprobability[:,n_burn:])
    param_maxlike = []
    for i in range(n_dim):
        param_maxlike.append(sampler.chain[:,n_burn:,i][mask][0])
    
    runtime = time.time() - start_time
    if verbose == True:
        print 'runtime was %f seconds' %(runtime)
        
    # Save the MCMC results to the hdf5 file.
    if verbose == True:
        print 'Saving results to hdf5 file'
    #Log Likelihood values
    f['/results/log_likelihoods'] = sampler.lnprobability
    # Parameter estimates.
    # Parameter labels
    param_label = ['RA', 'Dec', 'M200']
    param_units = ['degrees', 'degrees', '10^14 Solar Mass']
    
    grp_res = f.create_group('/results/parameters')
    grp_res.attrs['runtime'] = runtime
    
    for i in range(N_halos):
        # Create a group for each halo
        group_name = 'Halo{0}'.format(i)
        grp_comp = grp_res.create_group(group_name)
        for j in range(3):
            group_name = param_label[j]
            grp_param = grp_comp.create_group(group_name)
            grp_param.attrs['units'] = param_units[j]
            grp_param['median'] = param_median[i*3 + j]
            grp_param['max_likelihood'] = param_maxlike[i*3 + j]
            grp_param['quantiles'] = param_1sigma[i*3 +j]
            grp_param['quantiles'].attrs['quantile'] = '(16%, 84%)'
            grp_param['chain'] = sampler.chain[:, :, i*3 + j]
            grp_param['chain'].attrs['dimensions'] = '(n_step,n_walkers)'
    f.close()
    if verbose == True:
        print 'Finished MCMC analysis and results output'