import numpy as NP
import cosmo
from astropy.cosmology import FlatLambdaCDM
from profiles import nfw_Sigma
from profiles import nfw_Sigmabar
from profiles import nfwparam
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as PLT
from matplotlib.ticker import MaxNLocator


degrad = NP.pi/180.0
G = 6.673*10**-11 # m**3/kg/s**2
c = 299792458 #Units m/s
minMpc = 3.08568025*10**22 # km in a Megaparsec
kminMpc = 3.08568025*10**19 # km in a Megaparsec

def DistanceFraction(H0, Om, z_gals, z_clust):
    Cos = FlatLambdaCDM(H0=H0, Om0=Om)
    Ds = Cos.angular_diameter_distance(z_gals)
    Dl = Cos.angular_diameter_distance(z_clust)
    
    #Calculate Angular Diameter Distance between objects and Cluster
    DM1 = Cos.comoving_distance(z_clust)
    DM2 = Cos.comoving_distance(z_gals)
    Dls = (DM2 - DM1)/(1 + z_gals)
    return NP.array(Ds/(Dls*Dl))

def ShearCalc(ra_gal, dec_gal, z_gals, z_halo, M200, ra_hal, dec_hal, DF, masking=False):
    '''
    cat = catalog array of data including positions, elipticity components, and elipticity errors
    halos = array of including the ra and dec of halos to be fit
    N_halo = number of halos to be fit in the model
    z_halo = redshift of galaxy cluster
    M_200 = mass of halo(s)
    DF=array of distance fractions Dls*Dl/Ds for each galaxy
    
    del_c = characteristic overdensity of the CDM halo
    r_s = scale radius of the halo (Mpc)
    r = radius of interest (Mpc)
    z_halo = halo redshift
    h_scale = hubble scale H = h*100 km/s/Mpc
    Om = matter energy density
    Ol = dark energy density
    Or = radiation energy density
    '''

    N_gal = NP.shape(ra_gal)[0]
    N_halo = NP.shape(ra_hal)[0]
    h_scale=0.7
    Om=0.3
    Ol=0.7
    Or=0.0
    del_a = NP.zeros((N_gal,N_halo))
    del_d = NP.zeros((N_gal,N_halo))
    Sigma = NP.zeros((N_gal,N_halo))
    gamma = NP.zeros((N_gal,N_halo))
    del_c = NP.zeros((N_halo))
    r_s = NP.zeros((N_halo))
    
    ra_hal = NP.array(ra_hal)*degrad
    dec_hal = NP.array(dec_hal)*degrad
    
    ra_gal = NP.reshape(NP.array(ra_gal)*degrad, (len(ra_gal),))
    dec_gal = NP.reshape(NP.array(dec_gal)*degrad, (len(dec_gal),))
    
    #Calculate distance between halo centers and each point in the catalog and inverse sigma critical
    for h in NP.arange(N_halo):
        ra_hals = ra_gal*0 + ra_hal[h]
        AD = 60*cosmo.ProjectedLength(z_halo,h_scale,Om,Ol)
        div = NP.sin(dec_hal[h])*NP.sin(dec_gal)+NP.cos(dec_hal[h])*NP.cos(dec_gal)*NP.cos(NP.abs(ra_gal-ra_hals[h]))
        del_a[:,h] = AD*NP.cos(dec_hal[h])*NP.sin(NP.abs(ra_gal-ra_hals[h]))/(div*degrad)
        del_d[:,h] = -AD*(NP.sin(dec_hal[h])*NP.cos(dec_gal)-NP.cos(dec_hal[h])*NP.sin(dec_gal)*NP.cos(NP.abs(ra_gal-ra_hals[h]))/div)/degrad
                   
    del_r = NP.sqrt(del_a**2+del_d**2)
    
    #Calculate critical density and scale radius of each halo
    for h in NP.arange(N_halo):
        del_c[h], r_s[h] = nfwparam(M200[h],z_halo,h_scale=0.7,Om=0.3,Ol=0.7,Or=0.0)

    #calculate orientation angle of each galaxy w.r.t. the halos
    del_ra = NP.absolute(del_a)
    del_dec = NP.absolute(del_d)
    phi = NP.arctan2(del_dec, del_ra)
    
    for h in NP.arange(N_halo):
        mask_1 = ra_gal >= ra_hal[h]
        mask_2 = ra_gal < ra_hal[h]
        mask_3 = dec_gal >= dec_hal[h]
        mask_4 = dec_gal < dec_hal[h]
        
        mask_p1 = mask_1*mask_3
        phi[mask_p1,h] = NP.pi - phi[mask_p1,h]
        mask_p2 = mask_1*mask_4
        phi[mask_p2,h] += NP.pi
        mask_p3 = mask_2*mask_4
        phi[mask_p3,h] = -phi[mask_p3,h]

    #Calculate absolute shear for each halo
    Sigmacr = (DF/minMpc)*(c**2)/(4*NP.pi*G)
        
    for h in NP.arange(N_halo):
        #Calculate surface density of each halo
        #loop through each halo calculating the expected absolute shears
        Sigma[:,h] = nfw_Sigma(del_c[h],r_s[h],del_r[:,h],z_halo,h_scale,Om,Ol,Or)
        gamma[:,h] = ((nfw_Sigmabar(del_c[h],r_s[h],del_r[:,h],z_halo,h_scale,Om,Ol,Or)-Sigma[:,h]))/Sigmacr/(1-Sigma[:,h]/Sigmacr)


    #mask objects too close to the lens
    zeros = NP.zeros((N_gal,))
    emask = NP.ma.make_mask(zeros, shrink=False)
    emask = ~emask

    dist = del_r/AD
    for h in NP.arange(N_halo):
        mask = dist[:,h] >= 0.01
        emask *= mask
            
    if masking == True:
        gamma_m = gamma[emask,:]
        phi_m = phi[emask,:]
        N_gal_m = NP.shape(gamma_m)[0]
    
    #calculate the expected ellipticities of each galaxy due to all halos
        cos2phi = NP.cos(2*phi_m)
        sin2phi = NP.sin(2*phi_m)
        e1_exp = NP.zeros((N_gal_m,))
        e2_exp = NP.zeros((N_gal_m,))
        for h in NP.arange(N_halo):
            for g in NP.arange(N_gal_m):
                e1_exp[g,] -= gamma_m[g,h]*cos2phi[g,h]
                e2_exp[g,] -= gamma_m[g,h]*sin2phi[g,h]
            
        return e1_exp, e2_exp, emask
    else:
        cos2phi = NP.cos(2*phi)
        sin2phi = NP.sin(2*phi)
        e1_exp = NP.zeros((N_gal,))
        e2_exp = NP.zeros((N_gal,))
        for h in NP.arange(N_halo):
            for g in NP.arange(N_gal):
                e1_exp[g,] -= gamma[g,h]*cos2phi[g,h]
                e2_exp[g,] -= gamma[g,h]*sin2phi[g,h]
                
        return e1_exp, e2_exp, emask

def prior_check(ra, dec, M200, bounds_halo):
    #Check parameters against the uniform prior bounds
    if not all((bounds_halo[0][0] < ra < bounds_halo[0][1],
                bounds_halo[1][0] < dec < bounds_halo[1][1],
                bounds_halo[2][0] < M200 < bounds_halo[2][1])):
        return False


#Define residual function
def lnprob(theta, ra_gal, dec_gal, z_gal, e1_obs, e2_obs, e1_obs_err, e2_obs_err, DE, DF, N_halos, z_halo, bounds_halo):
    #unpack model parameters from the theta vector
    M200 = []
    ra_hal = []
    dec_hal = []
    for i in range(N_halos):
        ra_hal.append(theta[i*3])
        dec_hal.append(theta[i*3+1])
        M200.append(theta[i*3+2])

    for h in range(N_halos):
        #Check that parameters fall inside the set bounds
        #If they do not, return log likelihood of -infinity
        pc = prior_check(ra_hal[h], dec_hal[h], M200[h], bounds_halo[h])
        if pc == False:
            return -NP.inf, None
    
    #Calcualte expected shear components given position and mass of the halos
    e1_exp, e2_exp, emask = ShearCalc(ra_gal, dec_gal, z_gal, z_halo, M200, ra_hal, dec_hal, DF, masking=True)
    #Calculate residuals on shapes and convert to log likelihood
    e1_obs_m = e1_obs[emask]
    e2_obs_m = e2_obs[emask]
    e1_exp = NP.reshape(e1_exp, (NP.sum(emask),1))
    e2_exp = NP.reshape(e2_exp, (NP.sum(emask),1))
    
    DE_m = DE[emask]
    N_res = 2*len(e1_obs_m)
    res = NP.sum(((e1_obs_m-e1_exp)**2)/(DE_m**2+e1_obs_err**2) + ((e2_obs_m-e2_exp)**2)/(DE_m**2+e2_obs_err**2))
    return -N_res*NP.log(res/(N_res))/2.0, None

def lnprob_Chi2(theta, ra_gal, dec_gal, z_gal, e1_obs, e2_obs, e1_obs_err, e2_obs_err, DE, DF, N_halos, z_halo, bounds_halo):
    #unpack model parameters from the theta vector
    M200 = []
    ra_hal = []
    dec_hal = []
    for i in range(N_halos):
        ra_hal.append(theta[i*3])
        dec_hal.append(theta[i*3+1])
        M200.append(theta[i*3+2])

    for h in range(N_halos):
        #Check that parameters fall inside the set bounds
        #If they do not, return log likelihood of -infinity
        pc = prior_check(ra_hal[h], dec_hal[h], M200[h], bounds_halo[h])
        if pc == False:
            return -NP.inf, None
    
    #Calcualte expected shear components given position and mass of the halos
    e1_exp, e2_exp, emask = ShearCalc(ra_gal, dec_gal, z_gal, z_halo, M200, ra_hal, dec_hal, DF)
    
    #Calculate residuals on shapes and convert to log likelihood
    e1_obs_m = e1_obs[emask]
    e2_obs_m = e2_obs[emask]
    e1_exp = NP.reshape(e1_exp, (NP.sum(emask),1))
    e2_exp = NP.reshape(e2_exp, (NP.sum(emask),1))
    
    DE_m = DE[emask]
    N_res = 2*len(e1_obs_m)
    res = NP.sum(((e1_obs_m-e1_exp)**2)/(DE_m**2+e1_obs_err**2) + ((e2_obs_m-e2_exp)**2)/(DE_m**2+e2_obs_err**2))
    return -(res)/N_res, None

#Define log likelihood function
def lnprob_2Cat(theta, ra_gal, dec_gal, z_gal, e1_obs, e2_obs, mask_Cat_1, mask_Cat_2, e1_obs_1_err, e2_obs_1_err, 
                e1_obs_2_err, e2_obs_2_err, DE, DF, N_halos, z_halo, bounds_halo):
    #unpack model parameters from the theta vector
    M200 = []
    ra_hal = []
    dec_hal = []
    for i in range(N_halos):
        ra_hal.append(theta[i*3])
        dec_hal.append(theta[i*3+1])
        M200.append(theta[i*3+2])

    for h in range(N_halos):
        #Check that parameters fall inside the set bounds
        #If they do not, return log likelihood of -infinity
        pc = prior_check(ra_hal[h], dec_hal[h], M200[h], bounds_halo[h])
        if pc == False:
            return -NP.inf, None
    
    #Calcualte expected shear components given position and mass of the halos
    e1_exp, e2_exp, emask = ShearCalc(ra_gal, dec_gal, z_gal, z_halo, M200, ra_hal, dec_hal, DF, masking=False)
    
    #Separate the two catalogs of data
    emask1 = emask*mask_Cat_1
    emask2 = emask*mask_Cat_2
    
    e1_obs_1 = e1_obs[emask1]
    e2_obs_1 = e2_obs[emask1]
    e1_exp_1 = e1_exp[emask1]
    e2_exp_1 = e2_exp[emask1]
    DE1 = DE[emask1]
    N1 = NP.sum(emask1)
    
    e1_exp_1 = NP.reshape(e1_exp_1, (N1,1))
    e2_exp_1 = NP.reshape(e2_exp_1, (N1,1))
    
    e1_obs_2 = e1_obs[emask2]
    e2_obs_2 = e2_obs[emask2]
    e1_exp_2 = e1_exp[emask2]
    e2_exp_2 = e2_exp[emask2]
    DE2 = DE[emask2]
    N2 = NP.sum(emask2)
    
    e1_exp_2 = NP.reshape(e1_exp_2, (N2,1))
    e2_exp_2 = NP.reshape(e2_exp_2, (N2,1))
       
    #Calculate residuals on shapes and convert to log likelihood
    res1 = NP.sum(((e1_obs_1-e1_exp_1)**2)/(DE1**2+e1_obs_1_err**2) + ((e2_obs_1-e2_exp_1)**2)/(DE1**2+e2_obs_1_err**2))
    res2 = NP.sum(((e1_obs_2-e1_exp_2)**2)/(DE2**2+e1_obs_2_err**2) + ((e2_obs_2-e2_exp_2)**2)/(DE2**2+e2_obs_2_err**2))
        
    return -2.0*N1*NP.log(res1/(2.0*N1))/2.0 - 2.0*N2*NP.log(res2/(2.0*N2))/2.0, None
    
def timeseries(sampler, n_halo, n_burn, output_prefix):
    '''
    Plot the time series of all the walker chain steps.
    sampler is the emcee.EnsembleSampler() object and contains some number of
    chains after running sampler.run_mcmc.
    n_halo = [int] number of mass halos being fit
    n_burn = [int] the number of initial steps that are to be burned.
    output_prefix = the file name prefix to append to the output figures.
    '''
    # Parameter labels
    param_label_halo = ['$\mathrm{RA}$', '$\mathrm{Dec}$', '$M200$',]
    
    # Plot the time series for each subcluster component.
    for i in range(n_halo):
        fig, ax = PLT.subplots(3, 1, sharex=True, figsize=(8, 12))
        ax[0].set_title('Halo {0} MCMC Chains'.format(i))
        for j in range(3):
            ax[j].plot(sampler.chain[:, :, j + i*3].T, color='k', alpha=0.1)
            y_limits = ax[j].get_ylim()
            ax[j].set_ylim(y_limits)
            ax[j].fill_betweenx(ax[j].get_ylim(), x1=n_burn,color='r',alpha=0.5)
            ax[j].yaxis.set_major_locator(MaxNLocator(5))
            ax[j].set_ylabel('{0}'.format(param_label_halo[j]))
            # Reset the y_limits that may have been altered by the fill_betweenx
            # operation.
            ax[j].set_ylim(y_limits)
        # Add x-axis label to the last row.
        ax[2].set_xlabel('step number')
        # Save the figure
        filename = output_prefix+'_Halo{0}.png'.format(i)
        fig.savefig(filename)

    
    
    
    
    
    
    
    
    
    
    
    
    
    