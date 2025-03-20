
# Importing some important python modules
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii
import os 
import scipy
from PyAstronomy import pyasl
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
from copy import deepcopy
import h5py
import healpy as hp



def ERGS_correction(catalog, cz, radius = 1, num_out = 40):
    '''
    @author: Craig Anderson
    Code to correct for the GRM using the exclusion radius GRM subraction (ERGS), see Anderson+2024
    Parameters
    ----------
    catalog_name: ascii astroy table 
        Catalog of RMs
    cz: int or float
        The redshift to the object of interest
    radius: int or float
        The size of the exclusion radius in Mpc. 1 Mpc by default
    num_out: int
        The number of sources outside the exclusion zone to be used to calculate the GRM
    '''
    file = 'data_files/faraday2020v2.fits'
    hdu  = fits.open(file)
    #f = h5py.File('/home/chiara/Desktop/Hamburg_work/faraday2020v2.hdf5', 'r')
    rm_map = hp.read_map(hdu[1],field=0,dtype=float,nest=False)
    num_out = 40
        
    nside=hp.get_nside(rm_map)

    #define the function to find the pixel position in healpix fromat
    def pix_index(l,b,nside):

        phi = np.deg2rad(l)
        theta = 0.5 * np.pi - np.deg2rad(b)
        ipix = hp.ang2pix(nside, theta, phi)

        return ipix

    #main table
    master_RM_table = deepcopy(catalog)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    scale = cosmo.kpc_proper_per_arcmin(cz).to(u.Mpc/u.arcmin)
    # Check changing exclusion radius by masking the filament's radius 
    rm_correction_source_exclusion_radius_degs = ((radius*u.Mpc)/(scale)).to(u.deg).value
    num_RMs_to_average_for_foreground_correction = num_out

    ## Init columns for corrected RMs to go in to epsilon, for easy recognition of failed correction values
    master_RM_table['rm_corrected'] = np.abs(master_RM_table['rm'])*0+9e9 #Init to 9e9 so it is very noticeable if the correction fails for whatever reason
    master_RM_table['rm_correction'] = np.abs(master_RM_table['rm'])*0+9e9 #Init to 9e9 so it is very noticeable if the correction fails for whatever reason
    master_RM_table['rm_hut'] = np.abs(master_RM_table['rm'])*0+9e9 #Init to 9e9 so it is very noticeable if the correction fails for whatever reason

    #Parse out info for RMs, uncerts, positions, etc from main table
    ra_askap_rms, dec_askap_rms = master_RM_table['ra'], master_RM_table['dec']
    askap_rms_uncorrected = master_RM_table['rm']
    askap_rm_errs = master_RM_table['rm_err']

    #X-match the RMs with themselves to find k nearest neighbours for weighted averaging
    minmax_seps_recorder = [] #Init a column that will report the min and max separations of RMs in annular region use for correction of each RM
    askap_rms_cat_coords = SkyCoord(ra=ra_askap_rms*u.degree, dec=dec_askap_rms*u.degree)

    #Loop through each RM source, and correct for Galactic foreground
    for row_idx,rmcoord in enumerate(askap_rms_cat_coords):

        #print every 100th source, so we know things are working
        if row_idx%100==0:
            print('Correcting Galactic foreground for source #%d'%row_idx)	

        #Calc sep of the RM we want to correct from all other RMs in the main table
        seps = rmcoord.separation(askap_rms_cat_coords)

        #Exclude RMs closer than a specified radius to the RM we want to correct --- don't want to subtract out signal we're interested in, which could be spatially correlated!
        good_seps_boolean = seps.value > rm_correction_source_exclusion_radius_degs 

        #Get list of seps, RMs, RMerr, and positions for RMs we want to consider
        good_seps = seps[good_seps_boolean]
        useful_RMs = askap_rms_uncorrected[good_seps_boolean]
        useful_RM_errs = askap_rm_errs[good_seps_boolean]
        useful_rmcoords = askap_rms_cat_coords[good_seps_boolean]

        #Sort the seps, and get the k-nearest neighbour indexes --- i.e. the RMs that are closest to the present source bar the exclusion zone
        knnidxs = good_seps.argsort()[:num_RMs_to_average_for_foreground_correction]

        #Get the RMs, errs we'll estimate the Gal foreground from
        nearby_rms_uncorrected = useful_RMs[knnidxs]
        nearby_rm_errs = useful_RM_errs[knnidxs]
        nearby_rm_seps = good_seps[knnidxs].value
        nearby_rmcoords = useful_rmcoords[knnidxs]

        #Record what the min/max seps from the RMs to the cluster centres are
        minmax_seps_recorder.append([np.nanmin(nearby_rm_seps),np.nanmax(nearby_rm_seps)])

        #Calculate median of RMs --- this is our estimate for the Galactic foreground
        median_RM = np.nanmedian(nearby_rms_uncorrected) #Equal weighting

        #Implement correction
        master_RM_table[row_idx]['rm_corrected'] = master_RM_table[row_idx]['rm'] - median_RM
        master_RM_table[row_idx]['rm_correction'] = median_RM

        
        #read GRM from Huts and save it
        l=rmcoord.galactic.l.degree
        b=rmcoord.galactic.b.degree

        master_RM_table[row_idx]['rm_hut'] = rm_map[pix_index(l,b,nside)]

    return master_RM_table


def H22GRM_correction(results, model='ExtDepol', which='huts'):
    """
    Remove the Galactic contribution to Faraday rotation.
    
    results -- astropy.Table object (from astropy.table import Table; results = Table.read(fitstable.fits) )

    which -- 'huts' -> use the map from Hutschenreuter2022
          -- 'opperman' -> use the map from Opperman+2012  https://arxiv.org/abs/1111.6186
    """
    RM = results['rm']
    RMErrPos = results['rm_err']
    #RMErrNeg = results['RM_ErrNeg_%s'%model]
        
    RA = results['ra']
    DEC = results['dec']
    sc = SkyCoord(RA,DEC,unit=(u.deg,u.deg),frame='icrs')
    
    nest = False # usng RING ordering, not NESTED pixel ordering.
    
    if which == 'opperman':
        # Parameters from Opperman+2012
        nside = 128  # 128 pixels per side
        # Location of the map, stored in HEALPix format. 
        rmmap = 'faradaymap_Opperman2012.fits'

    elif which == 'huts':
        nside = 512
        rmmap = 'data_files/faraday2020v2.fits'
    else:
        raise ValueError(f"Map {which} not implemented. Try 'huts' or 'opperman'")

    hdu = fits.open(rmmap)
    if which == 'opperman':
        # Array with value of every pixel. Faraday depth is 3rd column
        rmmap = hp.read_map(hdu[3],field=0,dtype=float,nest=nest)
        # Faraday depth uncertainty is fourth column [rad m^-2]
        rmunc = hp.read_map(hdu[4],field=0,dtype=float,nest=nest)
    elif which == 'huts':
        # Array with value of every pixel. Faraday depth 
        rmmap = hp.read_map(hdu[1],field=0,dtype=float,nest=nest)
        # Faraday depth scatter uncertainty
        rmunc = hp.read_map(hdu[1],field=1,dtype=float,nest=nest)    

    # Need galactic longitude and latitude.
    sc_galactic = sc.icrs.galactic
    
    # Find for every coordinate, the correction value
    RMcorrection = np.zeros(len(results))
    RMcorrection_err = np.zeros(len(results))
    for i, coord in enumerate(sc_galactic):
        l = coord.l.value #lon 
        b = coord.b.value #lat
          
        # get pixel index of galactic coordinates lon,lat. 
        x = hp.ang2pix(nside,l,b,nest,lonlat=True)
        RMcorrection[i]     = rmmap[x]
        RMcorrection_err[i] = rmunc[x] 
    
    results['RM_corrected'] = RM - RMcorrection
    # Add error in quadrature. good assumption if error is indep. Gaussian
    results['RM_ErrPos_corrected'] = np.sqrt(RMErrPos**2 + RMcorrection_err**2)
    #results['RM_ErrNeg_%s_corrected'%model] = np.sqrt(RMErrNeg**2 + RMcorrection_err**2)
    
    return results