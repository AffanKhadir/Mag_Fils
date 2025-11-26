# Importing some important python modules
import numpy as np
import matplotlib.pyplot as plt
import running_bins
import cv2
import csv
from astropy.io import ascii
import scipy
import os 
import math
import urllib.request
from PyAstronomy import pyasl
from scipy.stats import norm
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.constants import c as lightspeed
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.io import fits
from copy import deepcopy
import h5py
import healpy as hp
import pandas as pd
from multiprocessing import Pool
from dustmaps.edenhofer2023 import Edenhofer2023Query
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.visualization import ZScaleInterval
import ast
from scipy.stats import binned_statistic
from glob import glob 
import dustmaps.edenhofer2023
from dustmaps.edenhofer2023 import Edenhofer2023Query
from dustmaps.config import config
from reproject import reproject_from_healpix
import aplpy
from astropy.table import Table
from photoz_codes.download_panstarrs_magnitudes import querycatalogue
from photoz_codes.tario_et_al_pythonscripts.prepare_test_galaxies_tarrio import prepare_test_galaxies_tarrio
from photoz_codes.tario_et_al_pythonscripts.linear_regression_zphot_tarrio import linear_regression_zphot_tarrio
from photoz_codes.tario_et_al_pythonscripts.compute_photo_z_pan_tarrio import compute_photo_z_pan_tarrio
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from sympy import *
from brokenaxes import brokenaxes
from numpy import logspace
from uncertainties import unumpy
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import json
from scipy.ndimage import gaussian_filter1d
import emcee
import corner



plt.rcParams.update({'font.size': 16})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Courier'


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
ra_cen, dec_cen = 211.8742, -27.0178

cen = SkyCoord(ra_cen, dec_cen, frame = 'icrs', unit = (u.deg, u.deg))  
scale = cosmo.kpc_proper_per_arcmin(0.0221).to(u.Mpc/u.arcmin)

radio_data = fits.open('image.i.EMU_1412-28.SB50413.cont.taylor.0.highres.restored.conv.fits')
rim = radio_data[0].data[0][0]
wcs = WCS(radio_data[0].header).celestial

def CGM_plotting(catalog_name, in_clust = False):
    '''
    Function to produce the plots of the RMs versus the CGM 

    Parameters
    ---------
    catalog_name: str
        The location of the RM catalogue 
    in_clust: bool
        Flag to include sources that have been identified to be inside the cluster. False by default. 
    
    Returns
    ------
    None
    '''
    
    data = Table.read(catalog_name)

    

    if in_clust == False:
        in_clust_array = data['in_clust']
        data = data[np.argwhere(in_clust_array == 0)]
    
    rrm = np.ravel(data['rm_corrected'].data)
    ra = data['ra']
    dec = data['dec']
    rrm_err = np.ravel(data['rm_err_corrected'].data)

    coords = SkyCoord(ra, dec, unit = 'deg')
    r = cen.separation(coords).to(u.arcmin) * scale

        #dist = 91.465 

    
    panstarrs_gal = Table.read('panstarrs_gals.fits')

    # Calculating the distance to the cluster centre 

    ra_gal = panstarrs_gal['RAJ2000']
    dec_gal = panstarrs_gal['DEJ2000']
    sources_gal = SkyCoord(ra_gal.data, dec_gal.data, frame ='icrs',  unit = (u.deg, u.deg))
    

    separation_gal = cen.separation(sources_gal)
    radius_gal  = separation_gal.to(u.arcmin) * scale
    panstarrs_gal = panstarrs_gal[radius_gal.value < 2.2 * 0.925]
    ra_gal = panstarrs_gal['RAJ2000']
    dec_gal = panstarrs_gal['DEJ2000']
    sources_gal = SkyCoord(ra_gal.data, dec_gal.data, frame ='icrs',  unit = (u.deg, u.deg))
    

    separation_gal = cen.separation(sources_gal)
    radius_gal  = separation_gal.to(u.arcmin) * scale


    sources_gal = SkyCoord(ra_gal.data, dec_gal.data, frame ='icrs',  unit = (u.deg, u.deg))



    # Removing sources that are outisde of 2R_500 of the cluster

    i_gal = panstarrs_gal['imag']
    i_gal_err = panstarrs_gal['e_imag']
    g_gal = panstarrs_gal['gmag']
    g_gal_err = panstarrs_gal['e_gmag']

    ra_gal = panstarrs_gal['RAJ2000']
    dec_gal = panstarrs_gal['DEJ2000']
    z_gal_data = Table.read('pan_photz_gals_corr.fits')




    z_gal = z_gal_data['z_phot']
    z_gal_err = z_gal_data['z_phot_err']



    vel_clust = 0.0221 * lightspeed.to(u.km/u.s)
    vel_gal = z_gal * lightspeed.to(u.km/u.s)
    vel_gal_err = z_gal_err * lightspeed.to(u.km/u.s)

    del_gal_indices = []

    # We will consider a gixed gap of 1000 km/s
    for i in range(len(z_gal)):
        if z_gal[i] > 0:
            interval1 = pd.Interval(vel_clust.value - 1000, vel_clust.value + 1000)
            interval2 = pd.Interval(vel_gal[i].value - vel_gal_err[i].value, vel_gal[i].value + vel_gal_err[i].value)

            if interval1.overlaps(interval2) == False or z_gal_err[i] > 0.4 * z_gal[i]: 
                del_gal_indices.append(i)
        else:
            del_gal_indices.append(i)
    z_gal_clust = np.delete(z_gal, del_gal_indices)
    z_gal_err_clust = np.delete(z_gal_err, del_gal_indices)
    ra_gal_clust = np.delete(ra_gal, del_gal_indices)
    dec_gal_clust = np.delete(dec_gal, del_gal_indices)
    i_gal_clust  = np.delete(i_gal, del_gal_indices)
    i_gal__err_clust  = np.delete(i_gal_err, del_gal_indices)
    g_gal_clust  = np.delete(g_gal, del_gal_indices)
    g_gal__err_clust  = np.delete(g_gal_err, del_gal_indices)
    radius_gal_clust = np.delete(radius_gal, del_gal_indices)



    phot_gal_clust = SkyCoord(ra_gal_clust[radius_gal_clust.value < 2.2 * 0.925], dec_gal_clust[radius_gal_clust.value < 2.2 * 0.925], unit = 'deg')

    # Here, we will alculate the distnace to the closest galaxy and plot RM as a function of this distance




    phot_b_impact_clust = []
    plt.figure(facecolor = 'white', figsize = (5.84685039, 4.46338583))
    for i in range(len(coords)):
        b_impact = coords[i].separation(phot_gal_clust).to(u.arcmin) * scale
        b_impact_closest = np.min(b_impact.value)
        b_impact_mean = np.mean(np.sort(b_impact.value)[:5])
        phot_b_impact_clust.append(b_impact_closest * 1000)
        #plt.errorbar(b_impact_closest, np.abs(rm_clust)[i], yerr = rm_err_clust[i], c = 'black', fmt = '.')
    plt.errorbar(phot_b_impact_clust, rrm, rrm_err, fmt = '.', capsize = 5)
    plt.xlabel('$b_\mathrm{nearest}$ (kpc)')
    plt.ylabel('RRM (rad m$^{-2}$)')
    plt.savefig('./GOOD_Figures/phot_impact_rm.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()

    


    fig, ax = plt.subplots(ncols=1, subplot_kw={'projection': wcs}, figsize = (9, 5))
    fig.patch.set_facecolor('white')
    colour = ax.scatter(ra, dec, c  = phot_b_impact_clust, cmap = 'plasma', transform = ax.get_transform('icrs'))
    colorbar = plt.colorbar(colour, ax=ax)
    colorbar.set_label('$b_{\mathrm{nearest}}$ (kpc)')
    ax.scatter(ra_gal_clust[radius_gal_clust.value < 2 * 0.925], dec_gal_clust[radius_gal_clust.value < 2 * 0.925], marker = 'x', s = 3, transform = ax.get_transform('icrs'))
    ax.set_aspect(1)
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('DEC (J2000)')
    plt.show()
    plt.close()


    data_in_pol_frac = Table()

    #data_in_pol_frac['r_sorted'] = r_final[r_final < 0.925 * 3]
    #data_in_pol_frac['RM'] =  master_RM_table['rm_corrected'][r_final < 0.925 * 3]
    #data_in_pol_frac['RM_err'] =  master_RM_table['rm_err'][r_final < 0.925 * 3]

    data_in_pol_frac['r_sorted'] = np.array(phot_b_impact_clust)
    data_in_pol_frac['RM'] = np.array(rrm)
    data_in_pol_frac['RM_err'] = np.array(rrm_err)


    obsresults = data_in_pol_frac

    iqr  = 6.99

    # Do the calculation
    b_running_in, b_scatter_in, b_scatter_errlow_in, b_scatter_errup_in = running_bins.calc_running_scatter(obsresults
                                                            , RMcol='RM' # column containing RRM
                                                            , xcol='r_sorted' # column containing distance to cluster
                                                            , RMerrcol = 'RM_err' # column containing RRM error
                                                            # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                            , xwidth=None # if we want set number of points
                                                            , M =  15      # number of points in sliding window
                                                            , method='iqr' # IQR is more robust than STD
                                                            , show=True
                                                            , sigmaRMextr = 6.99 # change this value to what scatter is far inside clusters
                                                                            # corrected for measurement errors!
                                                            )
    

    data_in_pol_frac = Table()


    data_in_pol_frac['r_sorted'] = np.ravel(r.data)
    data_in_pol_frac['RM'] = rrm
    data_in_pol_frac['RM_err'] = rrm_err


    obsresults = data_in_pol_frac

    # Do the calculation
    x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in = running_bins.calc_running_scatter(obsresults
                                                            , RMcol='RM' # column containing RRM
                                                            , xcol='r_sorted' # column containing distance to cluster
                                                            , RMerrcol = 'RM_err' # column containing RRM error
                                                            # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                            , xwidth=None # if we want set number of points
                                                            , M =  20      # number of points in sliding window
                                                            , method='iqr' # IQR is more robust than STD
                                                            , show=True
                                                            , sigmaRMextr = 6.99 # change this value to what scatter is far inside clusters
                                                                            # corrected for measurement errors!
                                                            )
    
    def find_nearest(array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx
    '''
    # Binning the data in bins of 5 kpc of the mean impact parameter 
    bin_linspace = np.linspace(5, 55, 11)


    # Bin the data using digitize
    bin_indices = np.digitize(np.array(phot_b_impact_clust) * 1000, bin_linspace)

    bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    '''

    phot_b_new  = np.array(phot_b_impact_clust)[np.argsort(np.array(phot_b_impact_clust))] 
    rrm_clust_new = rrm[np.argsort(np.array(phot_b_impact_clust))]
    r_clust_new = r.value[np.argsort(np.array(phot_b_impact_clust))]

    bin_linspace = np.linspace(0, 600, 6)

    #bin_indices = np.digitize(np.array(phot_b_impact_clust) * 1000, bin_linspace)
    bin_indices = np.digitize(np.array(phot_b_new), bin_linspace )
    bins = [1, 2, 3, 4, 5, 6]
    CGM_scat = []
    CGM_low_err = []
    CGM_up_err = []

    median_x, CGM_scat, CGM_low_err, CGM_up_err = [], [], [], []
    # The final CGM scatter plot! Yay!
    M = 20 # Number of points per bin
    for i in range(len(phot_b_new)):
        if i + M <= len(phot_b_new):
            # Fixed number of points, as long as we have enough points
            left_bound = phot_b_new[i]
            right_bound = phot_b_new[i+M-1]

        # Find indices of elements within the window
        window_indices = np.where((phot_b_new >= left_bound) & (phot_b_new <= right_bound))[0]
        # Extract the values within the window
        if len(window_indices) > 0 :
            curr_impact = phot_b_new[window_indices]
            impact_indices = abs(curr_impact[:, None] - b_running_in[None, :]).argmin(axis=-1)
            curr_rrm = rrm_clust_new[window_indices]
            curr_radius = r_clust_new[window_indices]
            curr_err_low = np.mean(b_scatter_errlow_in[impact_indices])
            curr_err_up = np.mean(b_scatter_errup_in[impact_indices])
            curr_idx = []
            for j in range(len(curr_radius)):
                curr_idx.append(find_nearest(x_running_in, curr_radius[j]))
                # Interpolation is not the best way to do this as the interpolation introduces a lot of uncharacteristic behaviour (interpolation produces a smooth function but the 
                # scatter plot is not smooth)
            curr_x_in = x_running_in[curr_idx]
            curr_scatter_in = (1+0.0221)**2 * scatter_in[curr_idx]

            curr_obs_scatter = np.subtract(*np.percentile(curr_rrm, [75, 25]))/1.349
            curr_scatter_errup_in = scatter_errup_in[curr_idx]
            curr_scatter_errlow_in = scatter_errlow_in[curr_idx]
            curr_scatter_err_in = np.max(np.vstack((curr_scatter_errup_in, curr_scatter_errup_in)), axis=0)

            CGM_scatter_array = []
            ICM_scatter_std_array = []
            count = 0
            while count < 100:
                ICM_scatter_array = np.random.normal(loc = 0, scale = curr_scatter_in)
                ICM_scatter_std = np.subtract(*np.percentile(ICM_scatter_array, [75, 25]))/1.349
                if curr_obs_scatter**2 - ICM_scatter_std**2 >= 0:
                    count += 1
                    CGM_scatter_mean  = np.sqrt(curr_obs_scatter**2 - ICM_scatter_std**2)
                    ICM_scatter_std_array.append(ICM_scatter_std)
                    CGM_scatter_array.append(CGM_scatter_mean)
                
            CGM_bin = np.median(CGM_scatter_array)
            CGM_scat.append(np.median(CGM_scatter_array))
            ICM = np.median(ICM_scatter_std_array)
            ICM_err = (ICM - np.percentile(ICM_scatter_std_array, 16) + np.percentile(ICM_scatter_std_array, 84) - ICM )/2
            low_err = np.sqrt(ICM_err**2 + curr_err_low**2)
            up_err = np.sqrt(ICM_err**2 + curr_err_up ** 2)
            #low_err = np.sqrt((CGM_bin)**(-1) * curr_obs_scatter**2 * curr_err_low**2 + (CGM_bin)**(-1) * ICM**2 * ICM_err**2)
            #up_err = np.sqrt((CGM_bin)**(-1) * curr_obs_scatter**2 * curr_err_up**2 + (CGM_bin)**(-1) * ICM**2 * ICM_err**2)
            CGM_low_err.append(low_err)
            CGM_up_err.append(up_err)
            
            #CGM_low_err.append( CGM_bin - np.percentile(CGM_scatter_array, 16))
            #CGM_up_err.append( np.percentile(CGM_scatter_array, 84) - CGM_bin )
            
            median_x.append(np.median(curr_impact))

    CGM_scat = np.array(CGM_scat)
    CGM_up_err = np.array(CGM_up_err)
    CGM_low_err = np.array(CGM_low_err)
    plt.figure(figsize = (5.84685039, 4.46338583))
    plt.plot(median_x, CGM_scat)
    plt.fill_between(median_x, CGM_scat + CGM_up_err, CGM_scat - CGM_low_err, alpha = 0.5)
    plt.xlabel('$b_{\mathrm{nearest}}$ (kpc)')
    #plt.xlabel('$\ell_{\mathrm{CGM}}$ (kpc)')
    plt.ylabel('$\sigma_{\mathrm{RRM, CGM}}$ (rad m$^{-2}$)')
    plt.savefig('./GOOD_Figures/CGM_scatter.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def removing_duplicates(catalog, FDFs, spectra, cut_off = 20):
    '''
    Removes duplicate sources that were incorrectly found twice by Selavy by checking if they are within 1 POSSUM telescope beam (20 arcsec) of each other
    Only retains sources with the highest SNR
    Parameters
    -----------
    catalog: astropy ascii table
        catalog of RMs
    FDFs: astropy ascii table
        table of the FDFs for the sources 
    spectra: astropy ascii table 
        table of the Stokes spectra for the sources
    cutoff: int or float
        the telescope beam size used to remove duplicate sources in arcsec. 20 by default 

    Returns
    --------
    catalog: astropy ascii table
        catalog of RMs without duplicates
    FDFs: astropy ascii table
        table of the FDFs for the sources without duplicates
    spectra: astropy ascii table without duplicates
        table of the Stokes spectra for the sources without duplicates
    ''' 
    
    ra_list, dec_list, snr_list = catalog['ra'], catalog['dec'], catalog['SNR_PI']

    i = 0
    deleting = True
    while deleting:
        ra_c = ra_list[i]
        dec_c = dec_list[i]
        snr_c = snr_list[i]

        sep_list = SkyCoord(ra_list, dec_list, unit ='deg').separation(SkyCoord(ra_c, dec_c, unit = 'deg'))

        del_indices = np.bitwise_and((sep_list.arcsecond < cut_off), (snr_list < snr_c))
        catalog.remove_rows(del_indices) # removing the duplicate entry with the smaller SNR
        FDFs.remove_rows(del_indices)
        spectra.remove_rows(del_indices)
        ra_list, dec_list, snr_list = catalog['ra'], catalog['dec'], catalog['SNR_PI'] # updating the list inside the loop
                
        if  i % 100 == 0: 
            print('{:.0f} completed'.format(i))
        i += 1
        
        
        if i == len(ra_list):
            deleting = False
    return catalog, FDFs, spectra

def catalog_cleaning(sbid, show_plots = False):
    '''
    Parameters
    ---------
    sbid : string 
        The SBID of the directory containing all the data products from the single SB pipeline
    show_plots : bool
        Boolean  to display diagnostic plots. False by default 
    Returns
    -------
        None
    '''
    # Loading in the RM catalog, the FDF and the stokes spectra
    data = Table.read(sbid+'/catalog.csv')
    FDFs = Table.read(sbid+'/FDFs.fits')
    spectra = Table.read(sbid + '/spectra.fits')

    pol_frac, snr = data['polint']/data['stokesI'], data['SNR_PI']

    data_cleaned = Table()
    FDFs_cleaned  = Table()
    spectra_cleaned = Table()

    # Removing all sources with a SNR < 10 and polarisation fraction < 1/100 (only necessary for tiles observed before October 5, 2023)
    data_cleaned = data[np.bitwise_and((snr > 8), (pol_frac > 1/100))]
    FDFs_cleaned = FDFs[np.bitwise_and((snr > 8), (pol_frac > 1/100))]
    spectra_cleaned = spectra[np.bitwise_and((snr > 8), (pol_frac > 1/100))]
    
    data_new, FDFs_new, spectra_new  = removing_duplicates(data_cleaned, FDFs_cleaned, spectra_cleaned)

    data_new.write('./GOOD_Data/catalog_cleaned.csv', overwrite =True)
    FDFs_new.write('./GOOD_Data/FDFs_cleaned.fits', overwrite =True)
    spectra_new.write('./GOOD_Data/spectra_cleaned.fits', overwrite =True)

def ERGS_correction(catalog, cz, radius = 1, num_out = 40):
    '''
    @author: Craig Anderson
    Code to correct for the GRM using the exclusion radius GRM subraction (ERGS), see Anderson+2024
    Parameters
    ----------
    catalog: ascii astroy table 
        Catalog of RMs
    cz: int or float
        The redshift to the object of interest
    radius: int or float
        The size of the exclusion radius in Mpc. 1 Mpc by default
    num_out: int
        The number of sources outside the exclusion zone to be used to calculate the GRM
    '''
    file = 'faraday2020v2.fits'
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
    master_RM_table['rm_err_corrected'] = np.abs(master_RM_table['rm'])*0+9e9 

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
        def median_error_bootstrap(data, n_bootstrap=1000):
            data = data[~np.isnan(data)]
            medians = [np.median(np.random.choice(data, size=len(data), replace=True))
                    for _ in range(n_bootstrap)]
            return np.std(medians)
        median_RM_err = median_error_bootstrap(nearby_rms_uncorrected)
        #Implement correction
        master_RM_table[row_idx]['rm_corrected'] = master_RM_table[row_idx]['rm'] - median_RM
        master_RM_table[row_idx]['rm_correction'] = median_RM
        master_RM_table[row_idx]['rm_err_corrected'] = np.sqrt(master_RM_table[row_idx]['rm_err']**2 + median_RM_err**2 )
        
        #read GRM from Huts and save it
        l=rmcoord.galactic.l.degree
        b=rmcoord.galactic.b.degree

        master_RM_table[row_idx]['rm_hut'] = rm_map[pix_index(l,b,nside)]

    return master_RM_table

def scatter_plotting(r, rm, rm_err):
    '''
    Function to plot the RRM scatter profile

    Parameters
    --------
    r: numpy array object
        Radii 
    rm: numpy array object
        RMs 
    rm_err: numpy array object
        RM_errs
    
    Return
    ------
    None
    '''
    obsresults = Table()
    obsresults['r_sorted'] = r
    obsresults['RM'] = rm
    obsresults['RM_err'] = rm_err
    x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in = running_bins.calc_running_scatter(obsresults
                                                            , RMcol='RM' # column containing RRM
                                                            , xcol='r_sorted' # column containing distance to cluster
                                                            , RMerrcol = 'RM_err' # column containing RRM error
                                                            # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                            , xwidth=None # if we want set number of points
                                                            , M =  20      # number of points in sliding window
                                                            , method='iqr' # IQR is more robust than STD
                                                            , show=True
                                                            , sigmaRMextr = 6.99 # change this value to what scatter is far inside clusters
                                                                            # corrected for measurement errors!
                                                            , plotwidth=True
                                                            , plot_scatter=True
                                                            )
    fig, ax = plt.subplots(figsize = (5.84685039, 4.46338583))
    fig.set_facecolor('white')
    ax.fill_between(x_running_in, scatter_in - scatter_errlow_in, scatter_in + scatter_errup_in ,alpha=0.5, color='#1f77b4')
    ax.plot(x_running_in, scatter_in , color='#1f77b4')
    np.save('./GOOD_Data/scatter_A3581.npy',[x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in])

    #plt.fill_between(x_running_mask, scatter_mask - scatter_errlow_mask, scatter_mask + scatter_errup_mask ,alpha=0.5, color = 'tomato', label = 'Masked')
    #plt.plot(x_running_mask, scatter_mask, color = 'tomato')


    ax.set_ylabel(r"$\sigma_\mathrm{RRM, corr}$ (rad m$^{-2}$)")
    ax.set_xlabel('$r$ (Mpc)')
    ax.axvline(2 * 0.656, label = '2$R_{500, \mathrm{WH}}$', c = 'k', ls = '--')
    ax.axvline(2 * 0.925, label = '2$R_{500, \mathrm{eRASS1}}$', c = 'red', ls = '--')
    ax.set_xlim(0, 2)

    # Example bin width and location
    bin_width =0.2706555989289044
    x_center =0.2706555989289044 /2 + 0.2
    #bin_width = 0.13707605982961635
    #x_center = 0.13707605982961635 / 2 + 0.2
    y_level = 5  # relative height on the y-axis (can be data or axis fraction)


    # Add scale bar
    x_start = x_center - bin_width / 2
    x_end = x_center + bin_width / 2
    ax.hlines(y=y_level, xmin=x_start, xmax=x_end, colors='black', linewidth=2)
    ax.vlines([x_start, x_end], y_level - 0.5, y_level + 0.5, colors='black')  # end ticks
    ax.text(0.4, y_level - 2, f'{bin_width:.2f} Mpc', ha='center')
    plt.legend()
    plt.savefig('./GOOD_Figures/rrm_scatter.pdf', dpi = 300, bbox_inches = 'tight') 
    plt.close()

    plt.figure(facecolor='white', figsize = (5.84685039, 4.46338583))
    plt.errorbar(r[r < 2 * 0.925], rm[r < 2 * 0.925], yerr = rm_err[r < 2 * 0.925], fmt = '.', capsize = 5)
    plt.xlabel('$r$ (Mpc)')
    plt.xlim(0, 2)
    plt.ylabel('RRM (rad m$^{-2}$)')
    plt.axvline(2 * 0.656, label = '$2R_{500, \mathrm{WH}}$', c = 'k', ls = '--')
    plt.axvline(2 * 0.925, label = '$2R_{500, \mathrm{eRASS1}}$', c = 'red', ls = '--')
    plt.legend()
    plt.savefig('./GOOD_Figures/rrm_radial_profile.pdf', dpi = 300, bbox_inches = 'tight')

def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={format}")
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url
def getimages(ra,dec,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table

def optical_radio_contours(ra_bound, dec_bound, size = 1000):
    '''
    Function to create optical-radio contour overlays (no tick labels) using optical images from Pan-STARRS  and Stokes I from ASKAP

    Parameters
    ----------
    ra_bound: numpy array object
        ra of RMs
    dec_bound: numpy array object
        dec of RMs
    size: int 
        pixel size of the optical cutout 

    Returns
    -------
    None 
    '''
    for i in range(63, 64):
        figure = plt.figure(figsize=(10, 10), dpi=300)  # 10 inches × 100 dpi = 1000 px

        fh_r = geturl(ra_bound[i], dec_bound[i], size=size, filters="y", format="fits")[0]
        urllib.request.urlretrieve(fh_r, "r_image.fits")

        fh_g = geturl(ra_bound[i], dec_bound[i], size=size, filters="i", format="fits")[0]
        urllib.request.urlretrieve(fh_g, "g_image.fits")

        fh_b = geturl(ra_bound[i], dec_bound[i], size=size, filters="g", format="fits")[0]
        urllib.request.urlretrieve(fh_b, "b_image.fits")

        aplpy.make_rgb_cube(['r_image.fits', 'g_image.fits', 'b_image.fits'], 'cube.fits')
        aplpy.make_rgb_image('cube.fits', 'cube.png')

        fig = aplpy.FITSFigure('r_image.fits', figure = figure)
        fig.show_rgb('cube.png')
        sigma = 0.000179046
        fig.show_contour('cleaned_radio.fits', colors = 'white', levels = np.array([3, 6, 12, 24]) * sigma, linestyle = '-.-')
        #fig.tick_labels.set_font(size= 'x-large')
        fig.tick_labels.set_font(size= 'xx-large')
        fig.axis_labels.set_font(size= 'xx-large')
        fig.axis_labels.set_ytext('DEC (J2000)')
        fig.ax.set_position([0, 0, 1, 1])
        
        fig.show_markers(ra_bound, dec_bound, c = 'red', marker = 'X', s = 100)
        plt.savefig('./GOOD_Figures/optical-radio/{}'.format(str(i)) + 'ra-{:.6f}'.format(ra_bound[i]) +',dec-{:.6f}.pdf'.format(dec_bound[i]), dpi =300, bbox_inches = 'tight')
        print("Completed {}".format(str(i+1)) + " of {}".format(str(len(ra_bound))) + " sources...")

def optical_counterpart_matching(ra_masked, dec_masked, path):
    '''
    Function to find the optical counterpart for radio sources 
    
    Parameters 
    -----------
    ra_masked: numpy array object 
        RAs of the RMs 
    dec_masked: numpy array object
        DECs of the RMs
    path: str
        path where the optical-radio overlays are stored
    
    Returns
    --------
    None
    '''
    files_masked = glob(path+'/*')
    files_masked.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    # Initialize the lists for storing the clicks
    sky_right = []
    sky_left = []
    left_i_array = []
    right_i_array = []

    # Loop through images
    for i in range(80, 115):
        # Define global lists for storing clicks
        global right_clicks
        global left_clicks
        right_clicks = []
        left_clicks = []

        click_happened = False  # Flag to track if click happened

        # This function will be called on mouse click
        def mouse_callback(event, x, y, flags, params):
            nonlocal click_happened  # Ensure the callback updates the outer flag

            # Right-click event
            if event == cv2.EVENT_RBUTTONDOWN:
                right_clicks.append([x, y])
                click_happened = True

            # Left-click event
            if event == cv2.EVENT_LBUTTONDOWN:
                left_clicks.append([x, y])
                click_happened = True

        # Fetch the FITS file and WCS info
        fitsurl = geturl(ra_masked[i], dec_masked[i], size=1000, filters="i", format="fits")
        fh = fits.open(fitsurl[0])
        wcs = WCS(fh[0].header)

        # Load the image
        path_image = files_masked[i]
        img = cv2.imread(path_image)

        # Create OpenCV window
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        # Optional: Start window thread (only needed in special cases)
        cv2.startWindowThread()

        # Set mouse callback function for window
        cv2.setMouseCallback('image', mouse_callback)

        # Now display the image
        cv2.imshow('image', img)

        # Wait for the user to click
        while not click_happened:
            key = cv2.waitKey(20)  # Small delay to update window events

        # Once a click happens, close the window and move on
        cv2.destroyAllWindows()

        # If right click, save the coordinates as non-optical
        if len(right_clicks) > 0:
            sky_right.append(wcs.pixel_to_world(right_clicks[-1][0], 1000 - right_clicks[-1][1]))
            right_i_array.append(i)
        else:
            # Otherwise, save as optical counterpart
            sky_left.append(wcs.pixel_to_world(left_clicks[-1][0], 1000 - left_clicks[-1][1]))
            left_i_array.append(i)

    # Convert coordinates to degrees
    sky_left_deg = []
    sky_right_deg = []

    for idx in range(len(left_i_array)):
        sky_left_deg.append([sky_left[idx].ra.value, sky_left[idx].dec.value, left_i_array[idx]])

    for idx in range(len(right_i_array)):
        sky_right_deg.append([sky_right[idx].ra.value, sky_right[idx].dec.value, right_i_array[idx]])

    # Save to CSV
    file_optical = './GOOD_Data/optical_sources.csv'
    file_no_optical = './GOOD_Data/no_optical_sources.csv'

    mode = 'a'
    with open(file_optical, mode, newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows(sky_left_deg)

    with open(file_no_optical, mode, newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows(sky_right_deg)



def calc_phot_z(optical_name):
    '''
    Function to compute the photometric redshift

    Parameters
    ----------
    optical_name: str
        File name of the csv file that contains the location of the optical counterparts
    
    Returns
    --------
    z_phot: numpy array object
        The calculated photometric redshifts
    z_phot_err: numpy array object
        The calculated error in the photometric redshifts
    '''
    opthosts = ascii.read(optical_name, format = 'no_header')
    queried_data = querycatalogue(opthosts,racol='col1',deccol='col2'
                            , survey='panstarrs',X=3)
    feat_type = 'kron'

    ra_mean = queried_data['raMean']
    dec_mean = queried_data['decMean']
    r_kron = queried_data['rMeanKronMag']
    g = queried_data['gMeanKronMag']
    r = queried_data['rMeanKronMag']
    i = queried_data['iMeanKronMag']
    z = queried_data['zMeanKronMag']
    y = queried_data['yMeanKronMag']
    dir_name_cat_test = './GOOD_Data/final_test.fits'

    dir_name_ebv_map = './photoz_codes/tario_et_al_pythonscripts/lambda_sfd_ebv.fits'

    prepare_test_galaxies_tarrio(ra_mean, dec_mean, r_kron, g, r, i, z, y, feat_type, dir_name_cat_test, dir_name_ebv_map)

    dir_name_cat_test = './GOOD_Data/final_test.fits'
    dir_name_cat_out = './GOOD_Data/optical_predicted_z.fits'
    feat_type = 'kron'
    dir_name_cat_train = './photoz_codes/tario_et_al_pythonscripts/ps1_training_tarrio.fits'
    compute_photo_z_pan_tarrio(dir_name_cat_train, dir_name_cat_test, dir_name_cat_out
    , feat_type=feat_type, train_type='T5', std_type='T5'
    , use_5feat=1, use_4feat=1, Nneigh=100)
    optical_hdu = fits.open('./GOOD_Data/optical_predicted_z.fits')
    optical_table = Table(optical_hdu[1].data)
    z_phot = optical_table['z_phot']
    z_phot_err = optical_table['z_phot_err']
    return z_phot, z_phot_err

def redshift(z_phot, z_phot_err, z_spec, cz = 0.0221, gap = 1000):
    '''
    Function to produce the redshift plots and also store the best redshift and if the object is in the cluster (using a fixed gap of 1000 km/s)
    
    Parameters
    -----------
    z_phot: numpy array object
        Array of photometric redshifts 
    z_phot_err: numpy array object 
        Array of photometric errors
    z_spec: numpy array object
        Array of spectroscopic redshifts
    cz: float
        Cluster redshift 
    gap: int or float
        The fixed gap to be used in km/s for determination of cluster membership 

    Returns 
    -------
    z_best: numpy array object 
        Array of the 'best' redshift. If z_spec is available chooses it as default 
    in_clust: numpy array object 
        Boolean array with flags when a source is within the cluster as identified using the fixed gap
    '''
    
    '''
    
    figure = plt.figure(figsize = (5.84685039, 4.46338583))
    x = np.linspace(0, 2.5, 1000)
    plt.errorbar(z_spec[z_phot_err >= 0], z_phot[z_phot_err >= 0], yerr = z_phot_err[z_phot_err >= 0], fmt = '.', capsize = 5)
    plt.scatter(z_spec[z_phot_err < 0], z_phot[z_phot_err < 0], color = '#1f77b4')
    plt.plot(x, x, ls = '--')
    plt.axvline(0.0221, color = 'red', ls = '--', lw = 1, label = 'A3581 Redshift')
    plt.axhline(0.0221, color = 'red', ls = '--', lw = 1)
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.savefig('./GOOD_Figures/phot_spec.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()
    '''
    vel_clust =  cz * lightspeed.to(u.km / u.s)
    f = plt.figure(figsize = (5.84685039, 4.46338583))
    axs = f.subplots(2, 2)
    x = np.linspace(0, 2.5, 1000)
    # plot the same data on both axes
    for ax in axs.flatten():
        ax.errorbar(z_spec[z_phot_err >= 0], z_phot[z_phot_err >= 0], yerr = z_phot_err[z_phot_err >= 0], fmt = '.', capsize = 5)
        ax.scatter(z_spec[z_phot_err < 0], z_phot[z_phot_err < 0], color = '#1f77b4')
        ax.plot(x, x, ls = '--')
        ax.axvline(0.0221, color = 'red', ls = '--', lw = 1, label = 'A3581 Redshift')
        ax.axhline(0.0221, color = 'red', ls = '--', lw = 1)


    # zoom-in / limit the view to different portions of the data
    axs[0, 0].set_xlim(0, 0.3)  # outliers only
    axs[0, 1].set_xlim(2.3, 2.5)  # most of the data
    axs[1, 0].set_xlim(0, 0.3)  # outliers only
    axs[1, 1].set_xlim(2.3, 2.5)  # most of the data
    axs[0, 0].set_ylim(0.65, 0.9)  # outliers only
    axs[0, 1].set_ylim(0.65, 0.9)  # outliers only
    axs[1, 0].set_ylim(0, 0.35)  # most of the data
    axs[1, 1].set_ylim(0, 0.3)  # most of the data

    # hide the spines
    axs[0, 0].spines['bottom'].set_visible(False)
    axs[0, 1].spines['bottom'].set_visible(False)
    axs[1, 0].spines['top'].set_visible(False)
    axs[1, 1].spines['top'].set_visible(False)
    axs[0, 0].spines.right.set_visible(False)
    axs[1, 0].spines.right.set_visible(False)
    axs[0, 1].spines.left.set_visible(False)
    axs[1, 1].spines.left.set_visible(False)
    ax.legend()
    # hide ticks

    axs[0, 0].set_xticks([])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[1, 1].set_yticks([])
    '''
    axs[1, 0].set_xticks([0, 0.5, 1, 2])
    axs[1, 1].set_xticks([4, 4.5, 5])
    '''
    axs[1, 1].set_xticks([2.35, 2.45])

    d = .05  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=axs[0, 0].transAxes, color='k', clip_on=False)
    axs[0, 0].plot((-d, +d), (-d, +d), **kwargs) 
    axs[0, 0].plot((1-d, 1+d), (1-d, 1+d), **kwargs)        # top-left diagonal
    kwargs.update(transform=axs[1, 0].transAxes)  # switch to the bottom axes
    axs[1, 0].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    axs[1, 0].plot((1-d, 1+d), (-d, +d), **kwargs)  # bottom-center-left diagonal
    kwargs.update(transform=axs[1, 1].transAxes)  # switch to the bottom axes
    axs[1, 1].plot((-d, +d), (-d, d), **kwargs)  # bottom-left diagonal
    axs[1, 1].plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform = axs[0,1].transAxes)
    axs[0, 1].plot((-d, +d), (1-d, 1+d), **kwargs) 
    axs[0, 1].plot((1-d, 1+d), (-d, +d), **kwargs) 

    axs[1, 0].set_xlabel('Spectroscopic Redshift', ha = 'left', y =1)
    axs[1, 0].set_ylabel('Photometric Redshift', va = 'bottom', y = 1.2)
    plt.savefig('./GOOD_Figures/phot_spec.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()



    z_best = []
    in_clust = []

    for i in range(len(z_phot)):
        if z_spec[i] != - 1:
            z_best_curr = z_spec[i]
        else: 
            z_best_curr = z_phot[i]
        z_best.append(z_best_curr)
        if z_best_curr >= 0:
            vel_new = z_best_curr * lightspeed.to(u.km/u.s)
            if z_phot_err[i] > 0 :
                vel_err_new = z_phot_err[i] * lightspeed.to(u.km/u.s)
            else:
                vel_err_new = 0 * u.km / u.s
            interval1 = pd.Interval(vel_clust.value - gap, vel_clust.value + gap)

            interval2 = pd.Interval(vel_new.value - vel_err_new.value, vel_new.value + vel_err_new.value)
            if interval1.overlaps(interval2): 
                in_clust.append(1)
            else:
                in_clust.append(0)
        else:
            in_clust.append(0)
    z_best = np.array(z_best)
    plt.figure(facecolor = 'white', figsize = (5.84685039, 4.46338583))
    plt.hist(z_best[z_best > -1], bins = 100, edgecolor = 'k')
    plt.xlim(0, 0.5)
    plt.ylim(0, 11)
    plt.axvline(0.0221, ls = '--', color = 'red', label = 'A3581 Redshift')
    plt.legend()
    plt.xlabel('Redshift')
    plt.ylabel('Counts')
    plt.savefig('./GOOD_Figures/phot_hist.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()

    return z_best, np.array(in_clust)


def asymmetry_calc(ra_clust, dec_clust, rrm_clust, wcs, step = 0.01, show_plots = True, return_sigmas = False):
    '''
    A function that calculates the asymmetry in the RM grid of a galaxy cluster. The function does this by comparing the scatter on two sides of the cluster
    Parameters
    -----------
    ra_clust: numpy array of the cluster ras (in deg)
    dec_clust: numpy array of the cluster decs (in deg)
    rrm_clust: numpy array of the cluster rrms (in rad/m^2)
    step: float that indicates the step in the thetas
    
    Output
    -------
    theta_plot: the thetas that are used for plotting
    std_rrm1: the standard deviations for 'side 1', which is indicated as the right side of the line
    std_rrm2: the standard deviations for 'side 2', which is indicated as the left side of the line
    '''
    theta_linspace = np.arange(0, 181, step)
    std_rrm1 = []
    std_rrm1_err = []
    std_rrm2 = []
    std_rrm2_err = []
    theta_plot = []
    count = 0

    ra_new_cen, dec_new_cen =  np.sum(ra_clust * np.abs(rrm_clust))/np.sum(np.abs(rrm_clust)), np.sum(dec_clust * np.abs(rrm_clust))/np.sum(np.abs(rrm_clust))
    coord_new_cen = SkyCoord(ra_new_cen * u.deg, dec_new_cen * u.deg)
    for theta in theta_linspace:
        if 0 < theta < 90:
            x, y = wcs.world_to_pixel(SkyCoord(ra_clust * u.deg, dec_clust* u.deg))
            xcen, ycen = wcs.world_to_pixel(coord_new_cen)
            x1 = 1.1 *  max(x) 
            #y1 = (x1-xcen) * np.tan((theta+90) * np.pi / 180) + ycen
            y1 = (x1-xcen) * np.tan((theta) * np.pi / 180) + ycen
            #y1 = (x1 - xcen)/(np.tan(theta * np.pi/180)) + ycen
            
            x2 = 0.9 *  min(x)
            #y2 = (x2 - xcen) * np.tan((theta+90) * np.pi/ 180) + ycen
            y2 = (x2 - xcen) * np.tan((theta) * np.pi/ 180) + ycen
            #y2 = (x2 - xcen)/(np.tan(theta * np.pi/180)) + ycen

            m = (y2 - y1)/(x2-x1)
            b = y1 - m * x1
            yl = m * x + b
            
            deltay = yl - y

            rrm1 = rrm_clust[np.argwhere(deltay >0)]

            rrm2 = rrm_clust[np.argwhere(deltay < 0 )]

            std_rrm1.append(scipy.stats.iqr(rrm2)/1.349)
            std_rrm1_err.append(scipy.stats.iqr(rrm2)/(1.349 * np.sqrt(len(rrm2))))
            std_rrm2.append(scipy.stats.iqr(rrm1)/1.349)
            std_rrm2_err.append(scipy.stats.iqr(rrm1)/(1.349 * np.sqrt(len(rrm1))))

            #std_rrm1.append(np.mean(np.abs(rrm2)))
            #std_rrm1_err.append(scipy.stats.iqr(rrm2)/(1.349 * np.sqrt(len(rrm2))))
            #std_rrm2.append(np.mean(np.abs(rrm1)))
            #std_rrm2_err.append(scipy.stats.iqr(rrm1)/(1.349 * np.sqrt(len(rrm1))))
            
            
            #theta_plot.append(theta)

            theta_plot.append(theta +90)
            '''
            Diagnostic plots
            
            if theta == 500: 
                colors = []
                for i in range(len(deltay)):
                    if deltay[i] < 0:
                        colors.append('#ff7f0e')
                    else:
                        colors.append('#1f77b4')
                fig, ax = plt.subplots(ncols=1, subplot_kw={'projection': wcs},  figsize = (10, 5))
                c = ax.scatter(ra_clust, dec_clust, s = 70, c = rrm_clust, vmin = -40, vmax = 40, cmap = 'seismic', transform = ax.get_transform('icrs'), edgecolors='k')
                #ax.scatter(ra_clust, dec_clust, s = 10, c = colors, vmin = -40, vmax = 40, cmap = 'seismic', transform = ax.get_transform('icrs'))
                colorbar = plt.colorbar(c, ax =ax)
                colorbar.set_label('RRM (rad m$^{-2}$)')
                ax.scatter(ra_new_cen, dec_new_cen, c = 'k', transform = ax.get_transform('icrs'), marker = 'x', s = 500)
                #ax.axhline(dec_new_cen, transform = ax.get_transform('icrs'), c = 'k')
                ax.plot(wcs.pixel_to_world([x1, x2], [y1, y2]).ra.value, wcs.pixel_to_world([x1, x2], [y1, y2]).dec.value, c = 'k', transform = ax.get_transform('icrs'))
                ax.fill_between(wcs.pixel_to_world([x1, x2], [y1, y2]).ra.value, wcs.pixel_to_world([x1, x2], [y1, y2]).dec.value,  wcs.pixel_to_world([x1, x2], [y1+10**4, y2+10**4]).dec.value, color = '#ff7f0e', alpha = 0.2, transform = ax.get_transform('icrs'), label = 'Side 2')
                ax.fill_between(wcs.pixel_to_world([x1, x2], [y1, y2]).ra.value, wcs.pixel_to_world([x1, x2], [y1, y2]).dec.value,  wcs.pixel_to_world([x1, x2], [y1- 10**4, y2 - 10**4]).dec.value, color = '#1f77b4', alpha = 0.2, transform = ax.get_transform('icrs'), label = 'Side 1')
                ax.set_xlabel('RA (J2000)')
                ax.set_ylabel('DEC (J2000)')
                xlim = [min(ra_clust) - 0.3, max(ra_clust) + 0.3]*u.deg
                ylim = [min(dec_clust)-0.3, max(dec_clust)+0.3]*u.deg

                xlim, ylim = wcs.world_to_pixel(SkyCoord(xlim, ylim))

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                r_plotting = SphericalCircle((ra_cen * u.degree, dec_cen * u.degree),  (2* 0.925 * (u.Mpc) / scale).to(u.deg), linestyle = '--', facecolor = 'none', edgecolor = 'red', transform = ax.get_transform('icrs'))
                ax.add_patch(r_plotting)
                ax.invert_xaxis()
                ax.set_aspect(1)
                plt.legend()
                plt.title('$\\theta = {:.0f}$'.format(theta +90))
                plt.savefig('{:.2f}.pdf'.format(theta), dpi = 300, bbox_inches = 'tight')
                plt.show()
            '''

            '''
            if deltay[0] > 0:
                print('To the right')
            else:
                print('To the left')
        


            plt.scatter(ra_clust[0], dec_clust[0])
            plt.scatter(ra_clust, dec_clust)
            plt.scatter(ra_new_cen, dec_new_cen, c = 'k')
            plt.plot(np.array([x1, x2]) + ra_new_cen, np.array([y1, y2]) + dec_new_cen, c= 'green')

            plt.xlim(min(ra_clust) - 0.3, max(ra_clust) + 0.3)
            plt.scatter(ra-0.3_cen, dec_new_cen, c+0.3 = 'k')
            plt.axvline(ra_new_cen)
            plt.ylim(min(dec_clust), max(dec_clust))
            '''


        # Now we will consider the case that 90 < theta < 180

        if  90 < theta < 180:
            x, y = wcs.world_to_pixel(SkyCoord(ra_clust * u.deg, dec_clust * u.deg))

            x1 = 0.9 *  min(x)
            #y1 = -(x1 -xcen) / np.tan((theta) * np.pi / 180) + ycen
            y1 = -(x1 -xcen) / np.tan((theta -90) * np.pi / 180) + ycen
            #y1 = (x1-xcen) * np.tan(theta * np.pi / 180) + ycen

            x2 = 1.1 * max(x)
            #y2 = - (x2 - xcen) / np.tan((theta) * np.pi/ 180) + ycen
            y2 = - (x2 - xcen) / np.tan((theta -90) * np.pi/ 180) + ycen
            #y1 = (x1-xcen) * np.tan(theta * np.pi / 180) + ycen

            m = (y2 - y1)/(x2-x1)

            b = y1 - m * x1
            yl = m * x + b

            deltay = yl - y

            rrm1 = rrm_clust[np.argwhere(deltay < 0 )]
            rrm2 = rrm_clust[np.argwhere(deltay > 0)]

            std_rrm1.append(scipy.stats.iqr(rrm1)/1.349)
            #std_rrm1.append(np.mean(np.abs(rrm1)))
            std_rrm1_err.append(scipy.stats.iqr(rrm1)/(1.349 * np.sqrt(len(rrm1))))
            std_rrm2.append(scipy.stats.iqr(rrm2)/1.349)
            #std_rrm2.append(np.mean(np.abs(rrm2)))
            std_rrm2_err.append(scipy.stats.iqr(rrm2)/(1.349 * np.sqrt(len(rrm2))))

            #theta_plot.append(theta)
            theta_plot.append(theta - 90)
            #if count % 1200 == 0: 
            if theta == 142:
                fig, ax = plt.subplots(ncols=1, subplot_kw={'projection': wcs},  figsize = (10, 5))
                colors = []
                for i in range(len(deltay)):
                    if deltay[i] > 0:
                        colors.append('#ff7f0e')
                    else:
                        colors.append('#1f77b4')
                
                #ax.scatter(ra_clust, dec_clust, s = 200, alpha = 0.5, c = colors, vmin = -40, vmax = 40, cmap = 'seismic', transform = ax.get_transform('icrs'))
                c = ax.scatter(ra_clust, dec_clust, c = rrm_clust, vmin = -40, vmax = 40, cmap = 'seismic', transform = ax.get_transform('icrs'), edgecolors='k')
                colorbar = plt.colorbar(c, ax =ax)
                colorbar.set_label('RRM (rad m$^{-2}$)')
                ax.scatter(ra_new_cen, dec_new_cen, c = 'k', transform = ax.get_transform('icrs'), marker = '+', s = 500)
                #ax.axhline(dec_new_cen, transform = ax.get_transform('icrs'), c = 'k')
                ax.plot(wcs.pixel_to_world([x1, x2], [y1, y2]).ra.value, wcs.pixel_to_world([x1, x2], [y1, y2]).dec.value, c = 'k', transform = ax.get_transform('icrs'))
                ax.fill_between(wcs.pixel_to_world([x1, x2], [y1, y2]).ra.value, wcs.pixel_to_world([x1, x2], [y1, y2]).dec.value,  wcs.pixel_to_world([x1, x2], [y1-10**4, y2-10**4]).dec.value, color = '#1f77b4', alpha = 0.2, transform = ax.get_transform('icrs'), label = 'Side 1')
                ax.fill_between(wcs.pixel_to_world([x1, x2], [y1, y2]).ra.value, wcs.pixel_to_world([x1, x2], [y1, y2]).dec.value,  wcs.pixel_to_world([x1, x2], [y1+10**4, y2+10**4]).dec.value, color = '#ff7f0e', alpha = 0.2, transform = ax.get_transform('icrs'), label = 'Side 2')
                ax.set_xlabel('RA (J2000)')
                ax.set_ylabel('DEC (J2000)')
                r_plotting = SphericalCircle((ra_cen * u.degree, dec_cen * u.degree),  (2* 0.925 * (u.Mpc) / scale).to(u.deg), linestyle = '--', facecolor = 'none', edgecolor = 'red', transform = ax.get_transform('icrs'))
                ax.add_patch(r_plotting)
                xlim = [min(ra_clust) - 0.3, max(ra_clust) + 0.3]*u.deg
                ylim = [min(dec_clust) - 0.3, max(dec_clust) + 0.3]*u.deg

                xlim, ylim = wcs.world_to_pixel(SkyCoord(xlim, ylim))

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.invert_xaxis()
                ax.set_aspect(1)
                plt.legend()
                plt.title('$\\theta = {:.0f}^\circ$'.format(theta - 90))
                plt.savefig('./GOOD_Figures/{:.2f}.pdf'.format(theta), dpi = 300, bbox_inches = 'tight')
                plt.show()
                plt.close()
                
        count += 1
    if show_plots:
        std_rrm1 = np.array(std_rrm1)
        std_rrm2 = np.array(std_rrm2)
        std_rrm1_err = np.array(std_rrm1_err)
        std_rrm2_err = np.array(std_rrm2_err)
        theta_plot = np.array(theta_plot)
        std_rrm1 = std_rrm1[np.argsort(theta_plot)]
        std_rrm1_err = std_rrm1_err[np.argsort(theta_plot)]
        std_rrm2 = std_rrm2[np.argsort(theta_plot)]
        std_rrm2_err = std_rrm2_err[np.argsort(theta_plot)]
        theta_plot = np.sort(theta_plot)
        plt.figure(figsize = (5.84685039, 4.46338583))
        plt.plot(theta_plot, std_rrm1, label  = 'Side 1')
        plt.plot(theta_plot, std_rrm2, label = 'Side 2')
        plt.xlabel('$\\theta$ (deg)')
        plt.ylabel('$\sigma_{\mathrm{RRM}}$ (rad m$^{-2}$)')
        plt.legend()
        #plt.savefig('./Figures/asymm1.pdf', dpi = 300, bbox_inches  = 'tight')
        plt.show()
        plt.close()
        
        plt.figure(figsize = (5.84685039, 4.46338583))
        plt.fill_between(theta_plot, np.abs(std_rrm1 - std_rrm2) - np.sqrt(std_rrm1_err**2 + std_rrm2_err**2), np.abs(std_rrm1 - std_rrm2)  + np.sqrt(std_rrm1_err**2 + std_rrm2_err**2), alpha = 0.5, color = '#1f77b4')

        plt.plot(theta_plot, np.abs(std_rrm1 - std_rrm2), color = '#1f77b4')
        plt.axhline(0, color = 'red', ls = '--')
        plt.xlabel('$\\theta$ (deg)')
        plt.ylabel('$|\sigma_{\mathrm{\mathrm{RRM}, 1}} - \sigma_{\mathrm{\mathrm{RRM}, 2}}|$ (rad m$^{-2}$)')
        
        plt.savefig('./GOOD_Figures/asymm2.pdf', dpi = 300, bbox_inches  = 'tight')
        plt.show()
        plt.close()
    if return_sigmas:
        return theta_plot, std_rrm1, std_rrm2, std_rrm1_err, std_rrm2_err

def crossing_zero(theta_plot, std_rrm1, std_rrm2):
    '''
    Function to find where the graph crosses zero 

    Parameters
    ----------
    theta_plot: numpy array object
        Array of theta values 
    std_rrm1: numpy array object 
        Array of the standard deviations in side 1
    std_rrm2: numpy array object
        Array of the standard deviations in side 2
    
    Returns
    ---------
    crossing_zeros[0]: int 
        The first point where the difference in standard deviations hits 0 
    crossing_zeros[-1]: int 
        The second point where the difference in standard deviations hits 0
    '''
    std1 = np.array(std_rrm1)
    std2 = np.array(std_rrm2)
    diff = std1 - std2 
    crossing_zeros = []
    for i in range(len(diff) - 1): 
        if diff[i] * diff[i+1] < 0: 
            crossing_zeros.append(theta_plot[i])
    return crossing_zeros[0], crossing_zeros[-1]

def calc_CORM_errprop(ra,dec,rrm, ra_err, dec_err, rrm_err):
    '''
    Calculating CORM with errors
    '''
    ra = unumpy.uarray(ra, ra_err)
    dec = unumpy.uarray(dec, dec_err)
    rrm = unumpy.uarray(rrm, rrm_err)

    corm_ra = np.sum(ra* np.abs(rrm)) / np.sum(np.abs(rrm))
    corm_dec = np.sum(dec* np.abs(rrm)) / np.sum(np.abs(rrm))

    return corm_ra, corm_dec

def A3581_rm_plotting(ra_sorted, dec_sorted, rrm_sorted, corm_ra, corm_dec, theta, ra_cen, dec_cen):
    '''
    Function the RMs on the sky 

    Parameters
    -----------
    ra: numpy array object
        Array of the RAs
    dec: numpy array object 
        Array of the DECs
    rrm: numpy array objecy
        Array of the RRMs
    corm_ra: float
        The RA of the CORM
    corm_dec: float
        The DEC of the CORM
    theta: float 
        The angle of the axis of symmetry
    ra_cen: float
        The RA of the X-ray centroid 
    dec_cen: float 
        The DEC of the X-ray centroid

    Returns
    -------
    None
    '''
    fig, ax = plt.subplots(ncols=1, subplot_kw={'projection': wcs}, figsize = (10, 5))
    fig.patch.set_facecolor('white')
    #c = ax.scatter(ra_sorted, dec_sorted, c = sigmaRM_sorted, cmap = 'binary', transform = ax.get_transform('icrs'))
    c = ax.scatter(ra_sorted, dec_sorted, c = rrm_sorted, cmap = 'seismic', transform = ax.get_transform('icrs'), vmin = -40, vmax = 40, edgecolors = 'black')
    colorbar = fig.colorbar(c, ax = ax)
    r_plotting = SphericalCircle((ra_cen * u.degree, dec_cen * u.degree),  (2* 0.925 * (u.Mpc) / scale).to(u.deg), linestyle = '--', facecolor = 'none', edgecolor = 'k', transform = ax.get_transform('icrs'), label = 'A3581')
    ax.add_patch(r_plotting)
    #colorbar.set_label('$\Sigma_{\mathrm{RM}}$ (rad m$^{-2}$)')
    colorbar.set_label('RRM (rad m$^{-2}$)')
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('DEC (J2000)')
    x, y = wcs.world_to_pixel(SkyCoord(ra_sorted * u.deg, dec_sorted* u.deg))
    coord_line_cen = SkyCoord((corm_ra) * u.deg, (corm_dec)* u.deg)
    xcen, ycen = wcs.world_to_pixel(coord_line_cen)
    theta = theta + 90
    x1 =  max(x) 
    y1 = (x1-xcen) * np.tan(theta * np.pi / 180) + ycen

    x2 =  min(x)
    y2 = (x2 - xcen) * np.tan(theta * np.pi/ 180) + ycen
    ax.plot(wcs.pixel_to_world([x1, x2], [y1, y2]).ra.value, wcs.pixel_to_world([x1, x2], [y1, y2]).dec.value, c = 'k', transform = ax.get_transform('icrs'), label = 'Filament', ls = '--')


    x, y = wcs.world_to_pixel(SkyCoord(ra_sorted * u.deg, dec_sorted* u.deg))
    coord_line_cen = SkyCoord((ra_cen ) * u.deg, (dec_cen)* u.deg)
    xcen, ycen = wcs.world_to_pixel(coord_line_cen)
    theta = 143-12
    x1 =   max(x) 
    y1 = (x1-xcen) * np.tan(theta * np.pi / 180) + ycen

    x2 = min(x)
    y2 = (x2 - xcen) * np.tan(theta * np.pi/ 180) + ycen

    #ax.plot(wcs.pixel_to_world([x1, x2], [y1, y2]).ra.value, wcs.pixel_to_world([x1, x2], [y1, y2]).dec.value, c = 'red', transform = ax.get_transform('icrs'), label = 'Filament', ls = '--')

    ax.scatter(corm_ra, corm_dec, c ='black', transform = ax.get_transform('icrs'), s = 500, marker ='+')

    ax.set_aspect(1)
    xlim = [ra_cen - 2.1, ra_cen + 2.1]*u.deg
    ylim = [dec_cen - 2, dec_cen + 2]*u.deg

    xlim, ylim = wcs.world_to_pixel(SkyCoord(xlim, ylim))
    plt.title('A3581 RRMs')


    plt.savefig('./GOOD_Figures/A3581_rm.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()



def density_ploting(r):
    '''
    Function for plotting the RM source density (in deg^-2) as a function of radius from the cluster centre

    Parameters
    ----------
    r: numpy array object 
        Array of radii of RMs
    '''    
    bin_width = 0.3

    bins = np.arange(0, 2 * 0.925+ bin_width, bin_width)  # Annuli edges (0 to 1, step 0.1)

    # Count points in each annulus
    counts, edges = np.histogram(r, bins=bins)

    # Calculate the area of each annulus
    areas = np.pi * (edges[1:]**2 - edges[:-1]**2)  # Area of annuli

    # Calculate area density
    areas = areas/ ((scale.to(u.Mpc/u.deg)).value)**2
    error = np.sqrt(counts)/areas
    densities = counts / areas
    plt.figure(facecolor='white', figsize = (5.84685039, 4.46338583))
    # Plot the area density as a histogram
    bin_centers = 0.5 * (edges[:-1] + edges[1:])  # Midpoints of each bin
    plt.bar(bin_centers, densities, width=bin_width, edgecolor='black', alpha=0.7)
    plt.errorbar(bin_centers, densities, yerr = error, fmt = '.', capsize = 5)
    plt.xlabel('$r$ (Mpc)')
    plt.ylabel('RRM density (deg $^{-2}$)')
    plt.savefig('./GOOD_Figures/rm_dens.pdf', dpi = 300, bbox_inches ='tight')
    plt.show()
    plt.close()

def obtain_complexity(sigma_add_bound, m2_bound, model_bound, red_chi2_bound, sigma_add_threshold = 1, m2_threshold = 0.5, sigma_add_plus = 10**(-0.6), sigma_add_minus = 10**(-0.65), dm2 = 0.1):
    '''
    @author: Affan Khadir 
    Function to obtain the Faraday complexity of sources as defined in Khadir et al. (2024)

    Parameters
    -----------
    sigma_add_bound: numpy array object 
        Array of sigma_add values
    m2_bound: numpy array object 
        Array of m2 values
    model_bound: numpy array object 
        Array of best-fit QU model value
    red_chi2_bound: numpy array object 
        Array of reduced chi2 for best-fit QU models
    sigma_add_threshold: float 
        threshold for sigma_add, 1 by default 
    m2_threshold: float 
        threshold for m2, 0.5 by default
    sigma_add_plus: float
        upper err on sigma_add threshold, 10**(-0.6) by default
    sigma_add_minus: float
        lower err on sigma_add threshold, 10**(-0.65) by default
    dm2: float 
        err on m2_threshold, 0.1 by default

    Returns
    -------
    complexity_bound: numpy array object 
        Array of complexity flags. 1 for a Faraday complex source and 0 otherwise
    '''
    complexity_bound = []
    for i in range(len(sigma_add_bound)):
        if sigma_add_bound[i] < sigma_add_threshold - sigma_add_minus and m2_bound[i] < m2_threshold - dm2:
            complexity = 0
        elif sigma_add_bound[i] > sigma_add_threshold + sigma_add_plus and m2_bound[i] > m2_threshold + dm2:
            if model_bound[i] != 'm1':
                complexity = 1
            elif model_bound[i] == 'm1' and np.abs(red_chi2_bound[i] - 1) <= 0.5:
                complexity = 0
            else:
                complexity = 1
        else: 
            if model_bound[i] != 'm1':
                complexity = 1
            elif model_bound[i] == 'm1' and np.abs(red_chi2_bound[i] -1) <= 0.5:
                complexity  = 0
            else:
                complexity = 1
        complexity_bound.append(complexity)
    complexity_bound = np.array(complexity_bound)
    return complexity_bound

def complexity_plotting(catalog):
    '''
    Function to create the complexity plots

    Parameters
    ----------
    catalog: astropy Table object
        catalog of RM sources 

    Returns 
    --------
    None
    '''
    in_clust = np.ravel(catalog['in_clust'].data)
    catalog = catalog[in_clust == 0]
    SigmaRM = np.ravel(catalog['SigmaRM'].data)
    SigmaRM_errm = np.ravel(catalog['SigmaRMerrm'])
    SigmaRM_errp = np.ravel(catalog['SigmaRMerrp'])
    SigmaRM_err = np.mean([SigmaRM_errm, SigmaRM_errp], axis = 0)
    for i in range(len(SigmaRM_err)):
        if SigmaRM[i] == -1:
            SigmaRM[i] = 0
            SigmaRM_err[i] = 0
        if SigmaRM_err[i] > 10: 
            SigmaRM_err[i] = np.min([SigmaRM_errm[i], SigmaRM_errp[i]])


    rrm = np.ravel(catalog['rm_corrected'].data)
    rrm_err = np.ravel(catalog['rm_err_corrected'].data)



    fig, ax = plt.subplots(figsize = (5.84685039, 4.46338583))
    fig.set_facecolor('white')
    ax.errorbar(SigmaRM[SigmaRM > 0], np.abs(rrm[SigmaRM > 0]), xerr = SigmaRM_err[SigmaRM > 0] , yerr = rrm_err[SigmaRM > 0], fmt = '.', capsize = 5, color='#1f77b4', label = 'Depolarization Measured')
    ax.errorbar(SigmaRM[SigmaRM  == 0], np.abs(rrm[SigmaRM == 0]), xerr = SigmaRM_err[SigmaRM == 0], yerr = rrm_err[SigmaRM == 0], fmt = '.', capsize = 5, color='gray', label = 'No Depolarization Measured')    
    ax.set_xlabel('$\Sigma_{\mathrm{RM}}$ (rad m$^{-2}$)')
    ax.set_ylabel('|RRM| (rad m$^{-2}$)')
    plt.ylim(-5, 65)
    plt.savefig('./GOOD_Figures/new_rrm_depol.pdf', dpi = 300, bbox_inches = 'tight')
    print(pearsonr(SigmaRM[np.bitwise_and((SigmaRM > 0), np.abs(rrm) < 60)], np.abs(rrm)[np.bitwise_and((SigmaRM > 0), np.abs(rrm) < 60)]))
    plt.show()
    plt.close()

    fig, ax = plt.subplots(ncols=1, subplot_kw={'projection': wcs}, figsize = (10, 5))
    fig.patch.set_facecolor('white')

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.family'] = 'Courier'
    #c = ax.scatter(ra_nnew, dec_nnew, c = sigmaRM_nnew, cmap = 'binary', transform = ax.get_transform('icrs'))
    colour_nnew = []

    ra_nnew = np.ravel(catalog['ra'].data)
    dec_nnew = np.ravel(catalog['dec'].data)
    complex_array = np.ravel(catalog['complexity'].data)
    c = ax.scatter(ra_nnew[complex_array > 0], dec_nnew[complex_array > 0], c = 'k', transform = ax.get_transform('icrs'), s = 10)


    r_plotting = SphericalCircle((ra_cen * u.degree, dec_cen * u.degree),  (2* 0.925 * (u.Mpc) / scale).to(u.deg), linestyle = '--', facecolor = 'none', edgecolor = 'k', transform = ax.get_transform('icrs'), label = 'A3581')
    ax.add_patch(r_plotting)
    

    x, y = wcs.world_to_pixel(SkyCoord(ra_nnew * u.deg, dec_nnew* u.deg))
    coord_line_cen = SkyCoord((212.052) * u.deg, (-27.137)* u.deg)
    xcen, ycen = wcs.world_to_pixel(coord_line_cen)
    theta = 142
    x1 = 1 *  max(x) 
    y1 = (x1-xcen) * np.tan(theta * np.pi / 180) + ycen

    x2 = min(x)
    y2 = (x2 - xcen) * np.tan(theta * np.pi/ 180) + ycen
    ax.plot(wcs.pixel_to_world([x1, x2], [y1, y2]).ra.value, wcs.pixel_to_world([x1, x2], [y1, y2]).dec.value, c = 'k', transform = ax.get_transform('icrs'), label = 'Filament', ls = '--')
    filament_rms_nnew = []
    coord_nnew = SkyCoord(ra_nnew*u.deg, dec_nnew*u.deg)

    for i in range(len(ra_nnew)):
        sep_current = (np.min(coord_nnew[i].separation(wcs.pixel_to_world(np.linspace(x1, x2, 1000), np.linspace(y1, y2, 1000)))) * scale.to(u.Mpc/u.deg)).to(u.Mpc).value
        if sep_current < 0.2: 
            filament_rms_nnew.append(1)
        else: 
            filament_rms_nnew.append(0)



    x, y = wcs.world_to_pixel(SkyCoord(ra_nnew * u.deg, dec_nnew* u.deg))
    coord_line_cen = SkyCoord((212.052 ) * u.deg, (-27.137)* u.deg)
    xcen, ycen = wcs.world_to_pixel(coord_line_cen)
    theta = 146
    x11 = 1 *  max(x) 
    y11 = (x11-xcen) * np.tan(theta * np.pi / 180) + ycen
    x21 =  min(x)
    y21 = (x21 - xcen) * np.tan(theta * np.pi/ 180) + ycen
    #ax.plot(wcs.pixel_to_world([x11, x21], [y11, y21]).ra.value, wcs.pixel_to_world([x11, x21], [y11, y21]).dec.value, c = 'red', transform = ax.get_transform('icrs'), label = 'Filament', ls = '--')

    ax.scatter(212.052, -27.137, c ='black', transform = ax.get_transform('icrs'), s = 500, marker ='+')

    for idx,fd in enumerate(rrm):
        ra_t = ra_nnew[idx]
        dec_t = dec_nnew[idx]
        
        if fd>=0:
            r = SphericalCircle((ra_t*u.degree, dec_t*u.degree), np.abs(fd)* 3e-3*u.degree, edgecolor='red', facecolor='none', transform=ax.get_transform('icrs'), ls ='--')
            #ax.scatter(ra_t * u.degree, dec_t * u.degree, c = 'blue', marker = 'X')
            ax.add_patch(r)
        else:
            r = SphericalCircle((ra_t*u.degree, dec_t*u.degree), np.abs(fd)* 3e-3*u.degree, edgecolor='blue', facecolor='none', transform=ax.get_transform('icrs'), ls ='--')
            
            ax.add_patch(r)

    #colorbar.set_label('$\Sigma_{\mathrm{RM}}$ (rad m$^{-2}$)')
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('DEC (J2000)')
    ax.set_aspect(1)
    #plt.title('A3581 Complexity')
    plt.savefig('./GOOD_Figures/complex_sky.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()

def average_Stokes(freq, stokesI, stokesI_err, pol_frac):
    '''
    Function to find the best-fit average stokes I_0, alpha, and pol_frac

    Parameters
    ----------
    freq: numpy array object
        of frequencies (in MHz)
    stokesI: numpy array object 
        of Stokes I intensity in mJy
    pol_frac: numpy array object
        of fractional polarization 

    Returns:
    ---------
    I0_mean: float 
        Average Stokes I intensity at 800 MHz acorss all background sources
    I0_err: float   
        Error in Stokes I intensity at 800 MHz acorss all background sources
    alpha_mean: float 
        Average power law index for Stokes I intensity across all background sources
    alpha_err: float 
        Error in power law index for Stokes I intensity at 800 MHz across all background sources
    pol_frac_mean: float
        Average polarization fraction across all background sources     
    '''
    def power(x, I0, alpha):
        '''
        Function to model power law
        '''
        return I0 * (x/freq[0])**(alpha)

    I0_array = np.array([])
    I0_err_array = np.array([])
    alpha_array = np.array([])
    alpha_err_array = np.array([])
    plt.figure()

    for i in range(len(stokesI)):
        popt, pcov = curve_fit(power, freq[~np.isnan(stokesI[i])], stokesI[i][~np.isnan(stokesI[i])], sigma = stokesI_err[i][~np.isnan(stokesI[i])], absolute_sigma = True, p0 = (stokesI[i][0], -0.5))
        I0, alpha = popt[0], popt[1]
        I0_err, alpha_err = np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1]

        plt.plot(freq, stokesI[i], alpha = 0.05)
        I0_array = np.append(I0_array, I0)
        I0_err_array = np.append(I0_err_array, I0_err)
        alpha_array = np.append(alpha_array, alpha)
        alpha_err_array = np.append(alpha_err_array, alpha_err)
    
    I0_mean = np.median(I0_array)
    #I0_err = 1/len(I0_array) * np.sqrt(np.sum(I0_err_array**2))
    I0_err = np.std(I0_array)
    alpha_mean = np.median(alpha_array)
    #alpha_err = 1/len(alpha_array) * np.sqrt(np.sum(alpha_err_array**2))
    alpha_err = np.std(alpha_array)

    plt.plot(freq, power(freq, I0_mean, alpha_mean), c = 'red', label = 'Mean Stokes I')
    plt.ylim(0, 0.05)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Stokes I (mJy)')
    plt.savefig('./GOOD_Figures/stokesI_avg.pdf', dpi = 300, bbox_inches = 'tight')
    plt.legend()
    plt.show()
    plt.close()

    pol_frac_mean = np.median(pol_frac)

    return I0_mean, I0_err, alpha_mean, alpha_err, pol_frac_mean



def complex_averaging(ra_complex, dec_complex, rmimage, beamsize, pol_frac_mean, I0_mean, alpha_mean, freq, plot = False):
    '''
    Function to average Stokes Q and U for all pixels within a beam of the complex RM and to then conduct RM-synthesis on the averaged Stokes parameters to model beam depolarization

    Parameters
    ----------
    ra_complex: int
        complex RM RAs (pixel coordinates)
    dec_complex: int
        complex RM DECs (pixel coordinates)
    rmimage: numpy array object
        Array of RMs in the model 
    beamsize: int
        The beam size in pixels 
    
    Returns
    --------
    rm_complex: float
        Value of complex RM at the location of ra_complex, dec_complex 
    '''
    
    x_mesh, y_mesh = np.meshgrid(np.arange(2048), np.arange(2048))
    positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
    tree = scipy.spatial.KDTree(positions.T)
    indices = tree.query_ball_point([ra_complex, dec_complex], r = beamsize)
    ra_in_pix, dec_in_pix = positions.T[indices].T[0], positions.T[indices].T[1]
    rm_in_pix = rmimage[ra_in_pix, dec_in_pix]

    psi_0 = np.deg2rad(45)
    wl = (lightspeed.to(u.m/u.s) / freq).value
    def power(x, I0, alpha):
        '''
        Function to model power law
        '''
        return I0 * (x/wl[0])**(-alpha)
    Q_array = []
    Q_err_array = []
    U_array = []
    I_err_array = []
    U_err_array = []
    if plot:
        plt.figure(figsize = (5.84685039, 4.46338583))
    for j in range(len(rm_in_pix)):
        Q = pol_frac_mean * power(wl, I0_mean, alpha_mean)* np.cos(2 * psi_0 + 2 * rm_in_pix[j] * wl**2)
        U = pol_frac_mean * power(wl, I0_mean, alpha_mean) * np.sin(2 * psi_0 + 2 * rm_in_pix[j] * wl**2)
        Q_err = np.random.normal(scale = 1e-5* pol_frac_mean * I0_mean, size = len(Q))
        U_err = np.random.normal(scale = 1e-5 * pol_frac_mean * I0_mean, size = len(U))
        Q_array.append(Q)
        Q_err_array.append(Q_err)
        U_array.append(U)
        U_err_array.append(U_err)

        I_err = np.random.normal(scale = 1e-5 * I0_mean, size = len(Q))
        I_err_array.append(I_err)
        if plot:
            df = np.column_stack((freq, I0_mean * (freq/freq[0])**alpha_mean, Q, U, I_err, Q_err, U_err))
            np.savetxt('data1.dat', df)
            os.system('rmsynth1d data1.dat -S')
            os.system('rmclean1d data1.dat -S')
            phi, q_plotting, u_plotting = np.loadtxt('data1_FDFclean.dat', unpack = True)
            plt.plot(phi, np.sqrt(q_plotting**2 + u_plotting**2), alpha = 0.1)
    
    Q_mean = np.mean(Q_array, axis = 0)
    Q_err = (1/len(Q_array)) *  np.sqrt(np.sum(np.array(Q_err_array)**2, axis = 0))
    U_mean = np.mean(U_array, axis = 0) 
    U_err = (1/len(U_array)) *  np.sqrt(np.sum(np.array(U_err_array)**2, axis = 0))
    I_err = (1/len(U_err_array)) *  np.sqrt(np.sum(np.array(I_err_array)**2, axis = 0))

    df = np.column_stack((freq, I0_mean * (freq/freq[0])**alpha_mean, Q_mean, U_mean, I_err, Q_err, U_err))
    np.savetxt('data1.dat', df)
    os.system('rmsynth1d data1.dat -S')
    os.system('rmclean1d data1.dat -S')
    with open('data1_RMclean.json') as json_file:
        data = json.load(json_file)
        rm_complex = data['phiPeakPIfit_rm2']
    
    phi, q_plotting, u_plotting = np.loadtxt('data1_FDFclean.dat', unpack = True)
    if plot:
        plt.plot(phi, np.sqrt(q_plotting**2 + u_plotting**2), c = 'red', label = 'Mean FDF')
        plt.axvline(rm_complex, c = 'k', label = 'Mean RM = {:.2f}'.format(rm_complex))
        plt.xlim(-200, 200)
        plt.xlabel('$\phi$ (rad m$^{-2}$)')
        plt.ylabel('Flux (mJy)')
        plt.legend()
        plt.savefig('./GOOD_Figures/mean_FDF.pdf', dpi = 300, bbox_inches = 'tight')
        plt.show()
        plt.close()
    return rm_complex    

def world_to_pix(ra_curr, dec_curr, ra_cen, dec_cen, scale, pixsize, N_pix):
    '''
    Function to convert from ICRS coordinates to pixel coordinates (assuming a flat-sky)

    Parameters
    ----------
    ra_curr: numpy array object
        Array of ras (ICRS)
    dec_curr: numpy array object 
        Array of decs (ICRS)
    ra_cen: float 
        RA of the cluster centre 
    dec_cen: float 
        DEC of the cluster centre 
    scale: astropy object 
        Physical to angular conversion (Must be in kpc/deg)
    pixsize: int or float
        The physical size of each pixel in kpc
    N_pix: int
        The number of pixels in each side of the box

    Returns
    -------
    ra_pix: numpy array object 
        Array of pixel RAs
    dec_pix: numpy array object 
        Array of pixels DECs
    '''
    ra_pix, dec_pix = [], []
    for i in range(len(ra_curr)):
        if ra_curr[i] < ra_cen:
            ra_pix_curr = N_pix//2 - int((ra_cen - ra_curr[i]) * (scale / pixsize).value)
        else:
            ra_pix_curr = N_pix//2 + int((ra_curr[i] - ra_cen) * (scale / pixsize).value)
        
        if dec_curr[i] < dec_cen:
            dec_pix_curr = N_pix//2 - int((dec_cen - dec_curr[i]) * (scale / pixsize).value)
        else:
            dec_pix_curr = N_pix//2 + int((dec_curr[i] - dec_cen) * (scale / pixsize).value)

        ra_pix.append(ra_pix_curr)
        dec_pix.append(dec_pix_curr)

    return np.array(ra_pix), np.array(dec_pix)

def model_comparison(ra, dec, complexity, mod_file_name_log, mod_file_name, pol_frac_mean = 0.06545418353102181, I0_mean = 0.007457496080425132, alpha_mean = -0.7678407583726873):
    '''
    Function to creat the comarison plots of the models with and without lognormal fluctuations

    Parameters
    ----------
    ra: numpy array object 
        Array of RAs
    dec: numpy array objec
        Array of DECs
    complexity: numpy array object
        Array of flags for complex sources (used to model beam depolarization by averaging Stokes Q and U)
    mod_file_name_log: str 
        Path to file with the lognormal models
    mod_file_name: str
        Path to file with the models
    pol_frac_mean: float
        mean polarisation fraction for modelling Stokes Q and U for complex sources. Default to median values found in A3581 
    I0_mean: float
        mean Stokes I intensity at 800 MHz for modelling Stokes Q and U for complex sources. Default to median values found in A3581 
    alpha_mean: float
        mean Stokes I spectral index for modelling Stokes Q and U for complex sources. Default to median values found in A3581 

    Returns
    --------
    None
    '''
    files_log = glob(mod_file_name_log)
    files = glob(mod_file_name)
    # Storing all the file names in a dcitionary for easy access 

    files_dict_log = {}

    files_dict = {}

    dict_names = []

    B0_names = ['1.0', '5.0', '10.0']

    eta_names = ['0.00', '0.25', '0.50']

    for i in range(len(B0_names)):
        for j in range(len(eta_names)): 
            dict_names.append('B0={:s}'.format(B0_names[i]) + '_eta={:s}'.format(eta_names[j]))
            files_dict_log['B0={:s}'.format(B0_names[i]) + '_eta={:s}'.format(eta_names[j])] = [] # Initializing the dictionary
            files_dict['B0={:s}'.format(B0_names[i]) + '_eta={:s}'.format(eta_names[j])] = []
    # Placing the files in the relevant dictionary entry
    for i in range(len(files)):
        curr_dict_name_log = files_log[i].split('_')[8]+ '_'+ files_log[i].split('_')[10]
        curr_dict_name = files[i].split('_')[7] + '_' + files[i].split('_')[9]
        files_dict_log[curr_dict_name_log].append(files_log[i])
        files_dict[curr_dict_name].append(files[i])

    cen = SkyCoord(ra_cen, dec_cen, frame = 'icrs', unit = (u.deg, u.deg))
    scale = cosmo.kpc_proper_per_arcmin(0.0221).to(u.kpc/u.deg)
    pixsize = 2 * u.kpc

    #ra_sorted = ra
    #dec_sorted = dec

    #ra_sorted = ra
    #dec_sorted = dec

    coords = SkyCoord(ra * u.deg, dec * u.deg)


    #rrm_sorted = rrm
    #rrm_err_sorted = rrm_err

    r = (cen.separation(coords).to(u.arcmin)* scale.to(u.Mpc/u.arcmin)).value


    RM_cube = np.zeros((2048, 2048, 10))
    RM_cube_log = np.zeros((2048, 2048, 10))

    ra_pix, dec_pix = world_to_pix(ra, dec, ra_cen, dec_cen, scale, pixsize, 2048)


    ra_in = []
    r_in = []
    complexity_in = []

    dec_in = []

    ra_pix_in = []
    dec_pix_in = []

    for i in range(len(ra_pix)):
        # This ensures that we only use points that are inside the simulation box
        if 0 <= ra_pix[i] <= 2047 and 0 <= dec_pix[i] <= 2047:
            r_in.append(r[i])
            ra_in.append(ra[i])
            dec_in.append(dec[i])
            ra_pix_in.append(ra_pix[i])
            dec_pix_in.append(dec_pix[i])
            complexity_in.append(complexity[i])

    sources = SkyCoord(ra_in, dec_in, frame ='icrs',  unit = (u.deg, u.deg))
    
    beamsize = (scale.to(u.kpc/u.arcsec) * 10 * u.arcsec ) / (2 * u.kpc) # setting POSSUM beam size of 20 arcsecond for averaging 
    spectra = Table.read('./GOOD_Data/spectra_A3581.fits')
    freq = spectra['freq']
    curr_dict_name = 'B0=5.0_eta=0.25'
    for i in range(len(files_dict[curr_dict_name])):
        RM_cube[:, :, i] = np.load(files_dict[curr_dict_name][i]) / (1+0.0221)**2
        RM_cube_log[:, :, i] = np.load(files_dict_log[curr_dict_name][i]) / (1+0.0221)**2

        #RM_mean = np.mean(RM_cube, axis = 2)
        #RM_std = np.std(RM_cube, axis = 2)

    x_running_sim_array = []
    scatter_sim_array = []
    scatter_sim_array_log = []
    for i in range(RM_cube.shape[-1]):
        rm_sim_in_log = []
        rm_sim_in = []
        for j in range(len(ra_pix_in)):
            if complexity_in[j] == 0:
                rm_sim_in.append(RM_cube[ra_pix_in[j], dec_pix_in[j], i]) # No averaging for Faraday simple sources 
                rm_sim_in_log.append(RM_cube_log[ra_pix_in[j], dec_pix_in[j], i])
            else:
                rm_complex = complex_averaging(ra_pix_in[j], dec_pix_in[j], RM_cube[:, :, i], beamsize, pol_frac_mean, I0_mean, alpha_mean, freq[0])
                rm_complex_log = complex_averaging(ra_pix_in[j], dec_pix_in[j], RM_cube_log[:, :, i], beamsize, pol_frac_mean, I0_mean, alpha_mean, freq[0])
                rm_sim_in.append(rm_complex)
                rm_sim_in_log.append(rm_complex_log)
            #rm_err_sim_in.append(RM_std[ra_pix_in[j], dec_pix_in[j]])
            #rm_sim_in.append(RM_cube[ra_pix_in[i], dec_pix_in[i], 0])
            #rm_err_sim_in.append(RM_cube[ra_pix_in[i], dec_pix_in[i], 0])
        
        rm_err_sim_in = np.zeros(len(rm_sim_in))   
        data_sim = Table()
        data_sim_new = Table()
        data_sim['RM'] = rm_sim_in
        data_sim['RM_log'] = rm_sim_in_log
        data_sim['e_RM'] = rm_err_sim_in
        data_sim['r'] = r_in
        
        x_running_sim_in, scatter_sim_in, scatter_sim_errlow_in, scatter_sim_errup_in = running_bins.calc_running_scatter(data_sim
                                                                , RMcol='RM' # column containing RRM
                                                                , xcol='r' # column containing distance to cluster
                                                                , RMerrcol = 'e_RM' # column containing RRM error
                                                                # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                                , xwidth=None # if we want set number of points
                                                                , M = 20     # number of points in sliding window
                                                                , method='iqr' # IQR is more robust than STD
                                                                , show=False
                                                                , plotwidth=False
                                                                , sigmaRMextr = 0
                                                                , plot_scatter=False)

        x_running_sim_in_log, scatter_sim_in_log, scatter_sim_errlow_in_log, scatter_sim_errup_in_log = running_bins.calc_running_scatter(data_sim
                                                                , RMcol='RM_log' # column containing RRM
                                                                , xcol='r' # column containing distance to cluster
                                                                , RMerrcol = 'e_RM' # column containing RRM error
                                                                # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                                , xwidth=None # if we want set number of points
                                                                , M = 20     # number of points in sliding window
                                                                , method='iqr' # IQR is more robust than STD
                                                                , show=False
                                                                , plotwidth=False
                                                                , sigmaRMextr = 0
                                                                , plot_scatter=False)
        
        # change this value to what scatter is far outside clusters
                                                                        # corrected for measurement errors!
        '''
        plt.show()
                                        
        plt.figure(facecolor='white')
        plt.fill_between(x_running_sim_in , scatter_sim_in - scatter_sim_errlow_in, scatter_sim_in + scatter_sim_errup_in,alpha=0.5, color='#1f77b4')
        plt.plot(x_running_sim_in, scatter_sim_in, color='#1f77b4', label = 'Observed RM')
        plt.ylabel(r"$\sigma_\mathrm{RM,cor}$ [rad m$^{-2}$]")
        plt.xlabel('$r$ (Mpc)')
        plt.show()
        '''
    
        x_running_sim_array.append(x_running_sim_in)
        scatter_sim_array.append(scatter_sim_in)
        scatter_sim_array_log.append(scatter_sim_in_log)

    plt.figure(facecolor='white')
    plt.fill_between(x_running_sim_in_log, np.median(scatter_sim_array_log, axis = 0) - np.subtract(*np.percentile(scatter_sim_array_log, [75, 25], axis = 0)) / 1.349, np.median(scatter_sim_array_log, axis = 0) + np.subtract(*np.percentile(scatter_sim_array_log, [75, 25], axis = 0)) / 1.349,alpha=0.1, color='#1f77b4')
    plt.plot(x_running_sim_in_log, np.median(scatter_sim_array_log, axis = 0), color='#1f77b4', label = 'Lognormal $n_e$ fluctuations')

    plt.ylabel("$\sigma_\mathrm{RRM,corr}$ (rad m$^{-2}$)")
    plt.xlabel('$r$ (Mpc)')
    plt.plot(x_running_sim_in, np.median(scatter_sim_array, axis = 0), color = 'tomato', label = 'Mean $n_e$')
    
    plt.fill_between(x_running_sim_in, np.median(scatter_sim_array, axis = 0) - np.subtract(*np.percentile(scatter_sim_array, [75, 25], axis = 0)) / 1.349, np.median(scatter_sim_array, axis = 0) + np.subtract(*np.percentile(scatter_sim_array, [75, 25], axis = 0)) / 1.349, color = 'tomato', alpha = 0.1)
    #plt.fill_between(x_running_sim_in,  np.percentile(scatter_sim_array, 16, axis = 0), np.percentile(scatter_sim_array, 84, axis = 0), color = 'tomato', alpha = 0.5)
    plt.legend()
    plot_name = '$B_0  = 5.0\ \mu$G $\eta = 0.25$'
    plt.title(plot_name)
    plt.savefig('./GOOD_Figures/model_comparison.pdf', dpi =300, bbox_inches = 'tight')



def bhattacharyya_coefficient(mu1, sigma1, mu2, sigma2):
    """
    Calculates the Bhattacharyya coefficient for two normal distributions.
    """
    return np.sqrt((2 * sigma1 * sigma2) / (sigma1**2 + sigma2**2)) * \
           np.exp(- (mu1 - mu2)**2 / (4 * (sigma1**2 + sigma2**2)))

def overlap_score(x, y1, y1_errlow, y1_errup, y2, y2_errlow, y2_errup):
    ol_metric = 0
    ol_array = []
    for i in range(len(x)):
        scatter_space = np.linspace(-500, 500, 1000)
        ol = bhattacharyya_coefficient(y1[i], np.mean([y1_errlow[i], y1_errup[i]]), y2[i], np.mean([y2_errlow[i], y2_errup[i]]))
        ol_metric += ol
        ol_array.append(ol)
    return ol_metric, np.array(ol_array)

def rescale_scatter(scatter_sim_in, B0_ref, B0_new):
    """
    Rescale simulated scatter from reference B0 to a new trial B0.
    Assuming scatter ∝ B0.
    """
    factor = B0_new / B0_ref
    return scatter_sim_in * factor

def mcmc(ra, dec, rrm, rrm_err, complexity, mod_file_name, pol_frac_mean = 0.06545418353102181, I0_mean = 0.007457496080425132, alpha_mean = -0.7678407583726873):
    '''
    Function to find a best-fit B0 using mcmc for the scatter plots (using a reference of B0 = 0.5 muG)

    Parameters
    -----------
    ra: numpy array object 
        Array of RAs
    dec: numpy array objec
        Array of DECs
    rrm: numpy array object 
        Array of RRMs
    rrm_err: numpy array object 
        Array of RRM_err
    complexity: numpy array object
        Array of flags for complex sources (used to model beam depolarization by averaging Stokes Q and U)
    mod_file_name: str 
        Path to file with the models
    pol_frac_mean: float
        mean polarisation fraction for modelling Stokes Q and U for complex sources. Default to median values found in A3581 
    I0_mean: float
        mean Stokes I intensity at 800 MHz for modelling Stokes Q and U for complex sources. Default to median values found in A3581 
    alpha_mean: float
        mean Stokes I spectral index for modelling Stokes Q and U for complex sources. Default to median values found in A3581 
    

    Returns 
    --------
    x_running_sim_array: numpy array object
        Array of the radius values for the scatter (in Mpc)
    scatter_sim_array: numpy array object
        Array of the scatter value for each radial bin (in rad/m^2)
    '''

    files = glob(mod_file_name)

    # Storing all the file names in a dcitionary for easy access 

    files_dict = {}

    dict_names = []

    B0_names = ['1.0'] 
    # Only one B0 (this is really 0.5 muG)

    eta_names = ['0.00', '0.25', '0.50']


    for i in range(len(B0_names)):
        for j in range(len(eta_names)): 
            dict_names.append('B0={:s}'.format(B0_names[i]) + '_eta={:s}'.format(eta_names[j]))
            files_dict['B0={:s}'.format(B0_names[i]) + '_eta={:s}'.format(eta_names[j])] = [] # Initializing the dictionary\
    # Placing the files in the relevant dictionary entry
    for i in range(len(files)):
        curr_dict_name = files[i].split('_')[8]+ '_'+ files[i].split('_')[10]
        if files[i].split('_')[8] == 'B0=1.0':
            files_dict[curr_dict_name].append(files[i])
    plt.rcParams.update({'font.size': 16})
    # Defining a function to calculate the length of the overlap between two intervals 
    cen = SkyCoord(ra_cen, dec_cen, frame = 'icrs', unit = (u.deg, u.deg))
    scale = cosmo.kpc_proper_per_arcmin(0.0221).to(u.kpc/u.deg)
    pixsize = 2 * u.kpc

    #ra_sorted = ra
    #dec_sorted = dec

    #ra_sorted = ra
    #dec_sorted = dec

    coords = SkyCoord(ra * u.deg, dec * u.deg)


    #rrm_sorted = rrm
    #rrm_err_sorted = rrm_err

    r = (cen.separation(coords).to(u.arcmin)* scale.to(u.Mpc/u.arcmin)).value


    RM_cube = np.zeros((2048, 2048, 10))

    ra_pix, dec_pix = world_to_pix(ra, dec, ra_cen, dec_cen, scale, pixsize, 2048)

    rm_in = []

    rm_err_in = []
    ra_in = []
    r_in = []
    complexity_in = []

    dec_in = []

    ra_pix_in = []
    dec_pix_in = []

    for i in range(len(ra_pix)):
        # This ensures that we only use points that are inside the simulation box
        if 0 <= ra_pix[i] <= 2047 and 0 <= dec_pix[i] <= 2047:
            rm_in.append(rrm[i])
            rm_err_in.append(rrm_err[i])
            r_in.append(r[i])
            ra_in.append(ra[i])
            dec_in.append(dec[i])
            ra_pix_in.append(ra_pix[i])
            dec_pix_in.append(dec_pix[i])
            complexity_in.append(complexity[i])

    sources = SkyCoord(ra_in, dec_in, frame ='icrs',  unit = (u.deg, u.deg))



    separation_in = cen.separation(sources)

    r_in  = separation_in.to(u.arcmin) * scale.to(u.Mpc/u.arcmin)
    data_obs = Table()
    data_obs['r_sorted'] = r_in
    data_obs['RM'] = rm_in 
    data_obs['RM_err'] = rm_err_in


    x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in = running_bins.calc_running_scatter(data_obs
                                                            , RMcol='RM' # column containing RRM
                                                            , xcol='r_sorted' # column containing distance to cluster
                                                            , RMerrcol = 'RM_err' # column containing RRM error
                                                            # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                            , xwidth=None # if we want set number of points
                                                            , M = 10     # number of points in sliding window
                                                            , method='iqr' # IQR is more robust than STD
                                                            , show=True
                                                            , sigmaRMextr = 6.99 # change this value to what scatter is far outside clusters
                                                                            # corrected for measurement errors!
                                                            , plot_scatter = False
                                                            , plotwidth  =False)
    

    overlap_score_array = []
    beamsize = (scale.to(u.kpc/u.arcsec) * 10 * u.arcsec ) / (2 * u.kpc) # setting POSSUM beam size of 20 arcsecond for averaging 
    spectra = Table.read('./GOOD_Data/spectra_A3581.fits')
    freq = spectra['freq']
    ol_self, ol_self_array = overlap_score(x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in, scatter_in, scatter_errlow_in, scatter_errup_in)
    ol_self_int = np.sum(ol_self_array[np.argwhere(x_running_in < 0.75)])

    #plt.figure(facecolor='white')
    
    colo = ['k', 'tomato']
    median_array = [0.16, 0.59, 2.20]
    for k in range(len(dict_names)):
        for i in range(len(files_dict[dict_names[k]])):
            RM_cube[:, :, i] = np.load(files_dict[dict_names[k]][i]) / (1+0.0221)**2

        #RM_mean = np.mean(RM_cube, axis = 2)
        #RM_std = np.std(RM_cube, axis = 2)

        x_running_sim_array = []
        scatter_sim_array = []

        for i in range(RM_cube.shape[-1]):
            rm_sim_in = []
            for j in range(len(ra_pix_in)):
                #Removing these to increase efficiency
                rm_sim_in.append(RM_cube[ra_pix_in[j], dec_pix_in[j], i]) # For simplicity. Remove this when complex averaging 
                
                #if complexity_in[j] == 0:
                #    rm_sim_in.append(RM_cube[ra_pix_in[j], dec_pix_in[j], i]) # No averaging for Faraday simple sources 
                #else:
                #    rm_complex = complex_averaging(ra_pix_in[j], dec_pix_in[j], RM_cube[:, :, i], beamsize, pol_frac_mean, I0_mean, alpha_mean, freq[0])
                #    rm_sim_in.append(rm_complex)
                #rm_err_sim_in.append(RM_std[ra_pix_in[j], dec_pix_in[j]])
                #rm_sim_in.append(RM_cube[ra_pix_in[i], dec_pix_in[i], 0])
                #rm_err_sim_in.append(RM_cube[ra_pix_in[i], dec_pix_in[i], 0])
            
            rm_err_sim_in = np.zeros(len(rm_sim_in))    
            data_sim = Table()
            data_sim_new = Table()
            data_sim['RM'] = rm_sim_in
            data_sim['e_RM'] = rm_err_sim_in
            data_sim['r'] = r_in
           
            x_running_sim_in, scatter_sim_in, scatter_sim_errlow_in, scatter_sim_errup_in = running_bins.calc_running_scatter(data_sim
                                                                    , RMcol='RM' # column containing RRM
                                                                    , xcol='r' # column containing distance to cluster
                                                                    , RMerrcol = 'e_RM' # column containing RRM error
                                                                    # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                                    , xwidth=None # if we want set number of points
                                                                    , M = 10     # number of points in sliding window
                                                                    , method='iqr' # IQR is more robust than STD
                                                                    , show=False
                                                                    , plotwidth=False
                                                                    , sigmaRMextr = 0
                                                                    , plot_scatter=False)
            B0_curr_name = dict_names[k].split('_')[0][3:]
            eta_curr_name = dict_names[k].split('_')[1][4:]
            # The data for this is with B0 = 0.5 muG, so we rescale to explore B0 = 1 instead
            # change this value to what scatter is far outside clusters
                                                                            # corrected for measurement errors
        
            x_running_sim_array.append(x_running_sim_in)
            scatter_sim_array.append(scatter_sim_in)
    
        ndim = 1  # only fitting B0
        nwalkers = 20
        initial_pos = np.log(median_array[k]) + 1e-2 * np.random.randn(nwalkers, ndim)  # start near B0=5 μG

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_logprior,
                                        args=(x_running_in, scatter_in, scatter_errlow_in,
                                            scatter_errup_in, scatter_sim_array, ol_self_int))

        sampler.run_mcmc(initial_pos, 5000, progress=True)

        log_samples = sampler.get_chain(discard=1000, thin=10, flat=True)
        B0_samples = np.exp(log_samples)  # transform back to μG
        median = np.median(B0_samples)
        lower, upper = np.percentile(B0_samples, [16, 84])  # 68% central interval
        err_low = median - lower
        err_high = upper - median

        print(f"B0 = {median:.2f} (+{err_high:.2f}, -{err_low:.2f}) μG for eta = {eta_curr_name}")

        # Corner plot
        figure = corner.corner(
            B0_samples,
            labels=[r"$B_0\ [\mu G]$"],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12}
        )

        plt.show()
        plt.close()

        scatter_sim_rescaled = rescale_scatter(np.vstack(scatter_sim_array), B0_ref=0.5, B0_new=median)

        plt.ylabel("$\sigma_\mathrm{RRM,corr}$ (rad m$^{-2}$)")
        plt.xlabel('$r$ (Mpc)')
       
        plt.fill_between(x_running_in , scatter_in - scatter_errlow_in, scatter_in + scatter_errup_in,alpha=0.5, color='#1f77b4')
        plt.plot(x_running_in, scatter_in, color='#1f77b4', label = 'A3581')    

        plt.fill_between(x_running_sim_in, np.median(scatter_sim_rescaled, axis = 0) - np.subtract(*np.percentile(scatter_sim_rescaled, [75, 25], axis = 0)) / 1.349, np.median(scatter_sim_rescaled, axis = 0) + np.subtract(*np.percentile(scatter_sim_rescaled, [75, 25], axis = 0)) / 1.349, color = 'tomato', alpha = 0.5)
        #plt.fill_between(x_running_sim_in,  np.percentile(scatter_sim_array, 16, axis = 0), np.percentile(scatter_sim_array, 84, axis = 0), color = 'tomato', alpha = 0.5)
        plt.plot(x_running_sim_in, np.median(scatter_sim_rescaled, axis = 0), color = 'tomato', label = 'Model')
        if eta_curr_name  == '0.2':
            eta_curr_name = '0.25'
        B0_curr_name =f"{median:.2f}"
        plot_name = '$ B_0  = {}\ \mu$G'.format(B0_curr_name) + ' $\eta = {}$'.format(eta_curr_name)
        plt.title(plot_name)
        plt.xlim(0.1, 0.9)
        plt.legend()
        plt.axvline(0.75, c = 'red', ls = '--')
        #plt.savefig('./GOOD_Figures/ne_lognormal_fluct_' + dict_names[k] +'.pdf', dpi = 300, bbox_inches = 'tight')
        plt.show()
        plt.close()
        plt.scatter(r_in, rescale_scatter(np.array(rm_sim_in), B0_ref=0.5, B0_new=median_array[k]))
        np.save('./GOOD_Figures/ne_lognormal_fluct_' + dict_names[k] +'.npy', [r_in, rescale_scatter(np.array(rm_sim_in), B0_ref=0.5, B0_new=median_array[k])])
        plt.title(plot_name)
        plt.ylabel("RM (rad m$^{-2}$)")
        plt.xlabel('$r$ (Mpc)')
        plt.show()
        plt.close()

        ol_metric, ol_array = overlap_score(x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in, np.median(scatter_sim_rescaled, axis = 0), np.subtract(*np.percentile(scatter_sim_rescaled, [75, 25], axis = 0)) / 1.349, np.subtract(*np.percentile(scatter_sim_rescaled, [75, 25], axis = 0)) / 1.349)
        ol_self, ol_self_array = overlap_score(x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in, scatter_in, scatter_errlow_in, scatter_errup_in)
        overlap_score_array.append(1- ol_metric/ol_self)


        plt.plot(x_running_in, np.array(ol_array)/np.array(ol_self_array))
        ol_metric_int = np.sum(ol_array[np.argwhere(x_running_in < 0.75)])
        ol_self_int = np.sum(ol_self_array[np.argwhere(x_running_in < 0.75)])
        ol_metric_ext = np.sum(ol_array[np.argwhere(x_running_in > 1.1)])
        ol_self_ext  = np.sum(ol_self_array[np.argwhere(x_running_in > 1.1)])
        plt.title('Interior  = {:.3f}'.format(1- ol_metric_int/ol_self_int) + ' Exterior  = {:.3f}'.format(1- ol_metric_ext/ol_self_ext))
        plt.savefig('./GOOD_Figures/ne_lognormal_fluct_' + dict_names[k] + '_overlap.png', bbox_inches = 'tight')
        plt.close()




def log_likelihood_logprior(log_B0, x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in,
                   scatter_sim_array, ol_self_int,
                   logB0_prior_mu=np.log(1.0), logB0_prior_sigma=1.0, B0_min=1e-6, B0_max=1e3, sigma=0.1):
    """
    Log-likelihood with a Gaussian prior on log_B0 (weak), plus bounds.
    """
    if not np.isfinite(log_B0):
        return -np.inf
    if log_B0 < np.log(B0_min) or log_B0 > np.log(B0_max):
        return -np.inf

    B0 = np.exp(log_B0)

    try:
        scatter_sim_rescaled = rescale_scatter(
            np.vstack(scatter_sim_array), B0_ref=0.5, B0_new=B0
        )
        if np.any(~np.isfinite(scatter_sim_rescaled)):
            return -np.inf

        median_sim = np.median(scatter_sim_rescaled, axis=0)
        q75, q25 = np.percentile(scatter_sim_rescaled, [75, 25], axis=0)
        iqr_sim = (q75 - q25) / 1.349
        iqr_sim = np.where(iqr_sim <= 0, 1e-8, iqr_sim)

        ol_metric, ol_array = overlap_score(
            x_running_in, scatter_in,
            scatter_errlow_in, scatter_errup_in,
            median_sim, iqr_sim, iqr_sim
        )

        if ol_self_int == 0 or not np.isfinite(ol_self_int):
            return -np.inf

        mask = (x_running_in < 0.75)
        ol_metric_int = np.sum(ol_array[mask])
        loss = 1.0 - (ol_metric_int / ol_self_int)
        if not np.isfinite(loss):
            return -np.inf

        log_like_data = -0.5 * (loss / sigma) ** 2

        # Gaussian prior on log_B0
        log_prior = -0.5 * ((log_B0 - logB0_prior_mu) / logB0_prior_sigma) ** 2
        return log_like_data + log_prior

    except Exception:
        return -np.inf


def model_plotting(ra, dec, rrm, rrm_err, complexity, mod_file_name, pol_frac_mean = 0.06545418353102181, I0_mean = 0.007457496080425132, alpha_mean = -0.7678407583726873):
    '''
    Function to create the comparison plots of the A3581 scatter profile and the modelled clusters 

    Parameters
    -----------
    ra: numpy array object 
        Array of RAs
    dec: numpy array objec
        Array of DECs
    rrm: numpy array object 
        Array of RRMs
    rrm_err: numpy array object 
        Array of RRM_err
    complexity: numpy array object
        Array of flags for complex sources (used to model beam depolarization by averaging Stokes Q and U)
    mod_file_name: str 
        Path to file with the models
    pol_frac_mean: float
        mean polarisation fraction for modelling Stokes Q and U for complex sources. Default to median values found in A3581 
    I0_mean: float
        mean Stokes I intensity at 800 MHz for modelling Stokes Q and U for complex sources. Default to median values found in A3581 
    alpha_mean: float
        mean Stokes I spectral index for modelling Stokes Q and U for complex sources. Default to median values found in A3581 
    

    Returns 
    --------
    None 
    '''
    files = glob(mod_file_name)

    # Storing all the file names in a dcitionary for easy access 

    files_dict = {}

    dict_names = []

    B0_names = ['1.0', '5.0', '10.0']

    eta_names = ['0.00', '0.25', '0.50']


    for i in range(len(B0_names)):
        for j in range(len(eta_names)): 
            dict_names.append('B0={:s}'.format(B0_names[i]) + '_eta={:s}'.format(eta_names[j]))
            files_dict['B0={:s}'.format(B0_names[i]) + '_eta={:s}'.format(eta_names[j])] = [] # Initializing the dictionary

    # Placing the files in the relevant dictionary entry
    for i in range(len(files)):
        curr_dict_name = files[i].split('_')[8]+ '_'+ files[i].split('_')[10]
        files_dict[curr_dict_name].append(files[i])

    plt.rcParams.update({'font.size': 16})

    # Defining a function to calculate the length of the overlap between two intervals 

    
        
    cen = SkyCoord(ra_cen, dec_cen, frame = 'icrs', unit = (u.deg, u.deg))
    scale = cosmo.kpc_proper_per_arcmin(0.0221).to(u.kpc/u.deg)
    pixsize = 2 * u.kpc

    #ra_sorted = ra
    #dec_sorted = dec

    #ra_sorted = ra
    #dec_sorted = dec

    coords = SkyCoord(ra * u.deg, dec * u.deg)


    #rrm_sorted = rrm
    #rrm_err_sorted = rrm_err

    r = (cen.separation(coords).to(u.arcmin)* scale.to(u.Mpc/u.arcmin)).value


    RM_cube = np.zeros((2048, 2048, 10))

    ra_pix, dec_pix = world_to_pix(ra, dec, ra_cen, dec_cen, scale, pixsize, 2048)

    rm_in = []

    rm_err_in = []
    ra_in = []
    r_in = []
    complexity_in = []

    dec_in = []

    ra_pix_in = []
    dec_pix_in = []

    for i in range(len(ra_pix)):
        # This ensures that we only use points that are inside the simulation box
        if 0 <= ra_pix[i] <= 2047 and 0 <= dec_pix[i] <= 2047:
            rm_in.append(rrm[i])
            rm_err_in.append(rrm_err[i])
            r_in.append(r[i])
            ra_in.append(ra[i])
            dec_in.append(dec[i])
            ra_pix_in.append(ra_pix[i])
            dec_pix_in.append(dec_pix[i])
            complexity_in.append(complexity[i])

    sources = SkyCoord(ra_in, dec_in, frame ='icrs',  unit = (u.deg, u.deg))



    separation_in = cen.separation(sources)

    r_in  = separation_in.to(u.arcmin) * scale.to(u.Mpc/u.arcmin)
    data_obs = Table()
    data_obs['r_sorted'] = r_in
    data_obs['RM'] = rm_in 
    data_obs['RM_err'] = rm_err_in


    x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in = running_bins.calc_running_scatter(data_obs
                                                            , RMcol='RM' # column containing RRM
                                                            , xcol='r_sorted' # column containing distance to cluster
                                                            , RMerrcol = 'RM_err' # column containing RRM error
                                                            # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                            , xwidth=None # if we want set number of points
                                                            , M =  20      # number of points in sliding window
                                                            , method='iqr' # IQR is more robust than STD
                                                            , show=True
                                                            , sigmaRMextr = 6.99 # change this value to what scatter is far outside clusters
                                                                            # corrected for measurement errors!
                                                            , plot_scatter = False
                                                            , plotwidth  =False)

    overlap_score_array = []
    beamsize = (scale.to(u.kpc/u.arcsec) * 10 * u.arcsec ) / (2 * u.kpc) # setting POSSUM beam size of 20 arcsecond for averaging 
    spectra = Table.read('./GOOD_Data/spectra_A3581.fits')
    freq = spectra['freq']
 

    #plt.figure(facecolor='white')
    
    colo = ['k', 'tomato']
    for k in range(len(dict_names)):
        for i in range(len(files_dict[dict_names[k]])):
            RM_cube[:, :, i] = np.load(files_dict[dict_names[k]][i]) / (1+0.0221)**2

        #RM_mean = np.mean(RM_cube, axis = 2)
        #RM_std = np.std(RM_cube, axis = 2)

        x_running_sim_array = []
        scatter_sim_array = []

        for i in range(RM_cube.shape[-1]):
            rm_sim_in = []
            for j in range(len(ra_pix_in)):
                #Removing these to increase efficiency
                if complexity_in[j] == 0:
                    rm_sim_in.append(RM_cube[ra_pix_in[j], dec_pix_in[j], i]) # No averaging for Faraday simple sources 
                else:
                    rm_complex = complex_averaging(ra_pix_in[j], dec_pix_in[j], RM_cube[:, :, i], beamsize, pol_frac_mean, I0_mean, alpha_mean, freq[0])
                    rm_sim_in.append(rm_complex)
            
                #rm_err_sim_in.append(RM_std[ra_pix_in[j], dec_pix_in[j]])
                #rm_sim_in.append(RM_cube[ra_pix_in[i], dec_pix_in[i], 0])
                #rm_err_sim_in.append(RM_cube[ra_pix_in[i], dec_pix_in[i], 0])
            
            rm_err_sim_in = np.zeros(len(rm_sim_in))    
            data_sim = Table()
            data_sim_new = Table()
            data_sim['RM'] = rm_sim_in
            data_sim['e_RM'] = rm_err_sim_in
            data_sim['r'] = r_in
           
            x_running_sim_in, scatter_sim_in, scatter_sim_errlow_in, scatter_sim_errup_in = running_bins.calc_running_scatter(data_sim
                                                                    , RMcol='RM' # column containing RRM
                                                                    , xcol='r' # column containing distance to cluster
                                                                    , RMerrcol = 'e_RM' # column containing RRM error
                                                                    # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                                    , xwidth=None # if we want set number of points
                                                                    , M = 20     # number of points in sliding window
                                                                    , method='iqr' # IQR is more robust than STD
                                                                    , show=False
                                                                    , plotwidth=False
                                                                    , sigmaRMextr = 0
                                                                    , plot_scatter=False)
            B0_curr_name = dict_names[k].split('_')[0][3:]
            if B0_curr_name == '1.0':
                scatter_sim_in = scatter_sim_in * 2 # The data for this is with B0 = 0.5 muG, so we rescale to explore B0 = 1 instead
            # change this value to what scatter is far outside clusters
                                                                            # corrected for measurement errors!
            '''
            plt.show()
                                            
            plt.figure(facecolor='white')
            plt.fill_between(x_running_sim_in , scatter_sim_in - scatter_sim_errlow_in, scatter_sim_in + scatter_sim_errup_in,alpha=0.5, color='#1f77b4')
            plt.plot(x_running_sim_in, scatter_sim_in, color='#1f77b4', label = 'Observed RM')
            plt.ylabel(r"$\sigma_\mathrm{RM,cor}$ [rad m$^{-2}$]")
            plt.xlabel('$r$ (Mpc)')
            plt.show()
            '''
        
            x_running_sim_array.append(x_running_sim_in)
            scatter_sim_array.append(scatter_sim_in)

        
        plt.ylabel("$\sigma_\mathrm{RRM,corr}$ (rad m$^{-2}$)")
        plt.xlabel('$r$ (Mpc)')
       
        plt.fill_between(x_running_in , scatter_in - scatter_errlow_in, scatter_in + scatter_errup_in,alpha=0.5, color='#1f77b4')
        plt.plot(x_running_in, scatter_in, color='#1f77b4', label = 'A3581')
        
        plt.fill_between(x_running_sim_in, np.median(scatter_sim_array, axis = 0) - np.subtract(*np.percentile(scatter_sim_array, [75, 25], axis = 0)) / 1.349, np.median(scatter_sim_array, axis = 0) + np.subtract(*np.percentile(scatter_sim_array, [75, 25], axis = 0)) / 1.349, color = 'tomato', alpha = 0.5)
        #plt.fill_between(x_running_sim_in,  np.percentile(scatter_sim_array, 16, axis = 0), np.percentile(scatter_sim_array, 84, axis = 0), color = 'tomato', alpha = 0.5)
        plt.plot(x_running_sim_in, np.median(scatter_sim_array, axis = 0), color = 'tomato', label = 'Model')
        eta_curr_name = dict_names[k].split('_')[1][4:]
        B0_curr_name = dict_names[k].split('_')[0][3:]
        if eta_curr_name  == '0.2':
            eta_curr_name = '0.25'
        if B0_curr_name == '1.0':
            B0_curr_name = '1.0'
        elif B0_curr_name == '5.0':
            B0_curr_name = '2.5'
        else:
            B0_curr_name = '5.0'
        plot_name = '$ B_0  = {}\ \mu$G'.format(B0_curr_name) + ' $\eta = {}$'.format(eta_curr_name)
        
        plt.title(plot_name)
        plt.xlim(0.1, 0.9)
        plt.legend()
        plt.axvline(0.75, c = 'red', ls = '--')
        plt.savefig('./GOOD_Figures/ne_lognormal_fluct_' + dict_names[k] +'.pdf', dpi = 300, bbox_inches = 'tight')
        plt.close()
        ol_metric, ol_array = overlap_score(x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in, np.median(scatter_sim_array, axis = 0), np.subtract(*np.percentile(scatter_sim_array, [75, 25], axis = 0)) / 1.349, np.subtract(*np.percentile(scatter_sim_array, [75, 25], axis = 0)) / 1.349)
        ol_self, ol_self_array = overlap_score(x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in, scatter_in, scatter_errlow_in, scatter_errup_in)
        overlap_score_array.append(1- ol_metric/ol_self)


        plt.plot(x_running_in, np.array(ol_array)/np.array(ol_self_array))
        ol_metric_int = np.sum(ol_array[np.argwhere(x_running_in < 0.75)])
        ol_self_int = np.sum(ol_self_array[np.argwhere(x_running_in < 0.75)])
        ol_metric_ext = np.sum(ol_array[np.argwhere(x_running_in > 1.1)])
        ol_self_ext  = np.sum(ol_self_array[np.argwhere(x_running_in > 1.1)])
        plt.title('Interior  = {:.3f}'.format(1- ol_metric_int/ol_self_int) + ' Exterior  = {:.3f}'.format(1- ol_metric_ext/ol_self_ext))
        plt.savefig('./GOOD_Figures/ne_lognormal_fluct_' + dict_names[k] + '_overlap.png', bbox_inches = 'tight')
        plt.close()
    #plt.legend()
    #plt.ylim(0, 30)
    #plt.xlim(0, 2)
    #plt.savefig('./GOOD_Figures/poster_model.pdf', dpi =300, bbox_inches = 'tight')
    '''
    B0_array = [1,1, 1, 5, 5, 5, 10, 10, 10]
    eta_array = [0, 0.2, 0.5, 0, 0.25, 0.5, 0, 0.2, 0.5]
    plt.figure(figsize = (5.84685039, 4.46338583))
    B0_vals = np.array([0, 0.25, 0.5])
    eta_vals = np.array([1, 5, 10])

    y_edges =np.concatenate([
        [eta_vals[0] - (eta_vals[1] - eta_vals[0]) / 2],  # Left edge
        (eta_vals[:-1] + eta_vals[1:]) / 2,                   # Midpoints
        [eta_vals[-1] + (eta_vals[-1] - eta_vals[-2]) / 2]])

    x_edges = np.concatenate([
        [B0_vals[0] - (B0_vals[1] - B0_vals[0]) / 2],
        (B0_vals[:-1] + B0_vals[1:]) / 2,
        [B0_vals[-1] + (B0_vals[-1] - B0_vals[-2]) / 2]
    ])

    plt.hist2d(eta_array, B0_array, weights=overlap_score_array, bins=[x_edges, y_edges], cmap='Oranges', vmin = 0.3, vmax = 1)

    # Add labels and title
    plt.xlabel('$\eta$')
    plt.ylabel(' $B_0$ ($\mu$G)')
    plt.xticks(B0_vals)
    plt.yticks(eta_vals)
    plt.colorbar(label='$\Phi$')
    plt.savefig('./GOOD_Figures/lognormal_overlap.pdf', dpi = 300, bbox_inches = 'tight')
    '''

def peak_presence(x, rm_scat, loc, width):
    '''
    Function for finding a peak in an RM scatter profile
    
    Parameters
    ----------
    x: numpy array object
        Array of radii plotted
    rm_scat: numpy array object 
        Array of RM scatters 
    loc: float
        Location of peak to be found (in Mpc)
    width: float 
        Width of the peak to be found (in Mpc). This is the full width
    
    Returns
    -------
    peak_pres: int
        1 or 0 to indicate if a peak is present or not --- 1 if present, 0 if not present 
    peaks: int 
        The index of the peak
    '''

    #width_pix = np.min(np.argwhere(x>loc + width/2)) -  np.max(np.argwhere(x < loc - width/2)) 

    peaks, _ = scipy.signal.find_peaks(rm_scat, width  = len(x)/5)
    if len(peaks) > 0:
        for i in range(len(peaks)):
            if loc - width/2  < x[peaks[i]] < loc + width/2:
                peak_pres = 1 
            else:
                peak_pres = 0
    else:
        peak_pres = 0
        peaks = []
    return peak_pres, peaks


def peak_presence(radii, rm_scatter, smooth_sigma=1, min_rel_increase=0.5):
    '''
    Function to find the presence of re-enhancement (including peaks)
    Parameters
    ----------
        radii (array): shape (N,) - radial bins.
        rm_scatter (array): shape (B, N) - RM scatter values for B bootstrap realizations.
        smooth_sigma (float): smoothing applied to suppress noise.
        min_rel_increase (float): minimum relative increase after the minimum to count as re-enhancement.

    Returns
    --------
        reenhanced_flags (array): shape (B,), boolean array whether each sample has re-enhancement.
        fraction (float): fraction of samples with re-enhancement.
    '''
    smoothed = gaussian_filter1d(rm_scatter, sigma=smooth_sigma)
   
    min_idx = np.argmin(smoothed)
    min_val = smoothed[min_idx]
    reenhanced_flags = 0
    peaks,_ = scipy.signal.find_peaks(smoothed, prominence = (5, 20))
    # Check for significant increase after the minimum
    if min_idx < len(rm_scatter) - 1:
        post_min_vals = smoothed[min_idx+1:]
        if np.any(post_min_vals > min_val * (1 + min_rel_increase)) or len(peaks > 0):
            reenhanced_flags = 1
    '''
    print(reenhanced_flags)
    plt.scatter(radii[peaks], smoothed[peaks])
    plt.plot(radii, smoothed)
    plt.show()
    plt.close()
    '''

    return reenhanced_flags

def clump_plotting(catalog):
    x_data = fits.open('./GOOD_Data/em01_212117_024_Image_c010.fits')
    xim = x_data[0].data
    interval = ZScaleInterval()

    vmin, vmax = interval.get_limits(xim)

    wcs_x  = WCS(x_data[0].header).celestial
    fig, ax = plt.subplots(ncols=1, subplot_kw={'projection': wcs_x}, figsize = (10, 5))
    fig.patch.set_facecolor('white')

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.family'] = 'Courier'
    
    ax.imshow(xim, cmap  ='binary', origin = 'lower', vmin = 0 , vmax = 0.2)
    colour_nnew = []
   


    ra_nnew = np.ravel(catalog['ra'].data)
    dec_nnew = np.ravel(catalog['dec'].data)
    rrm = np.ravel(catalog['rm_corrected'].data)

    r_plotting = SphericalCircle((ra_cen * u.degree, dec_cen * u.degree),  (2* 0.925 * (u.Mpc) / scale).to(u.deg), linestyle = '-', facecolor = 'none', edgecolor = 'k', transform = ax.get_transform('icrs'),  label = '$2R_{500}$')
    ax.add_patch(r_plotting)
    
    r_plotting = SphericalCircle((ra_cen * u.degree, dec_cen * u.degree),  (1.1* (u.Mpc) / scale).to(u.deg), linestyle = '--', facecolor = 'none', edgecolor = 'k', transform = ax.get_transform('icrs'), label = '1.1 Mpc')
    ax.add_patch(r_plotting)

    #ax.scatter(ra_nnew, dec_nnew, c= 'k', s = 1, transform = ax.get_transform('icrs'))

    for idx,fd in enumerate(rrm):
        ra_t = ra_nnew[idx]
        dec_t = dec_nnew[idx]
        
        if fd>=0:
            r = SphericalCircle((ra_t*u.degree, dec_t*u.degree), np.abs(fd)* 3e-3*u.degree, edgecolor='red', facecolor='none', transform=ax.get_transform('icrs'), ls ='--')
            #ax.scatter(ra_t * u.degree, dec_t * u.degree, c = 'blue', marker = 'X')
            ax.add_patch(r)
        else:
            r = SphericalCircle((ra_t*u.degree, dec_t*u.degree), np.abs(fd)* 3e-3*u.degree, edgecolor='blue', facecolor='none', transform=ax.get_transform('icrs'), ls ='--')
            
            ax.add_patch(r)

    #colorbar.set_label('$\Sigma_{\mathrm{RM}}$ (rad m$^{-2}$)')
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('DEC (J2000)')
    ax.set_aspect(1)
    #ax.scatter(212.4, -27.13, transform = ax.get_transform('icrs'), marker = '+', c = 'green', s = 500)
    #ax.scatter(212.342,	-27.102,  transform = ax.get_transform('icrs'), marker = '+', c = 'black', s = 500)
    #ax.scatter(212.33904,	-27.08787,  transform = ax.get_transform('icrs'), marker = '+', c = 'green', s = 500)
    scale_fat = cosmo.kpc_proper_per_arcmin(0.021356).to(u.Mpc/u.deg)

    
    r = SphericalCircle((212.342 * u.deg, -27.102*u.degree), 0.724 * 0.7 * u.Mpc /scale_fat, edgecolor='green', facecolor='none', transform=ax.get_transform('icrs'), ls ='-', label = '[DZ2015b] 276')
    
    scale_fat = cosmo.kpc_proper_per_arcmin(0.7416).to(u.Mpc/u.deg)
    ax.add_patch(r)
    
    r = SphericalCircle((212.33904 * u.deg, -27.08787*u.degree), 0.35 * 2 * u.Mpc /scale_fat, edgecolor='green', facecolor='none', transform=ax.get_transform('icrs'), ls ='--', label = 'Background cluster')
    #ax.add_patch(r)
    plt.legend()
    xlim = [min(ra_nnew) - 0.3, max(ra_nnew) + 0.3]*u.deg
    ylim = [min(dec_nnew) - 0.3, max(dec_nnew) + 0.3]*u.deg

    xlim, ylim = wcs_x.world_to_pixel(SkyCoord(xlim, ylim))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_xaxis()
    ax.set_aspect(1)
    plt.savefig('./GOOD_Figures/clump_erosita.pdf', dpi = 300, bbox_inches = 'tight')
   
    plt.show()

def peak_bootstrapping(r, rrm, rrm_err, samples = 100):
    '''
    Function for bootstrapping and detecting the significance of the presence of a peak in the RM scatter at 1.1 Mpc 
    
    Parameters
    ----------
    r: numpy array object 
        The radii 
    rrm: numpy array object 
        The RRM values 
    rrm_err: numpy array object 
        The error in the RRM values 
    samples: int 
        The number of samples taken. Default is 1000
    
    Returns 
    -------
    confidence_level: float 
        Fraction of samples that also present a peak in the RM scatter
    '''

    dictionary = {'r' : r, 'rrm': rrm, 'rrm_err':rrm_err}

    df = pd.DataFrame(data = dictionary)
    df = df.apply(lambda col: col.astype(col.dtype.newbyteorder('=')) if col.dtype.byteorder in ('>', '<') else col)

    peak_flags = []
    
    for i in range(samples):
        new_data = df.sample(n = len(r), replace = True)
        r_new, rrm_new, rrm_err_new = new_data['r'], new_data['rrm'], new_data['rrm_err']
        data_in_pol_frac = Table()
        data_in_pol_frac['r_sorted'] = np.array(r_new)
        data_in_pol_frac['RM'] = np.array(rrm_new)
        data_in_pol_frac['RM_err'] = np.array(rrm_err_new)
        # Do the calculation
        x_running_in, scatter_in, scatter_errlow_in, scatter_errup_in = running_bins.calc_running_scatter(data_in_pol_frac
                                                                , RMcol='RM' # column containing RRM
                                                                , xcol='r_sorted' # column containing distance to cluster
                                                                , RMerrcol = 'RM_err' # column containing RRM error
                                                                # , xwidth=0.3 # in units of whatever is 'xcol' in
                                                                , xwidth=None # if we want set number of points
                                                                , M =  20      # number of points in sliding window
                                                                , method='iqr' # IQR is more robust than STD
                                                                , show=False
                                                                , sigmaRMextr = 6.99 # change this value to what scatter is far inside clusters
                                                                                # corrected for measurement errors!
                                                                , plotwidth = False
                                                                , plot_scatter = False
                                                                )
        
        peak_pres = peak_presence(x_running_in, scatter_in)

        peak_flags.append(peak_pres)
    confidence_level = np.sum(peak_flags)/len(peak_flags)
    print(confidence_level)
    return confidence_level

if __name__ == "__main__":

    A3581_cat = Table.read('./GOOD_Data/A3581_cat.fits')
    in_clust = A3581_cat['in_clust']
    A3581_cat = A3581_cat[in_clust == 0]

    #clump_plotting(A3581_cat)
    ra = np.ravel(A3581_cat['ra'].data)
    dec = np.ravel(A3581_cat['dec'].data)
    complexity = np.ravel(A3581_cat['complexity'].data)
    coords = SkyCoord(ra*u.deg, dec * u.deg)
    r = (coords.separation(cen).to(u.arcmin) * scale).value

    rrm = np.ravel(A3581_cat['rm_corrected'].data)
    rrm_err = np.ravel(A3581_cat['rm_err_corrected'].data)

    

    mod_file_name = './Abell3581_RMimage_model_NO_FLUCT/*'
    mod_file_name_log = './RMimages_ne_fluct_B_fluct_indep/*'
    
    #scatter_plotting(r, rrm , rrm_err)
    #complexity_plotting(A3581_cat)
    #CGM_plotting('./GOOD_Data/A3581_cat.fits')
    #peak_bootstrapping(r, rrm, rrm_err, samples = 1000)
    mcmc(ra, dec, rrm, rrm_err, complexity, mod_file_name_log)
    #asymmetry_calc(ra, dec, rrm, wcs, step = 1, show_plots = True, return_sigmas = False)
    '''
    #complexity_checking 
    spectra = Table.read('./GOOD_Data/spectra_A3581.fits')
    freq = spectra['freq']
    #complex_averaging(ra_complex[0], dec_complex[0], rmimage, beamsize, 0.06545418353102181, 0.007457496080425132, -0.7678407583726873, freq[0])
    #model_plotting(ra, dec, rrm, rrm_err, complexity, mod_file_name)

    catalog_cleaned = Table.read('./GOOD_Data/catalog_cleaned.fits')
    m2_vals_cleaned = np.load('./GOOD_Data/m2_vals_cleaned.npy')
    models_cleaned = np.load('./GOOD_Data/models_cleaned.npy')
    redchi2_cleaned = np.load('./GOOD_Data/red_chi2_cleaned.npy')
    sigmaAddq_cleaned = catalog_cleaned['sigmaAddQ']
    sigmaAddq_cleaned = catalog_cleaned['sigmaAddU']

    sigmaAdd_cleaned = np.sqrt(sigmaAddq_cleaned**2 + sigmaAddq_cleaned**2)

    catalog_complexity = obtain_complexity(sigmaAdd_cleaned, m2_vals_cleaned, models_cleaned, redchi2_cleaned)
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    x1 = np.linspace(min(m2_vals_cleaned), max(m2_vals_cleaned))
    x2 = np.linspace(min(sigmaAdd_cleaned), max(sigmaAdd_cleaned))
    x3 = np.linspace(0, 1)
    for i in range(len(catalog_cleaned)):
        if catalog_cleaned['ra'][i] in ra:
            index = np.argwhere(ra == catalog_cleaned['ra'][i])

            ax1.scatter(A3581_cat['m2'][index], m2_vals_cleaned[i], c = 'k')
            ax2.scatter(A3581_cat['sigmaAdd'][index], sigmaAdd_cleaned[i], c = 'k')
            ax3.scatter(A3581_cat['complexity'][index], catalog_complexity[i], c = 'k')
            if 
            print(A3581_cat['QU_mdl'][index], models_cleaned[i])
    ax1.set_xlabel('Old $m_2$')
    ax1.set_ylabel('New $m_2$')

    ax1.plot(x1, x1)
    ax2.plot(x2, x2)
    ax3.plot(x3, x3)

    ax2.set_xlabel('Old $\sigma_{\mathrm{{add}}}$')
    ax2.set_ylabel('New $\sigma_{\mathrm{{add}}}$')

    ax3.set_xlabel('Old Complexity')
    ax3.set_ylabel('New Complexity')
    plt.show()
    '''

    '''
    for j in range(len(np.argwhere(catalog_complexity == 1))):
        for i in np.argwhere(catalog_complexity == 1)[j]:
            if catalog_cleaned['dec'][i] in dec:
                phi_fdf = FDFs['phi_fdf'][i]
                fdf = FDFs['fdf'][i]
                intensity = np.abs(fdf)
                plt.plot(phi_fdf - A3581_cat['rm_correction'][i], intensity, c = 'k')
                plt.axvline(A3581_cat['rm_corrected'][i], c = 'red', ls = '--', label = 'Likely still extragalactic?')
                plt.axvline(100, c = 'blue', ls = '--', label = 'Galacitc emission+rotation')
                plt.xlabel('$\phi$ (rad m$^{-2}$)')
                plt.ylabel('Intensity (mJy bm$^{-1}$)')
                plt.title(A3581_cat['QU_mdl'][i])
                plt.xlim(-500, 500)
                plt.show()
                plt.close()
    '''

    '''
    A3581_rms = catalog[r.value < 3 * 0.925]
    rrm = A3581_rms['rm_corrected']
    rrm_err = A3581_rms['rm_err_corrected']
    r = r.value[r.value < 3 * 0.925]
    
    z_phot, z_phot_err = np.full(len(A3581_cat), fill_value = -1, dtype= 'float64'), np.full(len(A3581_cat), fill_value = -1, dtype = 'float64')

    z_phot_opt, z_phot_err_opt = calc_phot_z('./GOOD_Data/optical_sources.csv')
    indices = ascii.read('./GOOD_Data/optical_sources.csv', format = 'no_header')['col3']
    for i in range(len(z_phot_opt)):
        z_phot[indices[i]] = z_phot_opt[i]
        z_phot_err[indices[i]] = z_phot_err_opt[i]

    
    
    A3581_cat['z_phot'] = z_phot
    A3581_cat['z_phot_err'] = z_phot_err
    print(A3581_cat['z_phot'])
    A3581_cat.write('./GOOD_Data/A3581_cat.fits', overwrite = True)
   
    
    z_phot, z_phot_err = A3581_cat['z_phot'], A3581_cat['z_phot_err']
    z_spec_opt, indices = ascii.read('./GOOD_Data/optical_sources_new.csv')['Spec_z'], ascii.read('./GOOD_Data/optical_sources_new.csv')['Index']
    z_spec = np.full(len(A3581_cat), fill_value = -1, dtype = 'float64')
    z_spec[np.array(indices)] = z_spec_opt

    print([z_spec[z_phot == -1] > 0])
    z_best, in_clust = redshift(z_phot, z_phot_err, z_spec)
    A3581_cat['z_best'] = z_best
    print(len(in_clust), len(z_best))
    A3581_cat['in_clust'] = in_clust
    A3581_cat.write('./GOOD_Data/A3581_cat.fits', overwrite = True)
    '''
    
  


