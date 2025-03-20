'''
Code to clean POSSUM RM catalogs including stokes I, Q, U, FDF info
@author Affan Khadir 
@date Mar 18, 2025
'''

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
    data_cleaned = data[np.bitwise_and((snr > 10), (pol_frac > 1/100))]
    FDFs_cleaned = FDFs[np.bitwise_and((snr > 10), (pol_frac > 1/100))]
    spectra_cleaned = spectra[np.bitwise_and((snr > 10), (pol_frac > 1/100))]
    
    data_new, FDFs_new, spectra_new  = removing_duplicates(data_cleaned, FDFs_cleaned, spectra_cleaned)

    data_new.write(sbid + '/catalog_cleaned.csv', overwrite =True)
    FDFs_new.write(sbid + '/FDFs_cleaned.fits', overwrite =True)
    spectra_new.write(sbid + '/spectra_cleaned.fits', overwrite =True)

catalog_cleaning('../sb50413')

