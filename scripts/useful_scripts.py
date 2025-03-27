
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
import ast
from glob import glob 
import dustmaps.edenhofer2023
from dustmaps.edenhofer2023 import Edenhofer2023Query
from dustmaps.config import config
config['data_dir'] = '../data_files/'





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
    '''
    @author: Erik Osinga
    Remove the Galactic contribution to Faraday rotation.

    Parameters
    ----------
    results: astropy.Table object 
        (from astropy.table import Table; results = Table.read(fitstable.fits) )

    which: 'huts' -> use the map from Hutschenreuter2022
         : 'opperman' -> use the map from Opperman+2012  https://arxiv.org/abs/1111.6186
    
    Returns
    -------
    results: astropy.Table object 
        returns RM catalogue after correcting for the GRM
    '''
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

def fwhm_finder(x_array, y_array):
    '''
    @author: Affan Khadir 
    Python function to find the FWHm of a curve

    Parameters
    ----------
    x_array: numpy array
        array of x values
    y_array: numpy array
        array of y values 
    
    Returns
    -------
    FWHM: float 
        the fwhm of the curve
    '''
    maximum = np.nanmax(np.abs(y_array))
    minimum = np.nanmin(np.abs(y_array))

    difference = maximum - minimum

    HM = difference / 2

    nearest = min(range(len(np.abs(y_array - HM))), key=(np.abs(y_array -HM)).__getitem__)
    index_max = max(range(len(y_array)), key=y_array.__getitem__)

    FWHM = 2 * np.abs(x_array[nearest] - x_array[index_max])

    return FWHM

def find_m2(spectra_file, fdf_file):
    '''
    @author: Affan Khadir
    Python function to find the second moment using RM synthesis and RM cleaning

    Parameters
    ----------
    spectra_file: str
        file name for the Stokes spectra
    fdf_file: str 
        file name for the FDFs, used to extract dphi from the RMSF
    
    Returns
    -------
    None
    '''
    FDFs_final = Table.read(fdf_file)
    phi_rmsf = FDFs_final['phi_rmsf']
    rmsf = FDFs_final['rmsf']

    dphi = fwhm_finder(phi_rmsf[0], np.abs(rmsf[0]))

    # Preparing the Stokes spectra for RM synthesis and RM cleaning
    spectra_final = Table.read(spectra_file)
    m2_vals = []
    for i in range(len(spectra_final)):
        spectra_curr = spectra_final[i]
        stokes_table = Table()
        freq_spec = spectra_curr['freq']
        stokesI = spectra_curr['stokesI']
        stokesI_err = spectra_curr['stokesI_error']
        stokesQ = spectra_curr['stokesQ']
        stokesQ_err = spectra_curr['stokesQ_error']
        stokesU = spectra_curr['stokesU']
        stokesU_err = spectra_curr['stokesU_error']
        stokes_table['freq'] = freq_spec
        stokes_table['I'] = stokesI
        stokes_table['Q'] = stokesQ
        stokes_table['U'] = stokesU
        stokes_table['I_err'] = stokesI_err
        stokes_table['Q_err'] = stokesQ_err
        stokes_table['U_err'] = stokesU_err

        ascii.write(stokes_table, 'stokes_table.csv', overwrite = True, format = "no_header")

        os.system('rmsynth1d stokes_table.csv -S')
        os.system('rmclean1d stokes_table.csv -c -8 -S')

        m2 = float(pd.read_csv('stokes_table_RMclean.dat').values[15][0][10:]) / dphi
        m2_vals.append(m2)
    np.save('m2_vals.npy', m2_vals)


def fil_checker(data, cluster_ras, cluster_decs, cluster_r500, fil_width = 1, cz = 0.0221, fig = None, ax = None):
    '''
    @author: Affan Khadir

    Must recheck the plotting of clusters, filament, bridge region later!!!

    Function to produce flags for when a particular point is on the same region of the sky as a filament

    Parameters 
    ----------
    data: astropy.Table object
        The table of the data (ra, dec, val)
    cluster_ra: numpy array
        Array with the ras of the clusters
    cluster_dec: numpy array
        Array with decs of the clusters
    cluster_r500: numpy array
        Array with r_500 of the clusters
    fil_width: int or float 
        The width of the filament (in Mpc)
    cz: float 
        The redshift of the object-of-interest
    fig: matplotlib.pyplot figure object 
        Matplotlib figure object for plotting. None by default
    ax: matplotlib.pyplot axes object 
        Matplotlib axes object for plotting. None by default
    
    Returns
    -------
    fil_flag: numpy array
        Numpy array of booleans (1 for part of filament and 0 for not a part of the filament)
    clust_flag: numpy array
        Numpy array of booleans (1 for part of a cluster and 0 for not part of a cluster)
    bridge_flag: numpy array
        Numpy array of booleans (1 for part of a bride and not the whole filament, 0 for otherwise)
    '''
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    scale = cosmo.kpc_proper_per_arcmin(cz).to(u.Mpc/u.arcmin)

    ra, dec = data['ra'], data['dec']

    coords = SkyCoord(ra, dec, unit = 'deg')

    cluster_points = SkyCoord(cluster_ras, cluster_decs, unit = 'deg')
    fil_flag = np.zeros(len(data), dtype = bool)
    cluster_flag = np.zeros(len(data), dtype = bool)
    bridge_flag = np.zeros(len(data), dtype = bool)
    x_array = []
    y_array = []


    for i in range(len(cluster_points)):
        clust_new = np.delete(cluster_points, i)
        separation = cluster_points[i].separation(clust_new)
        min_index = np.argmin(separation)
        m = (cluster_decs[i] - np.delete(cluster_decs, i)[min_index]) / (cluster_ras[i] - np.delete(cluster_ras, i)[min_index])
        b = cluster_decs[i] - m * cluster_ras[i]
        def linear(x, m, b): 
            return m*x + b
        x = np.linspace(cluster_ras[i],np.delete(cluster_ras, i)[min_index], 3000)
        y = linear(x, m , b)
        x_array = x_array + np.ndarray.tolist(x)
        y_array  = y_array + np.ndarray.tolist(y)

        vec = [cluster_ras[i] - np.delete(cluster_ras, i)[min_index], cluster_decs[i] - np.delete(cluster_decs, i)[min_index]]
        unit_vec = vec/np.linalg.norm(vec)
        perp_vec = np.array([-unit_vec[1], unit_vec[0]])

        array_xy = [[x[j], y[j]] for j in range(len(x))]

        up_line = array_xy + np.tile(perp_vec * (fil_width * (u.Mpc) / scale).to(u.deg).value, (len(x), 1))
        up_x = up_line[:, 0]
        up_y = up_line[:, 1]
        down_line = array_xy - np.tile(perp_vec * (fil_width * (u.Mpc) / scale).to(u.deg).value, (len(x), 1))
        down_x = down_line[:, 0]
        down_y = down_line[:, 1]

        if fig is not None: 
            r_plotting = SphericalCircle((cluster_ras[i]* u.degree, cluster_decs[i]*u.degree),  (2* cluster_r500[i] * (u.Mpc) / scale).to(u.deg), linestyle = '--', facecolor = 'none', edgecolor = 'k', label = 'Clusters', transform = ax.get_transform('icrs'), lw = 10)
            ax.add_patch(r_plotting)
    
    if fig is not None:
        up_x_bridge = []
        up_y_bridge = []
        down_x_bridge = []
        down_y_bridge = []
        up_coords = SkyCoord(up_x * u.deg, up_y * u.deg)
        down_coords = SkyCoord(down_x * u.deg, down_y * u.deg)
        for j in range(len(up_x)):
            if np.min((up_coords[j].separation(cluster_points) * scale).to(u.Mpc).value / np.array(cluster_r500)) < 2:
                up_x_bridge.append(np.nan)
                up_y_bridge.append(np.nan)
            else:
                up_x_bridge.append( up_x[j])
                up_y_bridge.append( up_y[j])
            if np.min((down_coords[j].separation(cluster_points) * scale).to(u.Mpc).value / np.array(cluster_r500)) < 2:
                down_x_bridge.append(np.nan)
                down_y_bridge.append(np.nan)
            else: 
                down_x_bridge.append( down_x[j])
                down_y_bridge.append( down_y[j])
        ax.plot(up_x_bridge, up_y_bridge, c = 'tomato', ls = '-.', label = 'Bridge', transform = ax.get_transform('icrs'), lw = 5)
        ax.plot(down_x_bridge, down_y_bridge, c = 'tomato', ls = '-.', transform = ax.get_transform('icrs'), lw = 5)
    

    line_coords = SkyCoord(x_array, y_array, unit = 'deg')
    for j in range(len(coords)): 
            line_sep = min(coords[j].separation(line_coords))
            cluster_seps = (coords[j].separation(cluster_points) * scale).to(u.Mpc).value / np.array(cluster_r500)
            if (line_sep * scale).to(u.Mpc).value <= 1 or np.min(cluster_seps) <= 2:
                fil_flag[j] = 1
            if np.min(cluster_seps) <= 2:
                cluster_flag[j] = 1
            if (line_sep * scale).to(u.Mpc).value <= 1 and np.min(cluster_seps) > 2:
                bridge_flag[j] = 1
    return fil_flag, cluster_flag, bridge_flag

def dust_maps_plotting(catalog_name, radio_image_name, cluster_ras, cluster_decs, cluster_r500, fil_width = 1, cz = 0.0221):
    '''
    @author: Affan Khadir 
    Function to plot an overlay of the dustamps on the RM grid

    Parameters
    ---------
    catalog_name: str
        The file name of the RM catalogue 
    radio_image_name: str
        The file name of the radio image for obtaining the WCS object for plotting
    cluster_ra: numpy array
        Array with the ras of the clusters
    cluster_dec: numpy array
        Array with decs of the clusters
    cluster_r500: numpy array
        Array with r_500 of the clusters
    fil_width: int or float 
        The width of the filament (in Mpc)
    cz: float 
        The redshift of the object-of-interest

    Returns
    -------
    None
    '''
    # Loading in RM data

    catalog_final = Table.read(catalog_name)
    ra =  catalog_final['ra']
    dec = catalog_final['dec']

    ra_linspace = np.linspace(np.min(ra) -4, np.max(ra) +4, 2048)
    dec_linspace = np.linspace(np.min(dec) -4, np.max(dec) +4, 2048)

    ra_mesh, dec_mesh = np.meshgrid(ra_linspace, dec_linspace)

    ra_mesh_final = ra_mesh.ravel()
    dec_mesh_final = dec_mesh.ravel()
    coords = SkyCoord(ra * u.deg, dec * u.deg, distance = np.full(len(ra), 1244) * u.pc) 
    coords_mesh = SkyCoord(ra_mesh_final*u.deg, dec_mesh_final*u.deg, distance = np.full(len(ra_mesh_final), 1244) * u.pc)
    rrm, rrm_err = catalog_final['rm_corrected'], catalog_final['rm_err']
    
    # Loading in the dust map 

    
    ed = Edenhofer2023Query(integrated=True)
    ism_dust = ed.query(coords_mesh)
    

    
    # Loading in the radio data for WCS

    radio = fits.open(radio_image_name)
    wcs = WCS(radio[0].header).celestial
    fig, ax = plt.subplots(ncols=1, subplot_kw={'projection': wcs}, figsize = (20, 10))
    fig.patch.set_facecolor('white')

    for i in range(len(cluster_ras)):
        r_plotting = SphericalCircle((cluster_ras[i]* u.degree, cluster_decs[i]*u.degree),  (2* cluster_r500[i] * (u.Mpc) / scale).to(u.deg), linestyle = '-', facecolor = 'none', edgecolor = 'k', label = 'Clusters', transform = ax.get_transform('icrs'), lw = 10)
        ax.add_patch(r_plotting)
        ax.scatter(cluster_ra[z_indices][i] * u.degree, cluster_dec[z_indices][i]*u.degree, transform = ax.get_transform('icrs'), c = 'k', s= 0.1)

    fil_flag, cluster_flag, bridge_flag = fil_checker(catalog_final, cluster_ras, cluster_decs, cluster_r500, fig = fig, ax = ax)

    c1 = ax.pcolormesh(ra_mesh, dec_mesh, ism_dust.reshape(2048, 2048).T, shading = 'auto', cmap = 'viridis', transform = ax.get_transform('icrs'), vmin = 0.05, vmax = 0.12)
    
    c = ax.scatter(ra * u.degree, dec * u.degree, c = rrm, cmap = 'seismic', transform = ax.get_transform('icrs'), vmin = -50, vmax = +50)
   

    cb1 = fig.colorbar(c1, ax = ax)
    cb1.set_label('$E$')
    colorbar = fig.colorbar(c, ax = ax)
    colorbar.set_label('RRM (rad m$^{-2}$)')
    ax.set_aspect(1)

    xlim = [min(ra) - 2, max(ra) + 2]*u.deg
    ylim = [min(dec) - 2, max(dec) + 2]*u.deg

    xlim, ylim = wcs.world_to_pixel(SkyCoord(xlim, ylim))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_xaxis()
    plt.xlabel('RA (J2000)')
    plt.ylabel('DEC (J2000)')
    plt.title('Edenhofer+24 Dust Map Overlay (1244 pc)')
    plt.savefig('../figures/dust_overlay.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()
    

    fig2, ax2 = plt.subplots(figsize = (5.84685039, 4.46338583))
    fig2.set_facecolor('white')
    ism = ed(coords)
    ax2.errorbar( ism[~fil_flag], np.abs(rrm[~fil_flag]), yerr = rrm_err[~fil_flag], fmt = '.', capsize = 5, color = '#1f77b4', label = 'Off-Filament')
    ax2.errorbar(ism[bridge_flag], np.abs(rrm[bridge_flag]), yerr = rrm_err[bridge_flag], fmt = '.', capsize = 5, color = 'tomato', label = 'Bridge')
    ax2.errorbar(ism[cluster_flag], np.abs(rrm[cluster_flag]), yerr = rrm_err[cluster_flag], fmt = '.', capsize = 5, color = 'k', label = 'Cluster')
    ax2.set_xlabel('$E$')
    ax2.set_ylabel('|RRM| (rad m$^{-2}$)')
    plt.legend()
    plt.savefig('../figures/rrm_v_duste.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()

    fig3, ax3 = plt.subplots(figsize = (5.84685039, 4.46338583))
    fig3.set_facecolor('white')
    ax3.errorbar(np.mean(ism[~fil_flag]), np.mean(np.abs(rrm[~fil_flag])), color = '#1f77b4', label = 'Off-Filament', yerr = np.std((rrm_err[~fil_flag]))/np.sqrt(len(rrm_err[~fil_flag])) , fmt = '.', capsize = 5)
    ax3.errorbar(np.mean(ism[cluster_flag]), np.mean(np.abs(rrm[cluster_flag])), color = 'k', label = 'Cluster', yerr = np.std((rrm_err[~cluster_flag]))/np.sqrt(len(rrm_err[~cluster_flag])), fmt = '.', capsize =5 )
    ax3.errorbar(np.mean(ism[bridge_flag]), np.mean(np.abs(rrm[bridge_flag])), color = 'tomato', label = 'Bridge', yerr = np.std((rrm_err[~bridge_flag]))/np.sqrt(len(rrm_err[~bridge_flag])), fmt = '.', capsize = 5)
    ax3.set_xlabel('$E$')
    ax3.set_ylabel('|RRM| (rad m$^{-2}$)')
    plt.legend()
    plt.savefig('../figures/mean_rrm_v_duste.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

    return fil_flag, cluster_flag, bridge_flag


        

cluster_ras = [208.57704, 214.61075, 217.45457, 211.8742]
cluster_decs = [-26.89382, -27.37885, -29.74854, -27.0178]
cluster_r500 = [0.625, 0.8251953125, 0.52197265625, 0.925]

dust_maps_plotting('../data_files/POSSUM_cat_final.fits', '../data_files/image.i.EMU_1412-28.SB50413.cont.taylor.0.restored.conv.fits', cluster_ras, cluster_decs, cluster_r500)

def QU_fit(spectra_file):
    '''
    @author: Affan Khadir 
    Function to QU-fit Stokes spectra for sources with a variety of models and obtain the best-fit QU model

    Paramaeters
    ----------
    spectra_file: str
        The file name for the Stokes spectra 
    
    Returns
    --------
    None
    '''
    spectra_final = Table.read(spectra_file)
    freq_spec_final = spectra_final['freq']
    stokesI_final = spectra_final['stokesI']
    stokesI_err_final = spectra_final['stokesI_error']
    stokesQ_final = spectra_final['stokesQ']
    stokesQ_err_final = spectra_final['stokesQ_error']
    stokesU_final = spectra_final['stokesU']
    stokesU_err_final = spectra_final['stokesU_error']

    models_final = []
    red_chi2_final = []
    SigmaRM_final = []
    SigmaRM_errp_final = []
    SigmaRM_errm_final = []

    models = ['m1', 'm2', 'm3', 'm4', 'm11']
    model_params = [3, 4, 7, 8, 6]



    for i in range(5):
        stokes_table = Table()
        stokes_table['freq'] = freq_spec_final[i]
        stokes_table['I'] = stokesI_final[i]
        stokes_table['Q'] = stokesQ_final[i]
        stokes_table['U'] = stokesU_final[i]
        stokes_table['I_err'] = stokesI_err_final[i]
        stokes_table['Q_err'] = stokesQ_err_final[i]
        stokes_table['U_err'] = stokesU_err_final[i]
        
        
        ascii.write(stokes_table, './stokes_table.csv', overwrite = True, format = "no_header")
        os.system('rm ./*m1*')
        os.system('rm ./*m2*')
        os.system('rm ./*m3*')
        os.system('rm ./*m4*')
        os.system('rm ./*m11*')

        os.system('rm -r ./*m1*')
        os.system('rm -r ./*m2*')
        os.system('rm -r ./*m3*')
        os.system('rm -r ./*m4*')
        os.system('rm -r ./*m11*')

        os.system('qufit stokes_table.csv -m 1 --sampler pymultinest --ncores 16')
        os.system('qufit stokes_table.csv -m 2 --sampler pymultinest --ncores 16')
        os.system('qufit stokes_table.csv -m 3 --sampler pymultinest --ncores 16')
        os.system('qufit stokes_table.csv -m 4 --sampler pymultinest --ncores 16')
        os.system('qufit stokes_table.csv -m 11 --sampler pymultinest --ncores 16')


        fit_files = glob('*_m*.dat')
        fit_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        
        model_best = 'm1'
        model_params_best = 3
        file_data = pd.read_csv(fit_files[0], delimiter = '=', header = None)
        red_chi2_best = float(file_data.iat[10, 1])
        AICc_best = float(file_data.iat[12, 1])
        ln_evd_best = float(file_data.iat[14, 1]) 
        SigmaRM_best = -1 
        SigmaRM_errp_best = -1
        SigmaRM_errm_best = -1

        for j in range(1, len(fit_files)):
            fit_file = fit_files[j]
            file_data = pd.read_csv(fit_file, delimiter = '=', header = None)
            red_chi2 = float(file_data.iat[10, 1])
            AICc = float(file_data.iat[12, 1])
            ln_evd = float(file_data.iat[14, 1])

            if  j < 4:
                SigmaRM = ast.literal_eval(file_data.iat[2, 1])[-1]
                SigmaRM_errp = ast.literal_eval(file_data.iat[3, 1])[-1]
                SigmaRM_errm = ast.literal_eval(file_data.iat[4, 1])[-1]
            else:
                SigmaRM = -1
                SigmaRM_errp = -1
                SigmaRM_errm = -1
            
        
            if np.abs(red_chi2 - 1) < np.abs(red_chi2_best - 1) and AICc < AICc_best:
                ln_bayes_factor = ln_evd - ln_evd_best
                if ln_bayes_factor > 5:
                    model_best = models[j]
                    red_chi2_best = red_chi2
                    SigmaRM_best = SigmaRM
                    SigmaRM_errp_best = SigmaRM_errp
                    SigmaRM_errm_best = SigmaRM_errm
                    AICc_best = AICc
                    ln_evd_best = ln_evd
        models_final.append(model_best)
        red_chi2_final.append(red_chi2_best)
        SigmaRM_final.append(SigmaRM_best)
        SigmaRM_errp_final.append(SigmaRM_errp_best)
        SigmaRM_errm_final.append(SigmaRM_errm_best)
    np.save('../data_files/models_final.npy', models_final)
    np.save('../dta_files/red_chi2_final.npy', red_chi2_final)
    np.save('../data_files/SigmaRM_final.npy', SigmaRM_final)
    np.save('../data_files/SigmaRM_errp_final.npy', SigmaRM_errp_final)
    np.save('../data_files/SigmaRM_errm_final.npy', SigmaRM_errm_final)




