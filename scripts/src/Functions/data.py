import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from scipy import spatial


def get_mock_data(location, data_name):
    # These
    data_path = './Data/' + location + '/' + data_name + '/'
    data = np.load(data_path + 'data.npy')
    std = np.load(data_path + 'noise_sigma.npy')
    theta = np.load(data_path + 'theta.npy')
    phi = np.load(data_path + 'phi.npy')

    return theta, phi, data, std
    
    

def cluster_exclusion(ra, dec, rm, rm_err, nearest_cluster_ras, nearest_cluster_decs, neares_cluster_r500, nearest_cluster_z):
    '''
    Function to exclude possible extragalactic structure due to clusters 

    Parameters
    ----------
    ra: numpy array object 
        Array of ras
    dec: numpy array object 
        Array of decs
    rm: numpy array object 
        Array of rms 
    rm_err: numpy array object 
        Array of rms 
    nearest_cluster_ras: list
        RAs of clusters to mask 
    nearest_cluster_decs: list
        DECs of clusters to mask
    nearest_cluster_r500: list
        R500 of clusters to mask 
    nearest_cluster_z: list
        redshifts of clusters to mask
    '''
    ra_final = ra
    dec_final = dec
    rm_final = rm
    rm_err_final = rm_err
    coords_final = SkyCoord(ra_final, dec_final, unit = 'deg')

    cluster_points = SkyCoord(nearest_cluster_ras, nearest_cluster_decs, unit = 'deg')


    
    ra_exc = []
    dec_exc = []
    rm_exc = []
    rm_err_exc = []
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    for j in range(len(coords_final)):
        cluster_seps = []
        for i in range(len(cluster_points)):
                scale = cosmo.kpc_proper_per_arcmin(nearest_cluster_z[i]).to(u.Mpc/u.arcmin)
                cluster_seps.append((coords_final[j].separation(cluster_points[i]) * scale.to(u.Mpc/u.deg)).value / np.array(neares_cluster_r500[i]))
        if  np.min(cluster_seps) > 2:
            ra_exc.append(ra_final[j])
            dec_exc.append(dec_final[j])
            rm_exc.append(rm_final[j])
            rm_err_exc.append(rm_err_final[j])
    return ra_exc, dec_exc, rm_exc, rm_err_exc
 

def remove_extragal_rm(theta, phi, data, noise_sigma, cutoff = 3):
    """
    Function to remove the extragalactic RM
    
    Parameters
    ----------
    theta: numpy array
        The ras
    phi: numpy array
        The decs
    data: numpy array
        the rms
    std: numpy array
        RM_err
    cutoff: The cutoff for removing the extragalactic RM; 3 by default

    Returns
    --------
    ntheta: The RA of the points after removing the extragalactic RM for both simulations
    nphi: Th DEC of the points after removing the extragalactic RM for both simulations
    data: RM data after removal of extragalactic sources for both simulatons
    noise: The noise in the RM after the removal of the extragalctic sources for both simulations
    noise_sigma: The deviation in the noise in the RM after removal of the extragalactic sources for both simulations
    """
    #Loading the data

    points= np.vstack((theta, phi))
    points = np.transpose(points)

    #Code to find he nearest 10 neighbours of a given data point
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    scale = cosmo.kpc_proper_per_arcmin(0.0221).to(u.Mpc/u.arcmin)
    i = 0
    while i < len(data):
        currentdata = np.delete(data, i)
        currentnoise_sigma = np.delete(noise_sigma, i)
        currentpts = np.delete(points, i, 0)
        currentpoint = points[i]
        current_sky = SkyCoord(currentpoint[0] * u.deg, currentpoint[1] * u.deg, frame = 'galactic')
        rmlist = np.zeros((10))
        counter = 0
        while counter < 9:
            distance,index = spatial.KDTree(currentpts).query(currentpoint)
            closest_point = SkyCoord(currentpts[index][0]*u.deg, currentpts[index][1]*u.deg, frame = 'galactic')
            if (closest_point.separation(current_sky) * scale).to(u.kpc).value > 100:
                rmlist[counter] = currentdata[index]
                counter = counter + 1
            currentdata = np.delete(currentdata, index, 0)
            currentnoise_sigma = np.delete(currentnoise_sigma, index, 0)
            currentpts = np.delete(currentpts, index, 0)
        mean = np.mean(rmlist)
        std = np.std(rmlist)
        if data[i] >= (mean+3*std) or data[i]<=(mean - 3*std):
            data = np.delete(data, i)
            noise_sigma = np.delete(noise_sigma, i)
            points = np.delete(points, i, 0)
        i = i + 1


    ntheta = points[: , 0]
    nphi = points[:, 1]

    return ntheta, nphi, data, noise_sigma

def get_real_data():
    cluster_z = [0.0200, 0.0257, 0.0230, 0.0221]
    cluster_ras = [208.57704, 214.61075, 217.45457, 211.8742]
    cluster_decs = [-26.89382, -27.37885, -29.74854, -27.0178]
    cluster_r500 = [0.625, 0.8251953125, 0.52197265625, 0.925]
    catalog_final = Table.read('../data_files/sim_sky.fits')
    data = catalog_final['rm'].data
    std = catalog_final['rm_err'].data
    ra = catalog_final['ra'].data
    dec = catalog_final['dec'].data
    
    #ra_exc, dec_exc, rm_exc, rm_err_exc = cluster_exclusion(ra, dec, data, std, cluster_ras, cluster_decs, cluster_r500, cluster_z)
    
    #coordinates = SkyCoord(ra_exc * u.deg, dec_exc * u.deg, frame = 'icrs')
    coordinates = SkyCoord(ra * u.deg, dec * u.deg)    
    coordinates_gal = coordinates.galactic
    
    
    
    theta, phi = coordinates_gal.l.value, coordinates_gal.b.value
  

    #ntheta, nphi, ndata, nstd = remove_extragal_rm(theta, phi, rm_exc, rm_err_exc)
    ntheta, nphi, ndata, nstd = remove_extragal_rm(theta, phi, data, std)
    return ntheta, nphi, ndata, nstd
    
    
def get_simulated_data():
    raise NotImplementedError()
