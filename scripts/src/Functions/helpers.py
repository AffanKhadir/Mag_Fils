import nifty8 as ift
import numpy as np
import healpy as hp


def load_field_model(name, domain, a_dict):
    cfmaker = ift.SimpleCorrelatedField(domain, **a_dict, prefix=name)
    return cfmaker, cfmaker.power_spectrum


def equ2gal(ra_h, ra_m, ra_s, dec_deg, dec_min, dec_sec):
    """Conversion from equatorial coordinates to galactic coordinates.
    Note that while lon_equ is in [0, 2pi] due to the ra definition, lon gal is in in [-pi, pi].
    The two conventions lead to numerically equivalent results in all further processing."""
    r = hp.Rotator(coord=['C', 'G'], deg=False)
    colat_equ_rad = (90.-dec_deg-dec_min/60.-dec_sec/3600.)*np.pi/180.
    lon_equ_rad = (ra_h/24.+ra_m/1440.+ra_s/86400.)*2.*np.pi
    colat_gal_rad, lon_gal_rad = r(colat_equ_rad, lon_equ_rad)
    return colat_gal_rad, lon_gal_rad


def gal2gal(lon_deg, lat_deg):
    """Conversion from longitude and latitude in degrees to colatitude and longitude in radians
    (matching healpy convention)."""
    colat_rad = (90.-lat_deg)*np.pi/180.
    lon_rad = lon_deg*np.pi/180.
    return colat_rad, lon_rad


def density_estimation(m1, m2, xmin, xmax, ymin, ymax):
    x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    values = np.vstack([m1, m2])
    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(values)
    z = np.reshape(kernel(positions).T, x.shape)
    return x, y, z
