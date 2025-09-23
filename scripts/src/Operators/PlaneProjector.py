import nifty8 as ift
import numpy as np


class PlaneProjector(ift.LinearOperator):
    def __init__(self, domain, target, theta, phi, center):
        self._domain = ift.makeDomain(domain)
        assert isinstance(self._domain[0], ift.RGSpace) and len(self.domain[0].shape) == 2, 'Domain must be 2d rg space'
        self._target = ift.makeDomain(target)
        assert len(center) == 2, 'expecting 2d center coordinates'
        self.center = center[0], center[1]
        pix, ind = self.calc_pixels(self._target[0].shape[0], theta, phi)
        self._pixels = pix.astype(int)
        self._indices = np.asarray(ind).astype(int)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def _times(self, x):
        xval = np.zeros(self.target[0].shape)
        xval[self._indices] += x.val[self._pixels[:, 0], self._pixels[:, 1]]
        return ift.Field(self.target, xval)

    def _adjoint_times(self, x):
        xval = np.zeros(self.domain[0].shape)
        xval[self._pixels[:, 0], self._pixels[:, 1]] += x.val[self._indices]
        return ift.Field(self.domain, xval)

    def counts(self):
        pmap = ift.full(self.target, 1.)
        pmap = self.adjoint(pmap).val
        return ift.Field(self.domain, pmap)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return self._times(x)
        return self._adjoint_times(x)

    def calc_pixels(self, ndata, colat_rad, lon_rad):
        """Calculating the pixel numbers corresponding to the data points within the plane cutout."""
        lower_corner = self.center - np.asarray(self.domain[0].distances)*np.asarray(self.domain[0].shape)*0.5
        indices = []
        pixels = []
        for i in range(ndata):
            pix_colat = np.floor((colat_rad[i] - lower_corner[0])/self.domain[0].distances[0])
            pix_lon = np.floor((lon_rad[i] - lower_corner[1])/self.domain[0].distances[1])
            if (0 <= pix_colat < self.domain[0].shape[0]) & (0 <= pix_lon < self.domain[0].shape[1]):
                pixels.append([pix_colat, pix_lon])
                indices.append(i)
        return np.asarray(pixels), indices
