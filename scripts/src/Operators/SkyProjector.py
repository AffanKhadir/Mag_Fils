import nifty8 as ift
import numpy as np
import healpy as hp


class SkyProjector(ift.LinearOperator):
    def __init__(self, domain, target, theta, phi):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._pixels = self.calc_pixels(self._domain[0].nside, self._target[0].shape[0],
                                        theta, phi).astype(int)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def _times(self, x):
        return ift.Field(self.target, x.val[self._pixels])

    def _adjoint_times(self, x):
        ns = self.domain[0].nside
        pmap = np.zeros(12 * ns ** 2)
        for i in range(len(self._pixels)):
            pmap[self._pixels[i]] += x.val[i]
        return ift.Field(self.domain, pmap)

    def counts(self):
        ns = self.domain[0].nside
        pmap = np.zeros(12 * ns ** 2)
        for i in range(len(self._pixels)):
            pmap[self._pixels[i]] += 1
        pmap[pmap == 0] = 1
        return ift.Field(self.domain, pmap)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return self._times(x)
        return self._adjoint_times(x)

    @staticmethod
    def calc_pixels(nside, ndata, colat_rad, lon_rad):
        """Calculating the pixel numbers corresponding to the Data points.
        Note that longitude can be in [-2pi, 2pi]"""
        pixels = np.zeros((ndata,))
        for i in range(ndata):
            pixels[i] = hp.ang2pix(nside, colat_rad[i], lon_rad[i])
        return pixels
