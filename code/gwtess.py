import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import glob


class Gwtess():

    def __init__(self, tesspath, spectrapath):
        self.read_bandpass(tesspath)
        self.read_obs_spectra(spectrapath)
        self.obs_distance = 40.E6  # 40 Mpc

        self.time = [0.49, 0.53, 1.46, 2.49, 3.46, 4.51, 7.45, 8.46]

    def read_bandpass(self, path):
        self.tess_nm, self.tess_qe = np.genfromtxt(path,
                                                   delimiter=',', unpack=True)
        self.tess_ang = self.tess_nm * 10
        self._f2 = interp1d(self.tess_ang, self.tess_qe, kind='slinear',
                            bounds_error=False, fill_value=0)

    def read_obs_spectra(self, pathlist):
        self.numspec = len(pathlist)
        self.wavelength_ang = np.arange(4000, 12000, 1)
        spectra = np.zeros([self.numspec, self.wavelength_ang.shape[0]])

        for i, s in enumerate(pathlist):
            gw_ang, gw_f_lam = np.genfromtxt(s,
                                             delimiter=',', unpack=True)
            tess_qe_interp = self._f2(gw_ang)
            gw_f_lam_tess = tess_qe_interp * gw_f_lam
            f3 = interp1d(gw_ang, gw_f_lam_tess, kind='slinear',
                          bounds_error=False, fill_value=0)
            y = f3(self.wavelength_ang)
            spectra[i] = y
        self.spectra_l = spectra

    def get_spectra_nu(self):
        spectra_nu = (3.34E4 * self.wavelength_ang**2 *
                      self.spectra_l)
        return spectra_nu

    def get_obs_tesslc(self):
        lc = np.zeros(self.numspec)
        for i, spec in enumerate(self.get_spectra_nu()):
            f4 = interp1d(self.wavelength_ang,
                          spec, kind='slinear',
                          bounds_error=False, fill_value=0)
            integral = (
                integrate.quad(
                    f4, 4000, 12000, limit=500)[0] /
                (integrate.quad(
                    self._f2, 4000, 12000, limit=500)[0] * 3631))

            # get TESS band magntude using the AB formula
            m = - (5 / 2) * np.log10(integral)
            lc[i] = m
        return lc

    def get_obs_absmag(self):
        m = self.get_obs_tesslc()
        absM = self.get_absmag(m, self.obs_distance)
        return absM

    def get_absmag(self, m, distance):
        absM = m - (5 * (np.log10(distance) - 1))
        return absM

    def get_apmag(self, M, distance):
        m = M + (5 * (np.log10(distance) - 1))
        return m

    def TESS_noise_1h(self, mag):
        """
        returns noise in ppm for a transit of 1 hours duraton
        """
        mag_level, noise_level = np.genfromtxt(
            '../data/TessNoise_1h_v2.csv', delimiter=',',
            unpack=True,
            comments='#')

        # we probably shouldn't trust any extrapolated values
        # but it's all guess work at this stage anyway
        mag_interp = interp1d(mag_level, noise_level,
                              kind='cubic', fill_value='extrapolate')
        return mag_interp(mag)


if '__name__' == '__main__':
    tesspath = '../data/tess-bandpass.csv'
    spectrapath = glob.glob('../data/spectrum-at-*.csv')

    GWT = Gwtess(tesspath, spectrapath)

    eventAbsMag = GWT.get_obs_absmag()
