import numpy as np
# import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import glob
from scipy.special import gammainc


class Gwnimble:

    def __init__(self, filterpath, spectrapath, spectime, obs_distance=41.E6,
                 throughput=0.5, filterextrapolate=False, detectorarea=804.248):
        self.throughput = throughput
        self.detectorarea = detectorarea # in cm^2
        self.read_bandpass(filterpath)
        self.read_obs_spectra(spectrapath, filterextrapolate=filterextrapolate)
        self.obs_distance = obs_distance
        self.time = spectime
        self.filterlc = self.get_obs_filterlc()


    def read_bandpass(self, path):
        self.filter_nm, self.filter_pct = np.genfromtxt(path,
            unpack=True, skip_footer=1, skip_header=1)
        self.filter_ang = self.filter_nm * 10
        self.filter_qe = self.filter_pct / 100.
        self.filter_qe[self.filter_qe<0.0] = 0.0
        self.filter_thru = self.filter_qe * self.throughput
        self._f2 = interp1d(self.filter_ang, self.filter_thru, kind='slinear',
                            bounds_error=False, fill_value=0)

    def read_obs_spectra(self, pathlist, filterextrapolate=False):
        self.numspec = len(pathlist)
        self.wavelength_ang = np.arange(1800, 12000, 1)
        spectra = np.zeros([self.numspec, self.wavelength_ang.shape[0]])
        shotnoise = np.zeros(self.numspec)
        nphotons = np.zeros(self.numspec)
        for i, s in enumerate(pathlist):
            gw_ang, gw_f_lam = np.genfromtxt(s,
                                             delimiter=',', unpack=True)
            filter_qe_interp = self._f2(gw_ang)
            gw_f_lam_filter = filter_qe_interp * gw_f_lam
            if filterextrapolate:
                f3 = interp1d(gw_ang, gw_f_lam_filter, kind='slinear',
                              bounds_error=False, fill_value='extrapolate')
            else:
                f3 = interp1d(gw_ang, gw_f_lam_filter, kind='slinear',
                              bounds_error=False, fill_value=0)
            y = f3(self.wavelength_ang)
            y[y < 0] = 0
            shotnoise[i], nphotons[i] = self.get_photon_noise(
                y, integration=1, returnphotons=True)
            spectra[i] = y
        self.spectra_l = spectra
        self.shotnoise_l = shotnoise
        self.nphotons_l = nphotons

    def get_spectra_nu(self):
        spectra_nu = (3.34E4 * self.wavelength_ang**2 *
                      self.spectra_l)
        return spectra_nu

    def get_flam(self, w):
        flam = 5.03E7 * w * self._f50(w)
        return flam

    def get_photon_noise(self, spectrum, integration=0.5, returnphotons=False):
        self._f50 = interp1d(self.wavelength_ang, spectrum)
        nphot = integrate.quad(self.get_flam, 1800, 12000)[0]
        nphotons = nphot * self.detectorarea * (integration * 3600)
        shotnoisefrac = np.sqrt(nphotons) / nphotons
        if returnphotons:
            return shotnoisefrac, nphotons
        else:
            return shotnoisefrac

    def get_obs_filterlc(self):
        lc = np.zeros(self.numspec)
        for i, spec in enumerate(self.get_spectra_nu()):
            f4 = interp1d(self.wavelength_ang,
                          spec, kind='slinear',
                          bounds_error=False, fill_value=0)
            integral = (
                integrate.quad(
                    f4, 1800, 12000, limit=500)[0] /
                (integrate.quad(
                    self._f2, 1800, 12000, limit=500)[0] * 3631))

            # get TESS band magntude using the AB formula
            m = - (5 / 2) * np.log10(integral)
            lc[i] = m
        return lc

    def get_obs_absmag(self):
        m = self.filterlc
        absM = self.get_absmag(m, self.obs_distance)
        return absM

    def get_absmag(self, m, distance):
        absM = m - (5 * (np.log10(distance) - 1))
        return absM

    def get_apmag(self, M, distance):
        m = M + (5 * (np.log10(distance) - 1))
        return m

    def interp_obs(self, interptime):
        obsmag = self.get_obs_absmag()
        self._f5 = interp1d(self.time, obsmag, kind='cubic',
                            bounds_error=False,
                            fill_value='extrapolate')
        return self._f5(interptime)

    def interp_shot(self, mag):
        # obsmag = np.sort(self.filterlc[self.filterlc < 29])
        # shotnoise = self.shotnoise_l[np.argsort(self.filterlc[self.filterlc < 29])]
        # nphotons = self.nphotons_l[np.argsort(self.filterlc[self.filterlc < 29])]
        obsmag = np.sort(self.filterlc)
        shotnoise = self.shotnoise_l[np.argsort(self.filterlc)]
        nphotons = self.nphotons_l[np.argsort(self.filterlc)]
        f50 = interp1d(obsmag, shotnoise, kind='quadratic',
                               bounds_error=False,
                               fill_value='extrapolate')
        f51 = interp1d(obsmag, nphotons, kind='linear',
                               bounds_error=False,
                               fill_value='extrapolate')
        shot = f50(mag)
        nphot = f51(mag)
        nphot[nphot < 0] = 1.E-11
        shot[shot < 0] = 0
        read = 3000/ nphot
        shot = np.sqrt(shot**2 + 0.0001**2 + read**2)
        return shot

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


    def get_yerr_mag(self, mag, integration=0.5):
        noise = self.TESS_noise_1h(mag) / 1.E6 / np.sqrt(integration)
        yerr = np.log10(1.0 - (noise)) / -0.4
        return yerr

    def get_yerr_mag_shot(self, mag, integration=0.5):
        noise = self.interp_shot(mag) / np.sqrt(integration)
        yerr = np.log10(1.0 - (noise)) / -0.4
        return yerr

