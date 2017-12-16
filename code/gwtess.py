import numpy as np
# import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import glob
from scipy.special import gammainc


class Gwtess:

    def __init__(self, tesspath, spectrapath, spectime, obs_distance=41.E6,
                 mergerrate=1540, limitingmag=18.4,
                 skycoverage=0.041,):
        self.read_bandpass(tesspath)
        self.read_obs_spectra(spectrapath)
        self.obs_distance = obs_distance
        self.limitingmag = limitingmag
        self.skycoverage = skycoverage

        self.time = spectime
        self.mergerrate = mergerrate

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

    def get_snr(self, mag, exposure=0.5):
        noise = self.TESS_noise_1h(mag)
        snr = 1.E6 / (noise / np.sqrt(exposure))
        return snr

    def get_distances(self, years=10):
        # draw from a sphere
        # we want the volume to be 10 Gpc, so
        vol = 10
        r = (vol / ((4 / 3) * np.pi))**(1 / 3)
        ndim = 3
        center = [0, 0, 0]
        nmergers = self.mergerrate * vol
        x = np.random.normal(size=(nmergers * years, 3))
        ssq = np.sum(x**2, axis=1)
        fr = r * gammainc(ndim / 2, ssq / 2)**(1 / ndim) / np.sqrt(ssq)
        frtiled = np.tile(fr.reshape(nmergers * years, 1), (1, ndim))
        p = center + np.multiply(x, frtiled)
        distances = np.sqrt(p.T[0]**2 + p.T[1]**2 + p.T[2]**2)
        self.distances = distances * 1.E9

    def interp_obs(self, interptime):
        self.obsmag = self.get_obs_absmag()
        self._f5 = interp1d(self.time, self.obsmag, kind='cubic',
                            bounds_error=False,
                            fill_value='extrapolate')
        return self._f5(interptime)

    def is_detectable(self, apmag):
        return apmag < self.limitingmag

    def sim_events(self, fixedlum=True, years=10):
        self.get_distances(years=years)
        if fixedlum:
            apmags, detected = self.fixedlumsim()
            returnitems = [apmags, detected]
        elif not fixedlum:
            apmags, detected, is_onaxis = self.variablelumsim()
            returnitems = [apmags, detected, is_onaxis]

        return returnitems

    def fixedlumsim(self):
        interptime = np.arange(0, 3, 0.1)
        maxbrightness = np.min(self.interp_obs(interptime))

        apmags = self.get_apmag(maxbrightness, self.distances)

        detected = self.is_detectable(apmags)

        return apmags, detected

    def variablelumsim(self):
        interptime = np.arange(0, 3, 0.1)
        maxbrightness = np.min(self.interp_obs(interptime))

        size = self.distances.shape[0]

        # about 3.9% are on axis
        mod_on = self.get_onaxis(size)
        mod_off = self.get_offaxis(size)
        rand = np.random.sample(size=size)
        is_onaxis = rand < 0.039
        mod = np.where(is_onaxis, mod_on, mod_off)
        magmod = - 2.5 * np.log10(mod)
        absmag_mod = maxbrightness + magmod

        apmags = self.get_apmag(absmag_mod, self.distances)

        detected = self.is_detectable(apmags)

        return apmags, detected, is_onaxis

    def get_onaxis(self, size):
        # range from 3x as bright to 130x as bright
        # distributed uniformly in log space
        low = 1.1  # x3 in log_e
        high = 4.9  # x130 in log_e
        return np.exp(np.random.uniform(low, high, size))

    def get_offaxis(self, size):
        # range from half as bright to 10x as bright
        # distributed uniformly in log space
        low = -1.2  # x0.3 in log_e
        high = 2.6  # x13 in log_e
        return np.exp(np.random.uniform(low, high, size))

    def get_yerr_mag(self, mag, integration=0.5):
        noise = self.TESS_noise_1h(mag) / 1.E6 / np.sqrt(integration)
        yerr = np.log10(1.0 - (noise)) / -0.4
        return yerr



class Gwtess_galaxy(Gwtess):

    def __init__(self, tesspath, spectrapath, spectime,
                 obs_distance=41.E6,
                 mergerrate=1540,
                 skycoverage=0.041, ):
        super().__init__(tesspath, spectrapath, spectime,
                         obs_distance=41.E6,
                         mergerrate=1540, limitingmag=None,
                         skycoverage=0.041, )
        # self.read_galaxy(galaxypath)

    def is_detectable_with_galaxy(self, apmag, galaxymag,
                                  integration=6., sigma=5.):

        noise_no_gal = self.TESS_noise_1h(apmag) / 1.E6 / np.sqrt(integration)
        gal_noise = self.get_photon_noise(apmag, galaxymag)
        sigmalevel = 1. / np.sqrt((noise_no_gal)**2 + gal_noise**2)

        return sigmalevel > sigma

    def get_photon_noise(self, apmag_noise, apmag_target):
        # equation for photons for tess
        # from sullivan 2015
        photons_0mag = (1.514E6 * 69.1 * 3600)
        y = photons_0mag * 10**(-0.4 * apmag_noise)
        z = np.sqrt(y) / (photons_0mag * 10**(-0.4 * apmag_target))
        return z

    def variablelumsim(self):
        interptime = np.arange(0, 3, 0.1)
        maxbrightness = np.min(self.interp_obs(interptime))

        size = self.distances.shape[0]

        mod_on = self.get_onaxis(size)
        mod_off = self.get_offaxis(size)
        rand = np.random.sample(size=size)
        is_onaxis = rand < 0.039
        mod = np.where(is_onaxis, mod_on, mod_off)
        magmod = - 2.5 * np.log10(mod)
        absmag_mod = maxbrightness + magmod

        apmags = self.get_apmag(absmag_mod, self.distances)

        self.galaxy_apmags = self.get_galaxy_apmag(self.distances)

        detected = self.is_detectable_with_galaxy(apmags, self.galaxy_apmags)

        return apmags, detected, is_onaxis

    def get_galaxy_apmag(self, distances):
        galaxy_absmag = np.random.normal(-20.9, 1.0, 
            size=np.shape(distances)[0])
        galaxy_apmag = self.get_apmag(galaxy_absmag, distances)
        return galaxy_apmag

    def get_yerr_mag(self, mag, integration=0.5):
        noise_no_gal = self.TESS_noise_1h(apmag) / 1.E6 / np.sqrt(integration)
        gal_noise = self.get_photon_noise(apmag, galaxymag)
        noise = np.sqrt((noise_no_gal)**2 + gal_noise**2)
        yerr = np.log10(1.0 - (noise)) / -0.4
        return yerr


#     def read_galaxy(self, path):
#         self.galaxy_ang, self.galaxy_f_mjy = np.genfromtxt(path,
#                                                            delimiter=',',
#                                                            unpack=True)
#         self.galaxy_f_jy = self.galaxy_f_mjy * 1.E-3
#         tess_qe_interp = self._f2(self.galaxy_ang)
#         galaxy_f_jy_tess = tess_qe_interp * self.galaxy_f_jy

#         self._f6 = interp1d(self.galaxy_ang, galaxy_f_jy_tess, kind='slinear',
#                             bounds_error=True)
#         integral = (
#             integrate.quad(
#                 self._f6, 4000, 12000, limit=500)[0] /
#             (integrate.quad(
#                 self._f2, 4000, 12000, limit=500)[0] * 3631))
#         self.gal_obs_mag = - (5 / 2) * np.log10(integral)

#     def get_gal_obs_absmag(self):
#         m = self.gal_obs_mag
#         absM = self.get_absmag(m, self.obs_distance)
#         return absM

#     def gal_noise_term(self, distance):
#         absM = self.get_gal_obs_absmag()
#         apM = self.get_apmag(self, absM, distance)
#         # equations for photoncounts from sullivan 2015
#         ph0 = (1.514E6 * 70 * 3600) # for 1 hour integration
#         phgal= x * 10**(-0.4 * apM)
#         z = np.sqrt(y) / (x * 10**(-0.4*17.3))
#         noiseterm = ph0


if '__name__' == '__main__':
    tesspath = '../data/tess-bandpass.csv'
    spectrapath = glob.glob('../data/spectrum-at-*.csv')

    GWT = Gwtess(tesspath, spectrapath)

    eventAbsMag = GWT.get_obs_absmag()
