'''


module for sps models 


'''
import os 
import pickle
# --- jax --- 
import jax.numpy as np
from jax.nn import sigmoid
# --- astropy --- 
from astropy import units as U
from astropy.cosmology import Cosmology, Planck13
# -- jaxSED --- 
from . import util as UT


class BaseModel(object): 
    ''' Base class object for different SPS models. Different `Model` objects
    specify different SPS model. The primary purpose of the `Model` class is to
    evaluate the SED given a set of parameter values. 
    '''
    def __init__(self, cosmo=None, **kwargs): 
        self._init_model(**kwargs)
        
        if cosmo is None: 
            # default cosmology is Planck2013
            cosmo = Planck13

        assert isinstance(cosmo, Cosmology), "cosmo must be an astropy.cosmology.Cosmology instance"
        self.cosmo = cosmo

        # interpolators for speeding up cosmological calculations 
        self._zred_grid = np.linspace(0.0, 5.0, 1000)
        self._tage_grid = self.cosmo.age(_z).value # t_age 
        self._dlum_grid = self.cosmo.luminosity_distance(_z).to(U.cm).value # luminosity distance in cm
        print('input parameters : %s' % ', '.join(self._parameters))
    
    def sed(self, tt, zred, vdisp=0., wavelength=None, **kwargs):
        ''' compute the redshifted spectral energy distribution (SED) for a
        signle set of parameter values and redshift.
       

        Parameters
        ----------
        tt : 1-d array
            [Nparam] SPS parameters     

        zred : float 
            redshift of the SED 

        vdisp : float
            velocity dispersion  

        wavelength : array_like[Nwave,]
            If you want to use your own wavelength. If specified, the model
            will interpolate the spectra to the specified wavelengths. By
            default, it will use the speculator wavelength
            (Default: None)  

        resolution : array_like[N,Nwave]
            resolution matrix (e.g. DESI data provides a resolution matrix)  

        Returns
        -------
        outwave : [Nsample, Nwave]
            output wavelengths in angstrom. 

        outspec : [Nsample, Nwave]
            the redshifted SED in units of 1e-17 * erg/s/cm^2/Angstrom.
        '''
        tage = np.interp(zred, self._zred_grid, self._tage_grid) # age of the galaxy 
        dlum = np.interp(zred, self._dlum_grid, self._dlum_grid) # luminosity distance
            
        if wavelength is not None: 
            # make sure it's monotonically increasing
            assert np.all(np.diff(wavelength) >=0)

        # get SSP luminosity
        lum_ssp = self._emu(tt, tage)

        # redshift the spectra
        w_z     = wave_rest * (1. + zred)
        flux_z  = lum_ssp * 3.846e50 / (4. * np.pi * d_lum**2) / (1. + zred) # 10^-17 ergs/s/cm^2/Ang

        # apply velocity dispersion 
        if vdisp == 0: 
            wave_smooth = w_z 
            flux_smooth = flux_z
        else: 
            wave_smooth, flux_smooth = self._apply_vdisp(w_z, flux_z, vdisp)

        if wavelength is None: 
            return wave_smooth, flux_smooth 
        else: 
            outwave = wavelength
            
            # rebin flux to input wavelength 
            flux_rebin = np.interp(wavelength, wave_smooth, flux_smooth)
            #resampflux = UT.trapz_rebin(wave_smooth, flux_smooth, xnew=wavelength) # this works but takes forever.
            return wavelength, flux_rebin 
    
    def _init_model(self, **kwargs) : 
        return None 

    def _apply_vdisp(self, wave, flux, vdisp): 
        ''' apply velocity dispersion by first rebinning to log-scale
        wavelength then convolving vdisp. 

        Notes
        -----
        * code lift from https://github.com/desihub/desigal/blob/d67a4350bc38ae42cf18b2db741daa1a32511f8d/py/desigal/nyxgalaxy.py#L773
        * confirmed that it reproduces the velocity dispersion calculations in
        prospector
        (https://github.com/bd-j/prospector/blob/41cbdb7e6a13572baea59b75c6c10100e7c7e212/prospect/utils/smoothing.py#L17)
        '''
        if vdisp <= 0: return wave, flux
        pixkms = 10.0                                 # SSP pixel size [km/s]
        dlogwave = pixkms / 2.998e5 / np.log(10)

        wlog = 10**np.arange(np.log10(wave.min() + 10.), np.log10(wave.max() - 10.), dlogwave)
        flux_wlog = np.interp(wlog, wave, flux)
        #flux_wlog = UT.trapz_rebin(wave, flux, xnew=wlog, edges=None)

        # convolve  
        sigma = vdisp / pixkms # in pixels 
        smoothflux = UT.gaussian_filter1d(flux_wlog, sigma=sigma, axis=0)
        return wlog, smoothflux
    
    def _parse_theta(self, tt):
        ''' parse given array of parameter values 
        '''
        tt = np.atleast_2d(tt.copy()) 

        assert tt.shape[1] == len(self._parameters), 'given theta has %i instead of %i dims' % (tt.shape[1], len(self._parameters))

        theta = {} 
        for i, param in enumerate(self._parameters): 
            theta[param] = tt[:,i]
        return theta 


class ModelB(BaseModel): 
    ''' SPS model with non-parametric star formation and metallicity histories
    and flexible dust attenuation model. The SFH and ZH are based on non-negative
    matrix factorization (NMF) bases (Tojeiro+in prep). The dust attenuation
    uses a standard Charlot & Fall dust model.

    The SFH uses 4 NMF bases. If you specify `burst=True`, the SFH will
    include an additional burst component. 
    
    The ZH uses 2 NMF bases. Minimum metallicities of 4.49e-5 and 4.49e-2 are
    imposed automatically on the ZH. These limits are based on the metallicity
    limits of the MIST isochrones.
    
    The nebular emission is based on the default FSPS implementation by Nell Byler. 

    The dust attenuation is modeled using a 3 parameter Charlot & Fall model.
    `dust1` is tau_BC, the optical depth of dust attenuation of birth cloud
    that only affects young stellar population. `dust2` is tau_ISM, the optical
    depth of dust attenuation from the ISM, which affects all stellar emission.
    `dust_index` is the attenuation curve slope offset from Calzetti.
    
    The dust emission is modeled using the Draine & Li (2007) dust emission model
    with 3 free parameters: Umin, gamma_e, Q_PAH, 
    

    Parameters
    ----------
    cosmo : astropy.comsology object
        specify cosmology. If cosmo=None, uses astropy.cosmology.Planck13 by default.


    Notes 
    -----
    '''
    def __init__(self, cosmo=None): 
        # load wavelengt
        self.wave = np.load()

        # load emulators 
        self._load_emulators()
        
        super().__init__(cosmo=cosmo) # initializes the model

    def _emu(self, theta): 
        ''' PCA neural network emulator of the SED model. 

        Parameters 
        ----------
        theta : 1d array 
            Nparam array that specifies the parameter values. Last element is
            age of the galaxy in Gyr
        
        Returns
        -------
        lum_ssp : array_like[Nwave] 
            FSPS SSP luminosity in units of Lsun/A
    

        Notes
        -----
        '''
        # whiten theta
        wtheta = (theta[1:] - self._avg_thetas) / self._std_thetas
  
        lum_ssp = [] 
        for i_wave in range(4): 
            activations = wtheta
            for w, b, beta, gamma in self._emu_params[:-1]:
                outputs = np.dot(w, activations) + b
                activations = nonlin_act(outputs, beta, gamma) 

            final_w, final_b = self._emu_params
            x_pca = np.dot(final_w, activations) + final_b
            
            # un-whiten x_pca
            x_pca *= self._std_x_pca
            x_pca += self._avg_x_pca
            
            # unwhiten lum_ssp 
            loglum = np.dot(x_pca, M_pca) * self._std_logseds + self._avg_logseds
            lum_ssp.append(10**(loglum) * 10**theta[0])
        return lum_ssp 

    def _load_emulators(self): 
        ''' Load all the emulators and relevant files
        '''
        # load sed model emulator for each wavelength bins 
        self._emu_params = [] 
        self._avg_thetas, self._std_thetas = [], [] 
        self._avg_x_pcas, self._std_x_pcas = [], []
        self._M_pcas = [] 
        self._avg_logseds, self._std_logseds = [], [] 

        for i_wave in range(4): 
            with open('.pkl' % iwave, 'rb') as f:
                self._emu_params.append(pickle.load(f)) 

            # load avg, std of thetas 
            self._avg_thetas.append(np.load())
            self._avg_thetas.append(np.load())
            
            # load avg, std of x_pca 
            self._avg_x_pcas.append(np.load())
            self._std_x_pcas.append(np.load())
            
            # load pca matrix
            self._M_pcas.append(np.load())

            # load avg, std of logsed 
            self._avg_logseds.append(np.load())
            self._std_logseds.append(np.load())
            
        return None 


def nonlin_act(x, beta, gamma):
    ''' non-linear activation function from Alsing et al.(2019)
    '''
    return (gamma + sigmoid(beta * x) * (1 - gamma)) * x

