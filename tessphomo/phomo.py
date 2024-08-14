import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from glob import glob


from matplotlib import patches

#import PRF
from tqdm import tqdm

import lightkurve as lk
from lightkurve.correctors import DesignMatrix


from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astroquery.mast import Catalogs
import astropy.units as u
from astropy.time import Time
from astropy.table import QTable

from sklearn.linear_model import HuberRegressor

from copy import deepcopy

from .mast import get_tic_sources
from .plot import *
from .prf import TPFSceneModeler
from .meta import FFI_STRAP_COLS
from .utils import *




def estimate_offset_gadient_nomore(tpfmodel, tpf, err=None):

    
    gy,gx = np.gradient(tpfmodel, edge_order=1)    

    data = np.vstack((tpf-tpfmodel).reshape(-1, ))
    A = np.vstack( [gx.reshape(-1), gy.reshape(-1), np.ones_like(gx).reshape(-1)] ).T

    if not(err is None):
        A = np.vstack(1./err.reshape(-1)**2.)*A
        data = np.vstack(1./err.reshape(-1)**2.)*data
            
    w = np.linalg.solve( A.T.dot(A) , A.T.dot(data) )

    return w



def estimate_offset_gradient(model, data, err,return_all=False, add_terms=[], err_exponent=2.):
    
    dm_dx, dm_dy = np.gradient(model)
    data = np.vstack(data.ravel())

    
    A = np.vstack([model.ravel(), dm_dx.ravel(), dm_dy.ravel(), np.ones_like(model.ravel())] + add_terms).T

    A = np.vstack(1./err.ravel()**err_exponent)*A
    data = np.vstack(1./err.ravel()**err_exponent)*data

    w = np.linalg.solve( A.T.dot(A) , A.T.dot(data) )

    amp, dx_amp, dy_amp, c = w.T[0]

    if return_all:
        return amp, dx_amp, dy_amp, c
    
    #return dx_amp/amp, dy_amp/amp
    return dx_amp, dy_amp






class TESSTargetPixelModeler(object):

    def __init__(self, TPF, mag_lim=20., input_catalog=None, qflags=[0,1,3,5,7,11,14,15]):

        self.tpf = TPF

        self.tic_id = self.tpf.targetid

        self.quality_bitmask = TPF._hdu[1].data['QUALITY']
        self.qflags = qflags
        
        
        #nan_mask = ~np.isnan(np.sum(np.sum(self.tpf.flux.value, axis=1), axis=1) )
        #nan_mask &= np.array([all(f.ravel()>0.) for f in TPF.flux ] )

        nan_mask = make_quality_mask(self.quality_bitmask, qflags)

        tpfdata =  TPF._hdu[1].data
        time_values= tpfdata['TIME']
        flux_values = tpfdata['FLUX']
        flux_err_values = tpfdata['FLUX_ERR']
        
        self.nan_mask = nan_mask
        self.time = Time(time_values[nan_mask], scale='tdb', format='btjd')
        
        self.cadenceno = np.arange(len(time_values))[nan_mask] + TPF.get_header()['FFIINDEX']


        self.tpf_wcs = self.tpf.wcs
        self.tpf_med_data = np.nanmedian(flux_values[nan_mask], axis=0)
        self.tpf_med_err = np.nanmedian(flux_err_values[nan_mask], axis=0)
        
        self.tpf_flux = flux_values[nan_mask]
        self.tpf_flux_err = flux_err_values[nan_mask]
        
        
        #self.prf = self._get_prfmodel()
        if input_catalog is None:
            self.catalog = self._get_tic_sources(mag_lim=mag_lim)
        else:
            self.catalog=input_catalog

        self.row_ref = 0#self.tpf.hdu[1].header['1CRPX4']-self.tpf_med_data.shape[0]//2
        self.col_ref = 0#self.tpf.hdu[1].header['1CRPX4']-self.tpf_med_data.shape[1]//2

        self.source_tpf_modeler = None
        self.bkg_tpf_modeler = None
        self.allstar_tpf_modeler = None
        self.bestfit_tpfmodel = None

        
    def _get_prfmodel(self, prf_dir=None):

        return PRF.TESS_PRF(self.tpf.camera,self.tpf.ccd,self.tpf.sector,self.tpf.column,self.tpf.row)

    def _get_tic_sources(self, directory=None, mag_lim=20.):
        
        if directory is None:
            try:
                catalogTIC = get_tic_sources(self.tic_id, tpf_shape=self.tpf_med_data.shape)
            except:
                catalogTIC = get_tic_sources(self.tic_id, tpf_shape=self.tpf_med_data.shape)
        
        mag_cut = catalogTIC['Tmag'] < mag_lim
        source_catalog = catalogTIC.loc[mag_cut]

        self.catalog=source_catalog

        return source_catalog


    def _create_input_catalog(self, target_id=None, ):

        

        return 1. 


    def recompute_Tmag_from_gaiadr2(self):

        catalog_gaia = self.catalog
        catalog_gaia['Tmag_orig'] = self.catalog['Tmag'].copy()
        
        bp_rp = catalog_gaia['gaiabp']-catalog_gaia['gaiarp']
        Gmag =  catalog_gaia['GAIAmag']

        tmag_new = Gmag - 0.00522555 * bp_rp**3. + 0.0891337 * bp_rp**2. - 0.633923*bp_rp + 0.0324473

        catalog_gaia.loc[~np.isnan(tmag_new),'Tmag'] = catalog_gaia[~np.isnan(tmag_new)]

        self.catalog = catalog_gaia

        return catalog_gaia


    def generate_bkg_source_model(self, flux_scale=None, **kwargs):

        if self.bkg_tpf_modeler is None:

            self._get_star_scene()
            self.generate_source_model()
            bkg_tpf_modeler = deepcopy(self.allstar_tpf_modeler)

            zp =  self.source_tpf_modeler.zeropoint_mag
            flux_scale = 10**(0.4*(zp-self.catalog['Tmag'][0]))

            source_model = self.source_tpf_modeler.model * flux_scale
            allstar_model =  self.allstar_tpf_modeler.model 

            bkg_tpf_modeler.model = allstar_model - source_model 
            
            self.bkg_tpf_modeler = bkg_tpf_modeler

        bkg_source_tpfmodel = self.bkg_tpf_modeler.interpolate_scene(**kwargs)
        
        return bkg_source_tpfmodel

    
    def generate_source_model(self, flux_scale=None, normalize=True, **kwargs):

        if self.source_tpf_modeler is None:
            star_row_col = self._get_source_row_col()
            star_mags=[20.44]
            source_tpf_modeler = self._generate_tpf_scene_modeler(star_row_col[:1], star_mags, )
            self.source_tpf_modeler = source_tpf_modeler
        
        source_tpfmodel = self.source_tpf_modeler.interpolate_scene(**kwargs)

        if normalize:        
            return source_tpfmodel
        else:
            zp =  self.source_tpf_modeler.zeropoint_mag
            flux_scale = 10**(0.4*(zp-self.catalog['Tmag'][0]))
            return source_tpfmodel * flux_scale

    
    def _get_star_scene(self, **kwargs):

        if self.allstar_tpf_modeler is None:
            print('... building scene model with {} stars'.format(len(self.catalog)))
            star_row_col = self._get_source_row_col()
            star_mags = self.catalog['Tmag'].to_numpy()
            allstar_tpf_modeler = self._generate_tpf_scene_modeler(star_row_col, star_mags, )
            self.allstar_tpf_modeler = allstar_tpf_modeler

        allstar_tpfmodel = self.allstar_tpf_modeler.interpolate_scene(**kwargs)
        
        return allstar_tpfmodel

    
    def _get_source_row_col(self, ):

        # Propagate star positions by proper motions
        refepoch = 2015.5
        referenceyear = Time(refepoch, format='decimalyear', scale='tcb')
        deltayear = (self.time[0] - referenceyear).to(u.year)
        pmra = ((np.nan_to_num(np.asarray(self.catalog.pmRA)) * u.milliarcsecond/u.year) * deltayear ).to(u.deg).value
        pmdec = ((np.nan_to_num(np.asarray(self.catalog.pmDEC)) * u.milliarcsecond/u.year) * deltayear).to(u.deg).value
        #self.catalog.RA_orig += pmra
        #self.catalog.Dec_orig += pmdec
        radecs = np.vstack([self.catalog['RA_orig']+pmra, self.catalog['Dec_orig']+pmdec]).T

        # check for nans in RA_orig/Dec_orig (usually Gaia DR2), replace with generic RA, Dec from TIC 
        bad_radec = np.isnan(radecs[:,0])
        radecs[bad_radec,:] = np.vstack([self.catalog['ra'], self.catalog['dec']]).T[bad_radec,:]
        
        coords = self.tpf_wcs.all_world2pix(radecs, 0)
            
        return coords



    def _generate_tpf_scene_modeler(self, source_xy, source_mags, buffer=5):

        cam = self.tpf.camera
        ccd=self.tpf.ccd
        sector = self.tpf.sector
        ref_col = self.tpf.column
        ref_row=self.tpf.row
        source_rows, source_cols = source_xy.T
        tpfshape = self.tpf.flux.shape[1:]
        
        return TPFSceneModeler(cam, ccd, sector, ref_col, ref_row, source_cols, source_rows, source_mags,
                               tpfshape=tpfshape, buffer_size=buffer )
    
    def _generate_tpf_scene(self, source_xy, source_mags, dx=0, dy=0, buffer=5):

        #scene = np.zeros_like(self.tpf)

        size_x, size_y = self.tpf_med_data.shape
        buffer_size = (size_x+2*buffer, size_y+2*buffer)
        
        scene = np.zeros(buffer_size)

        dx+=buffer
        dy+=buffer

        for i in range(len(source_xy)):

            star_row, star_col = source_xy[i]
            star_mag = source_mags[i]

            try:
                scene += self.prf.locate(star_row-(self.row_ref-dx), star_col-(self.col_ref-dy), buffer_size) * 10.**(-0.4*(star_mag-20.44))
            except ValueError:
                pass

        return scene[buffer:-buffer,buffer:-buffer]


    def estimate_med_offset(self, fit_tpf=True, use_err=True):

        if fit_tpf or (self.bestfit_tpfmodel is None):
            tpfmodel, _, _ = self.fit_med_tpf_model(use_err=use_err, )
        
        else:
            tpfmodel =  self.bestfit_tpfmodel

        dx, dy = estimate_offset_gradient(tpfmodel, self.tpf_med_data, self.tpf_med_err,)

        self.bestfit_dx = dx
        self.bestfit_dy = dy
        
        return dx,dy


    def estimate_offset_coarse(self, dx_range=[-0.5, 0.5], dy_range=[-0.5, 0.5], step=0.1, **kwargs):

        dys = np.arange(dy_range[0], dy_range[1], step)
        dxs = np.arange(dx_range[0], dx_range[1], step)

        offsets = np.stack(np.meshgrid(dxs, dys)).T.reshape(-1, 2)

        # Set up Data
        err = np.vstack(self.tpf_med_err.reshape(-1, ) )
        data = np.vstack(self.tpf_med_data.reshape(-1, ) )
        data_err = data*np.vstack(1./err.reshape(-1)**2.)

        chi2_values = []
        
        for dx,dy in offsets:
            # Linear Algebra Least Squares Fitting

            star_tpf_model = self._get_star_scene(dx=dx, dy=dy, **kwargs)
            
            A = np.vstack([star_tpf_model.reshape(-1), np.ones_like(star_tpf_model).reshape(-1)]).T
            A_err = A*np.vstack(1./err.reshape(-1)**2.)
            
            w = np.linalg.solve( A.T.dot(A) , A.T.dot(data) )

            chi2_values.append( np.sum((A.dot(w) - data)/err**2.) )


        best_dx, best_dy = offsets[np.argmin(chi2_values)] 
        
        return best_dx, best_dy


    def fit_med_tpf_model(self, use_err=True, assume_constant_bkg_stars=True, **kwargs):
        
        star_tpf_model = self._get_star_scene(**kwargs)

        data = np.vstack(self.tpf_med_data.ravel( ) )

        bkg_terms =  self._get_bkg_model_terms()
        
        A_0 = np.vstack([star_tpf_model.ravel() ] + bkg_terms ).T

        if use_err:
            err = self.tpf_med_err
            A = np.vstack(1./err.ravel()**2.)*A_0
            data = np.vstack(1./err.ravel()**2.)*data
        else:
            A=A_0
            
        w = np.linalg.solve( A.T.dot(A) , A.T.dot(data) )
        
        flux_scale_factor, bkg_flux = w.T[0, :2]

        fit_tpf_model = star_tpf_model*flux_scale_factor+bkg_flux

        self.bestfit_tpfmodel = fit_tpf_model
        self.bestfit_flux_scale = flux_scale_factor
        self.bestfit_bkg_flux = bkg_flux
        self.bestfit_med_tpfmodel = A_0.dot(w).reshape(star_tpf_model.shape)

        return fit_tpf_model, flux_scale_factor, bkg_flux



    def get_contamination_ratio(self, fit_tpf=True, use_err=True, aperture=None, **kwargs):

        if aperture is None:
            aperture = self._get_aperture()

        if fit_tpf:
            self.fit_tpf_model(use_err=use_err, **kwargs)
    
        source_tpf = self.generate_source_model(**kwargs)
        contam_tpf = self.generate_bkg_source_model(**kwargs)

        bkg_ap_flux = self.bestfit_bkg_flux * np.sum(aperture)
        total_sum_flux = np.sum( aperture * (self.bestfit_tpfmodel-self.bestfit_bkg_flux) )
        source_sum_flux = np.sum(aperture * source_tpf )
        contam_sum_flux =np.sum(  aperture * contam_tpf )
        
        return {'crowdsap': contam_sum_flux/total_sum_flux,
                'flfrcsap': source_sum_flux/np.sum(source_tpf),
                'dilution': total_sum_flux/source_sum_flux, 
                'med_tpf_bkg_aperture_flux':bkg_ap_flux, 
                'tess_zeropoint_mag': 20.44-2.5*np.log10(self.bestfit_flux_scale) }   



    def plot_tpf_model(self, plot_color='C1', logscale=True, vmin=None, vmax=None, plot_bkg_stars=False,**kwargs):

        star_rowcol = self._get_source_row_col()
        star_mags = self.catalog['Tmag'].to_numpy()


        fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(8,4) , constrained_layout=True,sharex=True, sharey=True)

        if vmin is None:
            vmin = np.min(np.abs(self.bestfit_tpfmodel) )
        if vmax is None:
            vmax = np.max(self.bestfit_tpfmodel)

        if logscale:
        
            cax1=ax1.imshow(self.tpf_med_data, origin='lower', norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax), )
            cax2=ax2.imshow(self.bestfit_med_tpfmodel, origin='lower', norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmax), )
        else:
            cax1=ax1.imshow(self.tpf_med_data, origin='lower', vmin=vmin,vmax=vmax, )
            cax2=ax2.imshow(self.bestfit_med_tpfmodel, origin='lower', vmin=vmin, vmax=vmax )

        tpf_resids = (self.tpf_med_data-self.bestfit_med_tpfmodel)
        max_resid = np.max(np.abs(tpf_resids)) 
        cax3=ax3.imshow(tpf_resids, origin='lower', cmap='RdYlBu', vmin=-max_resid, vmax=max_resid)
            
        
        plt.colorbar(ax=[ax1,ax2], mappable=cax1, location='bottom', label='Flux [e-/sec]', shrink=0.5 )
        #plt.colorbar(ax=ax2, mappable=cax2, location='bottom', label='Flux [e-/sec]')
        plt.colorbar(ax=ax3, mappable=cax3, location='bottom', label='Residual [e-/sec]')
        
        
        for ax in (ax1,ax2,ax3):
            ax.set_xticks([])
            ax.set_yticks([])

            if plot_bkg_stars:
                ax.scatter(star_rowcol.T[0]-(self.row_ref), star_rowcol.T[1]-self.col_ref, s=(star_mags-20.44)**2., marker='*', edgecolor='0.5', color='w' , )
            ax.scatter(star_rowcol.T[0][0]-(self.row_ref), star_rowcol.T[1][0]-self.col_ref, s=(star_mags[0]-20.44)**2., marker='*', edgecolor=plot_color, color='w' , zorder=5)
            plot_aperture(ax=ax, aperture_mask=self.get_optimal_aperture(**kwargs), mask_color=plot_color)
            plot_ne_arrow(ax=ax, x_0=self.tpf_med_data.shape[0]*0.15, y_0=self.tpf_med_data.shape[0]*0.8, 
                          len_pix=self.tpf_med_data.shape[0]*0.1, wcs=self.tpf_wcs)
        
        ax1.set_title('Median TPF')
        ax2.set_title('Model TPF')
        ax3.set_title('Data$-$Model')
        ax1.set_xlim(-0.5, self.tpf_med_data.shape[0]-0.5)
        ax1.set_ylim(-0.5, self.tpf_med_data.shape[1]-0.5)

        #plt.suptitle('{}: Sector {}'.format(self.target_id, self.sector))

        return ax1,ax2,ax3


    def get_optimal_aperture(self, n_min_pix=2, n_max_pix=80, save_aperture=True, source_flux_scale=1., **kwargs):

        if self.bestfit_tpfmodel is None:
            basemodel, flux_scale, bkg_flux = self.fit_med_tpf_model( )
            
        else:
            flux_scale =  self.bestfit_flux_scale
            bkg_flux = self.bestfit_bkg_flux

            
        contam_tpf = self.generate_bkg_source_model(**kwargs).ravel() 
        source_tpf = self.generate_source_model(normalize=False, **kwargs).ravel()  * source_flux_scale
        bkg_tpf = bkg_flux * np.ones_like(contam_tpf)


        texp = (self.tpf._hdu[1].header['EXPOSURE'] * u.day).to(u.second).value
        source_tpf *= texp
        contam_tpf *= texp
        bkg_tpf *= texp
        
        #gebkg_tpf = (np.zeros_like(contam_tpf)+self.bestfit_bkg_flux).ravel()

        
        #cumulative_snr = source_tpf / np.sqrt(source_tpf+contam_tpf+bkg_tpf)
        sorted_pixels = np.argsort(source_tpf)[::-1]

        cumulative_snr = np.cumsum(source_tpf[sorted_pixels]) / np.sqrt( np.cumsum(source_tpf[sorted_pixels] + contam_tpf[sorted_pixels] + bkg_tpf[sorted_pixels], ) )

        n_good_pixels_snr = np.argmax(cumulative_snr) + 2

    
        n_good_pixels_snr = min(n_good_pixels_snr, n_max_pix)
        n_good_pixels_snr = max(n_good_pixels_snr, n_min_pix)

        
        aperture_mask = np.zeros_like(source_tpf).astype(bool)

        n_good_pixels_snr=int(n_good_pixels_snr)
        aperture_mask[sorted_pixels[:n_good_pixels_snr]] = True
        

        if save_aperture:
            self.best_aperture = aperture_mask.reshape(self.tpf_flux.shape[1:])

        #bkg_aperture = ((source_tpf+contam_tpf) / np.sqrt(source_tpf+contam_tpf+bkg_tpf)) < 1.
        
        return aperture_mask.reshape(self.tpf_flux.shape[1:])




    def get_prf_xy_timeseries(self, use_err=True, err_exponent=2., mask_source=True, **bkg_model_terms):

        if self.bestfit_tpfmodel is None:
            basemodel, _, _ = self.fit_med_tpf_model(use_err=True, )
        else:
            basemodel =  self.bestfit_tpfmodel


        gx, gy = np.gradient(basemodel)
        
        bkg_terms = self._get_bkg_model_terms(**bkg_model_terms)

        model_terms = [gx.ravel(), gy.ravel(), basemodel.ravel(), ] + bkg_terms
        
        tpf_fluxes = self.tpf_flux.copy()
        tpf_flux_errs = self.tpf_flux_err.copy()

        
        if mask_source:
            source_mask = self.get_optimal_aperture(n_min_pix=5., source_flux_scale=10., save_aperture=False)
            tpf_flux_errs[:,source_mask] = np.inf

        ws = [matrix_solve(model_terms, tpf_fluxes[i], tpf_flux_errs[i], power=err_exponent) for i in range(tpf_fluxes.shape[0])]

        dx, dy = -np.array(ws).T[0, :2, :]
        self.prf_dx, self.prf_dy = dx, dy
        
        return dx, dy



    def _get_bkg_model_terms(self, model_straps=True, gradient=True, bkg_poly_order=2):

        tpfshape = self.tpf.flux.value.shape[1:]
        bkg_terms = []
        bkg_terms.append(np.ones(tpfshape).ravel())

        if gradient:

            dx, dy = np.mgrid[:tpfshape[0], :tpfshape[1]]
            dx_ravel = dx.ravel()
            dy_ravel = dy.ravel()
        
            for i in range(1,bkg_poly_order+1):
                for j in range(0,i+1):
                    bkg_terms.append( dx_ravel**j * dy_ravel**(i-j))
            
        if model_straps:

            col_0 = self.tpf.column
            for i in range(tpfshape[0]):

                if np.isin(i+col_0, FFI_STRAP_COLS):

                    col_i = np.zeros(tpfshape)
                    col_i[:, i] = np.ones(tpfshape[1])
                    bkg_terms.append(col_i.ravel())

        return bkg_terms



    def get_cap_flux_timeseries(self, aperture=None, progress=True, **kwargs):

        if aperture is None:
            best_aperture = self.get_optimal_aperture(**kwargs)
        else:
            best_aperture=aperture

        try:
            dx_t, dy_t = self.prf_dx, self.prf_dy
        except:
            dxdt =  self.get_prf_xy_timeseries(use_err=True, **kwargs)
            dx_t, dy_t = np.array(dxdt).T



        sapflux_timeseries = np.array([])
        flfrc_sapflux_timeseries = np.array([])
        contam_sapflux_timeseries = np.array([])
        bkg_sapflux_timeseries = np.array([])
       
        
        #flux_scale_factor, bkg_flux = w.T[0]
        #fit_tpf_model = star_tpf_model*flux_scale_factor+bkg_flux

        
        tpf_fluxes = self.tpf_flux
        tpf_flux_errs = self.tpf_flux_err
        
        #(sap_flux - np.median(sap_flux)*crowding['crowdsap']) / crowding['flfrcsap']

        bkg_flux_array = np.ones_like(tpf_fluxes[0]).ravel()

        if progress:
            from tqdm import tqdm
            iterable = tqdm(np.arange(len(dx_t) ).astype(int) )
        else:
            iterable = np.arange(len(dx_t) ).astype(int)

        for i in iterable:                

            source_tpf = self.generate_source_model(dx=dx_t[i], dy=dy_t[i],)
            contam_tpf = self.generate_bkg_source_model(dx=dx_t[i], dy=dy_t[i],)
            allstar_tpf = source_tpf+contam_tpf

            contam_flux = np.sum(contam_tpf*best_aperture) #/ np.sum(allstar_tpf*best_aperture)
            flux_frac = np.sum(source_tpf*best_aperture) / np.sum(source_tpf)

            #print(sap_flux_i, contam_frac, flux_frac)

            # Fit for the Bkg Flux
            data = np.vstack(1./tpf_flux_errs[i].ravel()**2.)*np.vstack(tpf_fluxes[i].ravel())
            A = np.vstack(1./tpf_flux_errs[i].reshape(-1)**2.)*np.vstack([allstar_tpf.reshape(-1), bkg_flux_array]).T
            w = np.linalg.solve( A.T.dot(A) , A.T.dot(data) )

            #print(w)
            
            zero_flux_i, bkg_i = w.T[0]

            sap_flux_i = np.sum(tpf_fluxes[i][best_aperture])

            sap_flux_bkg_sub = sap_flux_i - np.sum(best_aperture) * bkg_i
            sap_flux_decrowd = (sap_flux_bkg_sub - contam_flux)/flux_frac

            #sap_flux_frac_corr = sap_flux_decrowd/flux_frac

            sapflux_timeseries = np.append(sapflux_timeseries, sap_flux_decrowd)
            flfrc_sapflux_timeseries = np.append(flfrc_sapflux_timeseries, flux_frac)
            contam_sapflux_timeseries = np.append(contam_sapflux_timeseries, contam_flux)
        

        return sapflux_timeseries , flfrc_sapflux_timeseries, contam_sapflux_timeseries




    def frame_solve(self, frame, bkg_terms,  dx=0, dy=0,):

        source = self.generate_source_model(dx=dx,dy=dy)
        #source/=np.sum(source)
        bkg_stars = self.generate_bkg_source_model(dx=dx,dy=dy)

        #bkg = self._get_bkg_model_terms(**bkg_kwargs)
            
        A = np.vstack([source.ravel(),bkg_stars.ravel()] + bkg_terms).T

        return np.linalg.solve(A.T.dot(A), A.T.dot(frame.ravel()))



    def _fit_prf_flux(self, data, source_scene, crowd_scene, bkg_terms, data_err=None, power=2.):

        model = [[source_scene.ravel(), crowd_scene.ravel()] + bkg_terms]

        w = matrix_solve(model, data, data_err, power=power)

        return w

    def _calc_aperture_flux(self, aperture, data, source_scene, crowd_scene, bkg_scene):

        sap_flux = np.sum(data[aperture])
        flfrcsap = np.sum(source_scene[aperture])/np.sum(source_scene[1:-1,1:-1])
        crowdsap = np.sum(crowd_scene[aperture])
        bkgsap =  np.sum(bkg_scene[aperture])
        
        return sap_flux, flfrcsap, crowdsap, bkgsap 
        

    def get_prf_flux_timeseries(self,  progress=False,  **kwargs):
        
        #all_stars = self._get_star_scene()
        bkg_terms = self._get_bkg_model_terms(**kwargs)

        y=self.tpf_flux

        try:
            dx_t, dy_t = self.prf_dx, self.prf_dy
        except:
            dx_t, dy_t = self.get_prf_xy_timeseries()

        nan_mask = np.isnan(dx_t)

        dx_t[nan_mask] = np.nanmedian(dx_t)
        dy_t[nan_mask] = np.nanmedian(dy_t)
    
        if progress:
            from tqdm import tqdm
            iterable = tqdm(np.arange(len(y) ).astype(int) )
        else:
            iterable = np.arange(len(y) ).astype(int)

        
        ws = np.asarray([self.frame_solve(y[i], bkg_terms, dx_t[i], dy_t[i],) for i in iterable])
        
        #for i, frame in iterable:
            
        #    source = self.generate_source_model(dx=dx_t[i],dy=dy_t[i])
        #    bkg_stars = self.generate_bkg_source_model(dx=dx_t[i],dy=dy_t[i])
            
        #    A = np.vstack([source.ravel(),bkg_stars.ravel(),          
        #        bkg.ravel()]).T

        #    ws_i = np.linalg.solve(A.T.dot(A), A.T.dot(frame.ravel()))

        #    ws = np.append(ws, ws_i)
            

        #model = np.asarray([A[:, :-1].dot(w[:-1]).reshape(y.shape[1:]) for w in ws])self

        prf_flux, zero_point_flux, bkg_flux = ws.T[:3]

        return prf_flux, zero_point_flux, bkg_flux, dx_t, dy_t



    def get_corrected_LightCurve(self, err_exponent=1.0, aperture=None, bad_data_mask=None, assume_catalog_mag=False, recompute_scene_motion=False, progress=True,  **kwargs):

        '''

        Extract PRF Photometry and Aperture photometry from the TPF. 

        
        '''


        if aperture is None:
            best_aperture = self.get_optimal_aperture()
        else:
            best_aperture=aperture

        all_stars = self._get_star_scene()
        bkg_terms = self._get_bkg_model_terms(**kwargs)        

        if recompute_scene_motion:
            dx_t, dy_t = self.get_prf_xy_timeseries(err_exponent=err_exponent, **kwargs)

        else:            
            try:
                dx_t, dy_t = self.prf_dx, self.prf_dy
            except:
                dx_t, dy_t = self.get_prf_xy_timeseries(err_exponent=err_exponent, **kwargs)


        if bad_data_mask is None:
            mask = np.isfinite(dx_t)
        else:
            mask=bad_data_mask
            mask &= np.isfinite(dx_t)
        

        if progress:
            iterable = tqdm(np.arange(len(dx_t) ).astype(int) )
        else:
            iterable = np.arange(len(dx_t) ).astype(int)


        y=self.tpf_flux
        y_err = self.tpf_flux_err
        raw_results=[]

        systematics = []
        
        for i in iterable:

            if not(mask[i]):
                raw_results.append([np.nan]*8)
                continue
                
                

            source_scene =  self.generate_source_model(dx=dx_t[i],dy=dy_t[i], normalize=True)
            crowd_scene = self.generate_bkg_source_model(dx=dx_t[i],dy=dy_t[i], )

            scene_model_terms = [source_scene.ravel(), crowd_scene.ravel()]+bkg_terms


            prf_fit_weights = self._fit_prf_flux(data=y[i], source_scene=source_scene,
                                             crowd_scene=crowd_scene, bkg_terms=bkg_terms,
                                             data_err=y_err[i], power=err_exponent)

            systematics.append(prf_fit_weights.T[0][1:])


            #print(prf_fit_weights.shape, prf_fit_weights.T)

            prf_flux_i, zero_point_flux_i = prf_fit_weights.T[0][:2]
            bkg_weights =  prf_fit_weights.T[0][2:]

            bkg_scene = np.dot(np.transpose(bkg_terms), np.vstack(bkg_weights)).reshape(y[i].shape)

            sap_flux_i, flfrcsap_i, crowdsap_i, bkgsap_i = self._calc_aperture_flux(aperture=best_aperture,
                                                                           data=y[i],source_scene=source_scene,
                                                                                    crowd_scene=crowd_scene*zero_point_flux_i,
                                                                            bkg_scene=bkg_scene)

            sapflux_err_i = np.sqrt(np.sum((y_err[i][best_aperture])**2. ) )

            scene = np.dot(np.transpose(scene_model_terms), np.vstack(prf_fit_weights)).reshape(y[i].shape)
            scene_chi2_i = np.sum((scene-y[i])**2./y_err[i]**2.)/len(bkg_terms[0])

            
            raw_results.append([prf_flux_i, zero_point_flux_i, sap_flux_i, flfrcsap_i, crowdsap_i, bkgsap_i, sapflux_err_i, scene_chi2_i ])
            

        #print(scene_chi2_i)

        #print( np.array(raw_results)[0] )



        raw_prf_flux, zp_flux, sap_flux, flfrcsap, crowdsap, bkgsap, sapflux_err, scene_chi2 = np.array(raw_results).T
        
        all_systematics = np.concatenate([np.array(systematics).T, [dx_t, dy_t]])

        raw_cap_flux = sap_flux - (crowdsap + bkgsap)
        raw_cap_flux /= flfrcsap

         ## Check for bad data points
        mask &= zp_flux!=0
        mask &= np.isfinite(zp_flux)
        mask &= np.isfinite(bkgsap)
        mask &= np.isfinite(dx_t)
        mask &= np.isfinite(dy_t)

        

        #corr_prf_flux_masked, prf_instrument_model = correct_flux(raw_prf_flux[mask], flux_err=sapflux_err[mask],systematics=[zp_flux[mask],bkgsap[mask],dx_t[mask],dy_t[mask]], assume_catalog_mag=assume_catalog_mag, mag=float(self.catalog['Tmag'][0]) )
        #corr_cap_flux_masked, cap_instrument_model = correct_flux(raw_cap_flux[mask],flux_err= sapflux_err[mask],systematics=[zp_flux[mask], bkgsap[mask], dx_t[mask], dy_t[mask]],  assume_catalog_mag=assume_catalog_mag, mag=float(self.catalog['Tmag'][0]))

                                

        corr_prf_flux_masked, prf_instrument_model = correct_flux(raw_prf_flux[mask], flux_err=sapflux_err[mask],
                                                                  systematics=all_systematics[:,mask],
                                                                  assume_catalog_mag=assume_catalog_mag, mag=float(self.catalog['Tmag'][0]) )
        corr_cap_flux_masked, cap_instrument_model = correct_flux(raw_cap_flux[mask],flux_err= sapflux_err[mask],
                                                                 systematics=all_systematics[:,mask],
                                                                 assume_catalog_mag=assume_catalog_mag, mag=float(self.catalog['Tmag'][0]))
                                                                  
        
        
        corr_prf_flux = np.ones_like(mask)*np.nan
        corr_cap_flux = np.ones_like(mask)*np.nan

        corr_prf_flux[mask] = corr_prf_flux_masked
        corr_cap_flux[mask] = corr_cap_flux_masked
        
        flux_unit=self.tpf.flux.unit
        
        lc_table = QTable([self.cadenceno, self.time, corr_prf_flux*flux_unit, corr_cap_flux*flux_unit, raw_prf_flux*flux_unit, raw_cap_flux*flux_unit, sap_flux*flux_unit, zp_flux, flfrcsap, crowdsap*flux_unit, bkgsap*flux_unit, sapflux_err*flux_unit, dx_t, dy_t, scene_chi2, mask],
           names=('cadenceno','time', 'cal_prf_flux', 'cal_cap_flux', 'raw_prf_flux', 'raw_cap_flux', 'sapflux', 'zp_flux_scale', 'flfrcsap','crowd_sapflux','bkg_sapflux', 'sapflux_err','col_offset', 'row_offset', 'scene_chi2','bad_prf_mask'), )
        
        
        return lc_table




    def get_deblended_PRF_lightcurves_OLD(self, min_sep=0.5, mag_lim = 17., ):

        allstar_mags = self.catalog['Tmag'].to_numpy()
        allstar_xy = self._get_source_row_col()
        ticids = self.catalog['ID']

        try:
            dx_t, dy_t = self.prf_dx, self.prf_dy
        except:
            dx_t, dy_t = self.get_prf_xy_timeseries()

        
        Tmag_0 = allstar_mags[0]


        deblended_sources = allstar_mags<=mag_lim
        deblended_sources &= np.min(allstar_xy, axis=1)>=0
        deblended_sources &= np.max(allstar_xy, axis=1)<self.tpf.flux.shape[1]

        indices = np.arange(len(deblended_sources)).astype(int)

        for s in indices:

            s_mag = allstar_mags[s]
            x_s, y_s = allstar_xy[s]
        
            star_distances = distance((x_s, y_s), allstar_xy)

            dist_cut = star_distances<min_sep
            nearby_mags = allstar_mags[dist_cut]
            nearby_dists = star_distances[dist_cut]

            for s_i in indices[dist_cut]:

                if s==s_i:
                    continue

            if (allstar_mags[s]>allstar_mags[s_i]).any():
                deblended_sources[s]=False
            
        
        bkg_mags = allstar_mags[~deblended_sources]
        bkg_xy = allstar_xy[~deblended_sources]

        
        source_ticids = ticids.to_numpy()[deblended_sources].astype(int)    
        source_mags = allstar_mags[deblended_sources]
        source_xy = allstar_xy[deblended_sources]


        bkg_scene_modeler = self._generate_tpf_scene_modeler(bkg_xy, bkg_mags, )
        source_star_modelers = [self._generate_tpf_scene_modeler(source_xy[i:i+1], np.array([20.44]), ) for i in tqdm(range(len(source_mags)) )]
    
        bkg_model_terms = self._get_bkg_model_terms(gradient=True, model_straps=True)

        ws = []
    
        for i,frame in tqdm(enumerate(self.tpf_flux)):

            bkg_star_model = bkg_scene_modeler.interpolate_scene(dx=dx_t[i], dy=dy_t[i]).ravel( )
            source_star_models = [mod.interpolate_scene(dx=dx_t[i], dy=dy_t[i]).ravel( ) for mod in source_star_modelers]

            A = np.vstack(source_star_models + [bkg_star_model] + bkg_model_terms).T        
            w_i = np.linalg.solve(A.T.dot(A), A.T.dot(frame.ravel()))   

            ws.append(w_i)
        

        return source_mags, source_xy, source_ticids, ws



    def get_deblended_PRF_lightcurves(self, source_ids=[], err_exponent=2., model_straps=True, bkg_poly_order=1, assume_catalog_mag=True, recompute_scene_motion=False, progress=True, bad_data_mask=None):


        allstar_mags = self.catalog['Tmag'].to_numpy()
        allstar_xy = self._get_source_row_col()

        allstar_ids = self.catalog['ID'].to_numpy().astype(int)

        deblended_sources = np.isin(allstar_ids, source_ids)


        source_mags = allstar_mags[deblended_sources]
        source_xy = allstar_xy[deblended_sources]

        bkg_mags = allstar_mags[~deblended_sources]
        bkg_xy = allstar_xy[~deblended_sources]

        deblended_ids = np.array(allstar_ids)[deblended_sources]

        bkg_scene_modeler = self._generate_tpf_scene_modeler(bkg_xy, bkg_mags, )
        source_star_modelers = [self._generate_tpf_scene_modeler(source_xy[i:i+1], np.array([20.44]), ) for i in range(len(source_mags)) ]
    
        bkg_model_terms = self._get_bkg_model_terms(gradient=True, model_straps=model_straps, bkg_poly_order=bkg_poly_order)

        
        # Determine the pointing model
        if recompute_scene_motion:
            dx_t, dy_t = self.get_prf_xy_timeseries(err_exponent=err_exponent, model_straps=model_straps, bkg_poly_order=bkg_poly_order)

        else:            
            try:
                dx_t, dy_t = self.prf_dx, self.prf_dy
            except:
                dx_t, dy_t = self.get_prf_xy_timeseries(err_exponent=err_exponent, model_straps=model_straps, bkg_poly_order=bkg_poly_order)


        # Use a bad Data mask or not
        if bad_data_mask is None:
            mask = np.isfinite(dx_t)
        else:
            mask=bad_data_mask
            mask &= np.isfinite(dx_t)

        
        # Whether or not to show a progress bar
        if progress:
            iterable = tqdm(range(len(self.tpf_flux)))
        else:
            iterable = range(len(self.tpf_flux))

        y=self.tpf_flux
        y_err = self.tpf_flux_err
        raw_results=[]

    
        for i in iterable:

            try:
                frame = self.tpf_flux[i]
                frame_err = self.tpf_flux_err[i]
            
                bkg_star_model = bkg_scene_modeler.interpolate_scene(dx=dx_t[i], dy=dy_t[i]).ravel( )
                source_star_models = [mod.interpolate_scene(dx=dx_t[i], dy=dy_t[i]).ravel( ) for mod in source_star_modelers]
            
                model = source_star_models + [bkg_star_model] + bkg_model_terms
                w_i = matrix_solve(model, frame, frame_err, power=err_exponent)
                
            except:
                w_i = [np.nan]*(A.shape[1])

            raw_results.append(w_i)


        raw_fluxes = np.array(raw_results).T[0][:len(source_ids)]
        systematics = np.concatenate([[dx_t, dy_t], np.array(raw_results).T[0][len(source_ids):] ])
    
        results = {}
        

        model_keys = ['col_offset', 'row_offset', 'zp_flux_scale'] + ['bkg_poly_'+'{}'.format(i).zfill(2) for i in range(3,len(systematics)) ]
        results['model_terms'] = {key:systematics[i] for i,key in enumerate(model_keys)}

        results['time'] = self.time
        results['source_ids'] = source_ids


        ## Check for bad data points
        mask &= np.isfinite(systematics[0])
        mask &= np.isfinite(dx_t)
        mask &= np.isfinite(dy_t)

        for i,ticid in enumerate(source_ids):

            f_i = raw_fluxes[i]

            corr_flux, systematics_model = correct_flux(f_i, systematics,assume_catalog_mag = assume_catalog_mag, mag=source_mags[i])
        
            results[ticid] = {'row':source_xy[i][1], 'col':source_xy[i][0], 'mag':source_mags[i],
                              'raw_prf_flux':raw_fluxes[i], 'cal_prf_flux':corr_flux, 'systematics_model':systematics_model }
    
        return results


    


def correct_flux(raw_flux, systematics, flux_err=None, do_pca=False, nterms=4, err_exponent=2., assume_catalog_mag=False, mag=None):

        
    dm = DesignMatrix(np.vstack(systematics).T)

    if do_pca:
        dm = dm.pca(nterms).append_constant()
    else:
        dm = dm.standardize().append_constant()

    X = dm.X

    if not(flux_err is None):
        Xw = X.T.dot(X / np.vstack(flux_err) ** err_exponent)
        B = np.dot(X.T, raw_flux/(flux_err**err_exponent) )
    else:
        Xw=X.T.dot(X)
        B=np.dot(X.T, raw_flux)

    w = np.linalg.solve(Xw, B).T

    system_model = X.dot(w)

    if assume_catalog_mag and not(mag is None):
        f_0 = mag_to_flux(mag, )
        corr_flux = (raw_flux - system_model) + f_0

    else:
        f_0 = w[-1]
        corr_flux = f_0 * (raw_flux/system_model)
    
    return corr_flux, system_model
        
def distance(x, coords):
    return np.sqrt( (x[0]-coords[:,0])**2. + (x[1]-coords[:,1])**2. )










