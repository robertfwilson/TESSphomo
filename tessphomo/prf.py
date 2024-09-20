import PRF
import numpy as np

from glob import glob
from astropy.io import fits

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.signal import fftconvolve, convolve
import os 

from .meta import TESS_ZEROPOINT_MAG

PKG_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PRF_FILE_DIRECTORY = PKG_DIRECTORY + '/data/mastprf_fitsfiles/'



class TESS_PRF(object):

    def __init__(self, cam, ccd, sector, column, row, localdatadir = PRF_FILE_DIRECTORY, **trim_kw):

        self.cam, self.ccd, self.sector, self.column, self.row = cam, ccd, sector, column, row
        self.oversamp = 9 #samples/pixel for TESS PRFs

        self.prfdata = self._load_prf_data(localdatadir)

        self.prfdata_row_index = (np.arange(self.prfdata.shape[0])-self.prfdata.shape[0]/2.)/self.oversamp 
        self.prfdata_col_index = (np.arange(self.prfdata.shape[1])-self.prfdata.shape[1]/2.)/self.oversamp

        self.prf_model = self._calculate_prf_model(**trim_kw)

                

    def _load_prf_data(self, localdatadir=None):

        if self.sector < 4:
            localdatadir = localdatadir + 'start_s0001/'
        else:
            localdatadir = localdatadir + 'start_s0004/'


        '''
        Below code repurposed from https://github.com/keatonb/TESS_PRF/blob/main/src/PRF/prf.py, 
        Credit belongs to Keaton Bell.  
        
        '''
        #Get PRF file info
        subdir = f'cam{int(self.cam)}_ccd{int(self.ccd)}/'
        filelist = None #local and online options
        

        filelist = glob(os.path.join(localdatadir, subdir) + '*.fits')
        
        #One directory on MAST has some errant files with `phot` in filename
        filelist = [file for file in filelist if 'phot' not in file]

        cols = np.array([int(file[-9:-5]) for file in filelist])
        rows = np.array([int(file[-17:-13]) for file in filelist])

        rownum=self.row
        colnum=self.column

        #Bilinear interpolation between four surrounding PRFs
        LL = np.where((rows < rownum) & (cols < colnum))[0] #lower left
        LR = np.where((rows > rownum) & (cols < colnum))[0] #lower right
        UL = np.where((rows < rownum) & (cols > colnum))[0] #upper left
        UR = np.where((rows > rownum) & (cols > colnum))[0] #uppper right
        dist = np.sqrt((rows-rownum)**2. + (cols-colnum)**2.)

        # Find the 4 closest inds
        surroundinginds = [subset[np.argmin(dist[subset])] for subset in [LL,LR,UL,UR]]
        
        #Following https://stackoverflow.com/a/8662355
        points = []
        for ind in surroundinginds:
            hdulist = fits.open(filelist[ind])
            prf = hdulist[0].data
            points.append((cols[ind],rows[ind],prf))
            hdulist.close()

        points = sorted(points)

        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        prfdata = (q11 * (x2 - colnum) * (y2 - rownum) +
                    q21 * (colnum - x1) * (y2 -  rownum) +
                    q12 * (x2 - colnum) * ( rownum - y1) +
                    q22 * (colnum - x1) * ( rownum - y1)
                    ) / ((x2 - x1) * (y2 - y1) + 0.0)

        '''
        Above code repurposed from https://github.com/keatonb/TESS_PRF/blob/main/src/PRF/prf.py, 
        Credit belongs to Keaton Bell.  
        
        '''

        return prfdata



    def _calculate_prf_model(self, extrapolate_wings=True, max_radius=6., nsigma=5.):

        '''
        Take loaded prfdata and calulate an interpolatable modeluse it to define a splione model that can be interpolated
        
        '''

        prfdata = self.prfdata.copy()
        prf_supsampled_row_inds = self.prfdata_row_index
        prf_supsampled_col_inds = self.prfdata_col_index

        
        prfmodel_data = trim_prf_model(prfdata, nsigma=nsigma, extrapolate_wings=extrapolate_wings, max_radius=max_radius, )

        # Make last columns/rows = 0 
        prfmodel_data[[0,-1],:]=0.
        prfmodel_data[:,[0,-1]]=0.
        interp_prf_model =  RectBivariateSpline(prf_supsampled_col_inds, prf_supsampled_row_inds, prfmodel_data, 
                                                  kx=1, ky=1, s=0, )

        self.prf_model = interp_prf_model

        return interp_prf_model



    def interpolate(self, row, col, row_index=None, col_index=None, tpf_shape=(15,15)):        

        if col_index is None:
            col_index = np.arange(tpf_shape[1])
        if row_index is None:
            row_index = np.arange(tpf_shape[0])

        
        return self.prf_model(row_index-row, col_index-col,  )    



    

class TESS_PRF_Model(object):

    def __init__(self, camera, ccd, sector, column, row, localdatadir = PRF_FILE_DIRECTORY):

        if sector <4:
            localdatadir = localdatadir + 'start_s0001/'
        else:
            localdatadir = localdatadir + 'start_s0004/'
        #print(localdatadir)
        self.prf = PRF.TESS_PRF(camera, ccd, sector, column, row, localdatadir = localdatadir)
        self.model = self.prf.reshaped

        
    def _interp(self, col, row, flux, tpf_size=(15,15), trim_prf_model=False, renormalize=True):

        prf_model = self.model
        
        x=col
        y=row
        
        os_factor = prf_model.shape[0]
        os_frac = 1./os_factor
    
        prf_size = prf_model.shape[2:]

        #if trim_prf_model:
        #    prf_model = np.roll(prf_model, shift=(-prf_size[0]//2+size[0], -prf_size[1]//2+size[1]), axis=(2,3) )[:,:,:2*size[0], :2*size[1]]


        ###############################################################
        ## NEXT ~60 Lines of Code adapted from
        ## https://github.com/keatonb/TESS_PRF/blob/main/src/PRF/prf.py
        ###############################################################
        
        # Integers pixels to define Position on TPF
        colint = np.floor(col-0.)
        rowint = np.floor(row-0.)

        # Fractional Pixels to select PRF subsampled model
        colfrac = (col-0.) % 1
        rowfrac = (row-0.) % 1


        # Define Locations where pixels are sampled in PRF model
        #pixel_samples = np.arange(-os_frac/2., 1.+os_frac/2., os_frac)
        pixel_samples = np.arange(-1/18,19.1/18,1/9)

        #pixel_samples = np.linspace(0,1,os)

        
        #Find four surrounding subpixel PRF models
        
        colbelow = np.max(np.where(pixel_samples < colfrac)[0])
        colabove = np.min(np.where(pixel_samples >= colfrac)[0])
        rowbelow = np.max(np.where(pixel_samples < rowfrac)[0])
        rowabove = np.min(np.where(pixel_samples >= rowfrac)[0])

        
        # Lower left PRF
        #prf_LL_weight = (row_dist + col_dist)
        #prf_LL = prf_model[colbelow, rowbelow,:,:] * prf_LL_weight
        # upper left PRF
        #prf_UL_weight =  * rowfrac - colfrac+os_frac
        #prf_UL = prf_model[colbelow, rowbelow,:,:] * prf_UL_weight
        #lower right PRF
        #prf_LR_weight = * os_frac-row_dist + col_dist
        #prf_LR = prf_model[colbelow, rowbelow,:,:] * prf_LR_weight
        # upper right PRF
        #prf_UR_weight = os_frac-row_frac - col_frac+os_frac
        #prf_UR = prf_model[colbelow, rowbelow,:,:] * prf_UR_weight

        points = []
        LL = prf_model[rowbelow,colbelow,:,:]
        points.append((pixel_samples[colbelow],pixel_samples[rowbelow],LL))
        UL = prf_model[rowabove,colbelow,:,:]
        points.append((pixel_samples[colbelow],pixel_samples[rowabove],UL))
        LR = prf_model[rowbelow,colabove,:,:]
        points.append((pixel_samples[colabove],pixel_samples[rowbelow],LR))
        UR = prf_model[rowabove,colabove,:,:]
        points.append((pixel_samples[colabove],pixel_samples[rowabove],UR))
            
        points = sorted(points)
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
        
        subsampled = (q11 * (x2 - colfrac) * (y2 - rowfrac) +
                      q21 * (colfrac - x1) * (y2 -  rowfrac) +
                      q12 * (x2 - colfrac) * ( rowfrac - y1) +
                      q22 * (colfrac - x1) * ( rowfrac - y1)
                      ) / ((x2 - x1) * (y2 - y1) + 0.0)
        #re-normalize to 1

        if renormalize:
            subsampled /= np.sum(subsampled)
        subsampled *= flux

        tpfmodel = np.zeros(tpf_size)
        
        midprf = int(prf_size[0]/2), int(prf_size[1]/2)

        #print(tpfmodel.shape, subsampled.shape)
        
        tpfmodel[int(np.max([0,rowint-midprf[0]])):int(np.min([tpf_size[0],rowint+midprf[0]+1])),
                 int(np.max([0,colint-midprf[1]])):int(np.min([tpf_size[1],colint+midprf[1]+1])),] = subsampled[
            int(np.max([0,midprf[0]-rowint])):int(np.min([2*midprf[0]+1,midprf[0]-rowint+tpf_size[0]])),
            int(np.max([0,midprf[1]-colint])):int(np.min([2*midprf[1]+1,midprf[1]-colint+tpf_size[1]])),
            ]

        ###############################################################
        ## ABOVE ~60 Lines of Code adapted from
        ## https://github.com/keatonb/TESS_PRF/blob/main/src/PRF/prf.py
        ###############################################################
        
        return tpfmodel





class TPFSceneModeler(object):

    def __init__(self, camera, ccd, sector, ref_col, ref_row, source_cols, source_rows, source_mags, 
                 tpfshape=(15,15), buffer_size=5):

        self.cam, self.ccd, self.sector, self.ref_col, self.ref_row = camera, ccd, sector, ref_col, ref_row        
        self.TESS_PRF = TESS_PRF(camera, ccd, sector, ref_col, ref_row)

        self.zeropoint_mag = TESS_ZEROPOINT_MAG
        self.buffer_size=buffer_size
        
        self.os_factor=self.TESS_PRF.oversamp
        self.tpfshape = tpfshape

        self.source_cols = np.array(source_cols)
        self.source_rows = np.array(source_rows)
        self.source_mags = np.array(source_mags)
        self.source_fluxes= 10.**(-0.4*(self.source_mags-self.zeropoint_mag))
        

        #print(self.model.shape)

        self.catalog = None
        self.center = tpfshape[0]//2+.5, tpfshape[1]//2+.5

        self.row_index, self.col_index = np.arange(0,tpfshape[0]), np.arange(0,tpfshape[1])
        
        self.oversampled_row_index = np.arange(-buffer_size, tpfshape[0]+buffer_size, 1./self.os_factor)
        self.oversampled_col_index = np.arange(-buffer_size, tpfshape[1]+buffer_size, 1./self.os_factor)

        self.model = self._make_scene_model()



    def _interp(row_offset, col_offset):

        row_coords = d_row+self.row_index
        col_coords = d_col+self.col_index

        return self.model((row_coords.ravel(), col_coords.ravel()))



    def _make_scene_model_data(self,  cols=None, rows=None, mags=None, tpfshape=None, buffer_size=None):

        return 1.

    def _recompute_scene_model(self, ):

        super_sampled_row_i = self.oversampled_row_index.copy()
        super_sampled_col_i = self.oversampled_col_index.copy()
        super_sampled_scene_model_data = self.model_data.copy()

        scene_model = RectBivariateSpline(super_sampled_row_i, super_sampled_col_i, super_sampled_scene_model_data, 
                                                  kx=1, ky=1, s=0, )

        self.model = scene_model
        

    def _make_scene_model(self, cols=None, rows=None, mags=None, tpfshape=None, buffer_size=None):

        source_cols=cols
        source_rows=rows
        source_mags=mags
        
        if source_rows is None:
            source_rows = self.source_rows
        
        if source_cols is None:
            source_cols = self.source_cols
            
        if source_mags is None:
            source_mags = self.source_mags
            source_fluxes = self.source_fluxes
        else:
            source_fluxes = 10.**(-0.4*(source_mags-self.zeropoint_mag))
            
        if tpfshape is None:
            tpfshape = self.tpfshape

            if buffer_size is None:
                buffer_size = self.buffer_size  
            
            super_sampled_row_i = self.oversampled_row_index
            super_sampled_col_i = self.oversampled_col_index

        else:
            super_sampled_row_i = np.arange(-buffer_size, tpfshape[0]+buffer_size, 1./self.os_factor)
            super_sampled_col_i = np.arange(-buffer_size, tpfshape[1]+buffer_size, 1./self.os_factor)
            


        super_sampled_scene_model_data = np.zeros((len(super_sampled_row_i), len(super_sampled_col_i)) )

        min_row, max_row = super_sampled_row_i[0], super_sampled_row_i[-1]
        min_col, max_col = super_sampled_col_i[0], super_sampled_col_i[-1]

        n_sources = len(source_rows)

        for i in range(n_sources):

            r_i=source_rows[i]
            c_i=source_cols[i]

            if r_i < min_row:
                continue 
            if r_i>max_row:
                continue
            if c_i<min_col:
                continue
            if c_i>max_col:
                continue

            prf_flux = self.TESS_PRF.interpolate(row=r_i, col=c_i, 
                                                 row_index=super_sampled_row_i, 
                                                 col_index=super_sampled_col_i) * source_fluxes[i]
            
            super_sampled_scene_model_data += prf_flux 


        scene_model = RectBivariateSpline(super_sampled_row_i, super_sampled_col_i, super_sampled_scene_model_data, 
                                                  kx=1, ky=1, s=0, )

        self.model = scene_model
        self.model_data =  super_sampled_scene_model_data
        
        return scene_model

        

    def interpolate_scene(self, dx=0., dy=0., flux_scale=1.):


        col_offset=dx
        row_offset=dy
        row_i = self.row_index
        col_i = self.col_index

        if np.abs(dx)>5. or np.abs(dy)>5.:
            print('**************************\n**\n**    WARNING: POINTING OFFSET LARGER THAN MODELED AREA \n**\n********************')
        
        scene = self.model(col_i+col_offset, row_i+row_offset)
        
        return scene*flux_scale
    


class TPFSceneModeler_OLD(TESS_PRF_Model):

    def __init__(self, camera, ccd, sector, ref_col, ref_row, source_cols, source_rows, source_mags, tpfshape=(15,15), os_factor=9, buffer_size=5):

        
        self.col_offset, self.row_offset = np.linspace(-0.5, 0.5, os_factor), np.linspace(-0.5, 0.5, os_factor)
        self.prf = self._get_prf_model(camera, ccd, sector, ref_col, ref_row)

        self.zeropoint_mag = TESS_ZEROPOINT_MAG
        self.buffer_size=buffer_size
        self.os_factor=os_factor
        self.shape = tpfshape

        self.source_cols = source_cols
        self.source_rows = source_rows
        self.source_mags = source_mags
        
        self.model = self._make_scene_model_convolve(source_rows, source_cols, source_mags, tpfshape)

        #print(self.model.shape)

        self.catalog = None
        self.center = tpfshape[0]//2+.5, tpfshape[1]//2+.5


    def _get_prf_model(self, camera, ccd, sector, ref_col, ref_row, prf_dir=PRF_FILE_DIRECTORY, trim_outside=False ):

        #if sector<4:
        #    prf_dir+='/start_s0001/'
        #else:
        #    prf_dir+='/start_s0004/'
        #return PRF.TESS_PRF(camera, ccd, sector, ref_col, ref_row, localdatadir=prf_dir)

        model = TESS_PRF_Model(camera, ccd, sector, ref_col, ref_row, localdatadir=prf_dir)
        
        return model
        
    def _make_scene_model(self,  cols=None, rows=None, mags=None, tpfshape=None):

        if cols is None:
            cols=self.source_cols
            rows=self.source_rows
            mags = self.source_mags
            tpfshape=self.shape

        
        size_x, size_y = tpfshape
        buffered_size = (size_y+2*self.buffer_size, size_x+2*self.buffer_size)

        scene_model = np.zeros((self.os_factor, self.os_factor, buffered_size[0], buffered_size[1]))

        for i,xi in enumerate(self.col_offset):
            for j,yi in enumerate(self.row_offset):
                for k in range(len(cols)):
                    star_row = rows[k]+self.buffer_size + 0.5 # 0.5 added to agree with indexing convention
                    star_col = cols[k]+self.buffer_size + 0.5 #

                    if star_row<0 or star_row>buffered_size[0] or star_col<0 or star_col>buffered_size[0]:
                        continue

                    

                    #if star_row+xi > buffered_size[1] or star_col+yi > buffered_size[0]:
                    #    continue
                    
                    star_mag = mags[k]
                    try:
                        #scene_model[i,j]+=self.prf.locate(star_row+yi, star_col+xi, buffered_size) * 10.**(-0.4*(star_mag-self.zeropoint_mag))
                        scene_model[i,j]+= self.prf._interp(star_col+xi, star_row+yi, 1., buffered_size) * 10.**(-0.4*(star_mag-self.zeropoint_mag))
                    except ValueError:
                        print(star_row, star_col, )
                        scene_model[i,j]+= self.prf._interp(star_col+xi, star_row+yi, flux=1., tpf_size=buffered_size) * 10.**(-0.4*(star_mag-self.zeropoint_mag))
                        pass

        return scene_model


    def _make_scene_model_convolve(self, cols=None, rows=None, mags=None, tpfshape=None):

        if cols is None:
            cols=self.source_cols
            rows=self.source_rows
            mags = self.source_mags
            tpfshape=self.shape

        star_fluxes =  10.**(-0.4*(np.array(mags)-self.zeropoint_mag))
        
        scene_model = calculate_scene_model_fftconvolve(self.prf.model, star_cols=cols, star_rows=rows,
                                          star_flux=star_fluxes, tpfsize=tpfshape, buffersize=self.buffer_size)

        return scene_model

    def interpolate_scene(self, dx=0, dy=0, flux_scale=1.):
        bs = self.buffer_size
        buffered_size = self.shape[0]+2*bs, self.shape[1]+2*bs
        #print(buffered_size, self.center)
        buffered_scene = self._interp(-dy+self.center[0]+bs, -dx+self.center[1]+bs, flux=1., tpf_size=buffered_size, renormalize=False)
        return buffered_scene[bs:-bs,bs:-bs]*flux_scale





def calculate_scene_model_fftconvolve(prf_model, star_cols, star_rows, star_flux, tpfsize, buffersize=5):

    prf_size = prf_model.shape[2:]
    os_factor = prf_model.shape[0]


    dx = np.linspace(-0.5, 0.5, os_factor)

    buffered_size = (tpfsize[0]+2*buffersize, tpfsize[1]+2*buffersize)

    new_cols = star_cols+buffersize+0.5
    new_rows = star_rows+buffersize+0.5


    full_scene_prf_model = np.zeros((os_factor, os_factor, tpfsize[0]+2*buffersize, tpfsize[1]+2*buffersize))

    for col_i in range(os_factor):
        for row_i in range(os_factor):

            full_scene_prf_model[row_i, col_i] = calculate_scene_convolve(prf_model, star_cols=new_cols+dx[col_i], 
                                                        star_rows=new_rows+dx[row_i], star_flux=star_flux,tpfsize=buffered_size)

    return full_scene_prf_model



def bilinear_interp_weights(xfrac, yfrac):

    w_ll = (1.-xfrac)*(1.-yfrac)
    w_lu = (1.-xfrac)*yfrac
    w_ul = xfrac*(1.-yfrac)
    w_uu = xfrac*yfrac
    
    return w_ll, w_lu, w_ul, w_uu


def calculate_scene_convolve(prf_model, star_cols, star_rows, star_flux, tpfsize):

    os_factor = prf_model.shape[0]
    os_frac = 1./os_factor
    
    prf_size = prf_model.shape[2:]
    
    scene_model_weights = np.zeros((os_factor, os_factor, tpfsize[0], tpfsize[1]))

    pixel_samples = np.linspace(0, 1, os_factor)
    d_samp=pixel_samples[1]-pixel_samples[0]

    for i in range(len(star_cols)):

        row=star_rows[i]
        col=star_cols[i]

        if any([row<0, col>=tpfsize[0], col<0, row>=tpfsize[1]]):
            continue

        tpfcolint = int(np.floor(col) )
        tpfrowint = int(np.floor(row) )
    
        # Fractional Pixels to select PRF subsampled model
        tpfcolfrac = col % 1.
        tpfrowfrac = row % 1.


        #if row<0 or col<0:
        #    print(row,col)
        
        #pixel_samples = np.arange(-1/18,19.1/18,1/9)
    
    
        #colbelow = np.max(np.where(pixel_samples < tpfcolfrac)[0])
        #colabove = np.min(np.where(pixel_samples >= tpfcolfrac)[0])
        #rowbelow = np.max(np.where(pixel_samples < tpfrowfrac)[0])
        #rowabove = np.min(np.where(pixel_samples >= tpfrowfrac)[0])        
        
        colbelow = int(tpfcolfrac//d_samp)
        colabove = colbelow+1
        rowbelow = int(tpfrowfrac//d_samp)
        rowabove = rowbelow+1
    
        prf_colfrac = (tpfcolfrac-pixel_samples[colbelow])/d_samp
        prf_rowfrac = (tpfrowfrac-pixel_samples[rowbelow])/d_samp

        c_up_r_up, c_up_r_lo, c_lo_r_up, c_lo_r_lo = bilinear_interp_weights(prf_colfrac, prf_rowfrac)
    
        #c_up_r_up = 0.25 * ( (1.-prf_colfrac) + (1.-prf_rowfrac) )
        #c_up_r_lo = 0.25 * ( (1.-prf_colfrac) + prf_rowfrac )
        #c_lo_r_up = 0.25 * ( prf_colfrac + (1.-prf_rowfrac) )
        #c_lo_r_lo = 0.25 * ( prf_colfrac + prf_rowfrac )

        #sum_weights = (c_up_r_up+c_up_r_lo+c_lo_r_up+c_lo_r_lo)
        
        #if np.sum([c_up_r_up, c_up_r_lo, c_lo_r_up, c_lo_r_lo])!=1:
        #    print([c_up_r_up, c_up_r_lo, c_lo_r_up, c_lo_r_lo], np.sum([c_up_r_up, c_up_r_lo, c_lo_r_up, c_lo_r_lo]) )
        
        #if any(np.isnan([c_up_r_up, c_up_r_lo, c_lo_r_up, c_lo_r_lo])):
        #    print(col, row, 'nans?', prf_colfrac, tpfcolfrac)

        #if any([tpfcolint<0, tpfcolint>=tpfsize[0], tpfrowint<0, tpfrowint>=tpfsize[1]]):
        #    continue

        
        scene_model_weights[rowbelow, colbelow, tpfrowint, tpfcolint] += c_lo_r_lo * star_flux[i] #/ sum_weights)
        scene_model_weights[rowabove, colbelow, tpfrowint, tpfcolint] += c_lo_r_up * star_flux[i] #/ sum_weights)
        scene_model_weights[rowbelow, colabove, tpfrowint, tpfcolint] += c_up_r_lo * star_flux[i]#/ sum_weights)
        scene_model_weights[rowabove, colabove, tpfrowint, tpfcolint] += c_up_r_up * star_flux[i]#/ sum_weights)


    scene_model = np.zeros(shape=tpfsize, dtype=np.float64)

    for k in range(os_factor):
        for j in range(os_factor):

            #print()
            #print(scene_model_weights[i,j,:,:].shape,  prf_model[i,j,:,:].shape, scene_model.shape)

            scene_model += convolve(scene_model_weights[k,j,:,:], prf_model[k,j,:,:], mode='same', )

    return scene_model




def trim_prf_model(prf_model, nsigma=3., extrapolate_wings=True, max_radius=6., oversamp=9):

    prf_model_trimmed = prf_model.copy()

    # Get everything within one pixel of the edge
    edges = np.concatenate([prf_model_trimmed[0:9,1:-1].ravel(), 
                            prf_model_trimmed[-9:,1:-1].ravel(), 
                            prf_model_trimmed[:,-9:].ravel(),
                            prf_model_trimmed[:,:9].ravel() ]).ravel()

    xi,yi = np.mgrid[:prf_model_trimmed.shape[0], :prf_model_trimmed.shape[1]]/oversamp
    ri = np.sqrt((xi-np.median(xi))**2. + (yi-np.median(yi))**2.)
    
    trim_cut = prf_model_trimmed < nsigma*np.std(edges) + np.mean(edges)
    trim_cut |= ri>max_radius
       

    if extrapolate_wings:

        xi,yi = np.mgrid[:prf_model_trimmed.shape[0], :prf_model_trimmed.shape[1]]

        ri = np.sqrt((xi-np.median(xi))**2. + (yi-np.median(yi))**2.).ravel()
        r_max = np.max(ri) + 5.

        x_edges = r_max * np.cos(np.arange(0, 2.*np.pi, np.pi/6.))+np.median(xi)
        y_edges = r_max * np.sin(np.arange(0, 2.*np.pi, np.pi/6.))+np.median(yi)


        zi = prf_model_trimmed
        zi[trim_cut] = np.nan
        #z[:,[0,-1]] = 1e-6
        #z[[0,-1],:] = 1e-6

        
        nan_cut = np.isnan(zi)

        x_input = np.append(xi[~nan_cut], x_edges)
        y_input = np.append(yi[~nan_cut], y_edges)

        z_edges = griddata((xi[~nan_cut], yi[~nan_cut]), zi[~nan_cut], (x_edges, y_edges), method='nearest' ) * 3e-3
        z_input = np.append(zi[~nan_cut], z_edges)

        zi = griddata((x_input, y_input), np.log10(z_input), (xi.ravel(), yi.ravel()), method='linear')

        
        return 10.**(zi.reshape(prf_model_trimmed.shape))

    prf_model_trimmed[trim_cut] = 0.

    return prf_model_trimmed

