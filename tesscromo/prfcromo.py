import PRF
import numpy as np

from scipy.signal import fftconvolve
import os 


PKG_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PRF_FILE_DIRECTORY = PKG_DIRECTORY + 'data/mastprf_fitsfiles/'


class TESS_PRF_Model(object):

    def __init__(self, camera, ccd, sector, column, row, localdatadir = PRF_FILE_DIRECTORY):

        #
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
        
        tpfmodel[int(np.max([0,rowint-midprf[0]])):int(np.min([prf_size[0],rowint+midprf[0]+1])),
                 int(np.max([0,colint-midprf[1]])):int(np.min([prf_size[1],colint+midprf[1]+1])),] = subsampled[
            int(np.max([0,midprf[0]-rowint])):int(np.min([2*midprf[0]+1,midprf[0]-rowint+prf_size[0]])),
            int(np.max([0,midprf[1]-colint])):int(np.min([2*midprf[1]+1,midprf[1]-colint+prf_size[1]])),
            ]

        ###############################################################
        ## ABOVE ~60 Lines of Code adapted from
        ## https://github.com/keatonb/TESS_PRF/blob/main/src/PRF/prf.py
        ###############################################################
        
        return tpfmodel



   
    


class TPFSceneModeler(TESS_PRF_Model):


    def __init__(self, camera, ccd, sector, ref_col, ref_row, source_cols, source_rows, source_mags, tpfshape=(15,15), os_factor=11, buffer_size=5):

        
        self.col_offset, self.row_offset = np.linspace(-0.5, 0.5, os_factor), np.linspace(-0.5, 0.5, os_factor)
        self.prf = self._get_prf_model(camera, ccd, sector, ref_col, ref_row)

        self.zeropoint_mag = 20.44
        self.buffer_size=buffer_size
        self.os_factor=os_factor
        self.shape = tpfshape
        
        self.model = self._make_scene_model(source_cols, source_rows, source_mags, tpfshape)

        #print(self.model.shape)

        self.catalog = None
        self.center = tpfshape[0]//2+.5, tpfshape[1]//2+.5


    def _get_prf_model(self, camera, ccd, sector, ref_col, ref_row, prf_dir=None ):

        return PRF.TESS_PRF(camera, ccd, sector,ref_col,ref_row)

        
    def _make_scene_model(self, cols, rows, mags, tpfshape):

        size_x, size_y = tpfshape
        buffered_size = (size_y+2*self.buffer_size, size_x+2*self.buffer_size)

        scene_model = np.zeros((self.os_factor, self.os_factor, buffered_size[0], buffered_size[1]))

        for i,xi in enumerate(self.col_offset):
            for j,yi in enumerate(self.row_offset):
                for k in range(len(cols)):
                    star_row = rows[k]+self.buffer_size
                    star_col = cols[k]+self.buffer_size

                    if star_row<0 or star_row>buffered_size[0] or star_col<0 or star_col>buffered_size[0]:
                        continue

                    

                    #if star_row+xi > buffered_size[1] or star_col+yi > buffered_size[0]:
                    #    continue
                    
                    star_mag = mags[k]
                    try:
                        scene_model[i,j]+=self.prf.locate(star_row+yi, star_col+xi, buffered_size) * 10.**(-0.4*(star_mag-self.zeropoint_mag))
                    except ValueError:
                        pass

        return scene_model


    def interpolate_scene(self, dx=0, dy=0, flux_scale=1.):
        bs = self.buffer_size
        buffered_size = self.shape[0]+2*bs, self.shape[1]+2*bs
        #print(buffered_size, self.center)
        buffered_scene = self._interp(dx+self.center[0]+bs, dy+self.center[1]+bs, flux=1., tpf_size=buffered_size, renormalize=False)
        return buffered_scene[bs:-bs,bs:-bs]*flux_scale
