import numpy as np
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord, Angle
import lightkurve as lk



def get_tic_sources(ticid, tpf_shape=[15,15], mag_lim=20.):

    pix_scale = 21.0

    try:
        catalogTIC = Catalogs.query_object('TIC '+str(ticid), radius=Angle((np.max(tpf_shape)/1.5+5) * pix_scale, "arcsec"), catalog="TIC")
    except:
        catalogTIC = Catalogs.query_object(str(ticid), radius=Angle((np.max(tpf_shape)/1.5+5.) * pix_scale, "arcsec"), catalog="TIC")

    mag_cut = catalogTIC['Tmag'] < mag_lim
    source_catalog = catalogTIC.to_pandas().loc[mag_cut]


    return source_catalog





def retrieve_tess_ffi_cutout_from_mast(cutout_size, sector, ticid=None, coords=None):

    if ticid is None:
        if coords is None:
            print('.\n.\n.\n.\nMUST SPECIFY TICID OR ASTROPY COORDINATES.\n.\n.\n.\n')
        else:
            search = lk.search_tesscut(coords, sector=sector)
                        
    elif isinstance(ticid , int):
        search = lk.search_tesscut('TIC '+str(ticid),sector=sector)
    else:
        search = lk.search_tesscut(ticid, sector=sector)
    
    tpf = search.download_all(cutout_size=cutout_size,  quality_bitmask=None)

    return tpf



def retrieve_tess_ffi_cutout_from_aws(cutout_size, sector, ticid=None, coords=None):

    return 1. 




