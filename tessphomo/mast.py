import numpy as np
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import lightkurve as lk
import pandas as pd


def get_tic_sources(ticid, tpf_shape=[15,15], mag_lim=20.):

    pix_scale = 21.0

    try:
        catalogTIC = Catalogs.query_object('TIC '+str(ticid), radius=Angle((np.max(tpf_shape)/1.5+5) * pix_scale, "arcsec"), catalog="TIC")
    except:
        catalogTIC = Catalogs.query_object(str(ticid), radius=Angle((np.max(tpf_shape)/1.5+5.) * pix_scale, "arcsec"), catalog="TIC")

    mag_cut = catalogTIC['Tmag'] < mag_lim
    source_catalog = catalogTIC.to_pandas().loc[mag_cut]


    return source_catalog




def get_source_catalog_coords(target_crd: object,
                       mag_lim: float = 20,
                       tpf_shape: int = 25,
                       pix_scale: float = 21) -> pd.DataFrame:
    """
    Queries the Vizier catalog for sources around a given target coordinate.

    Parameters:
    -----------
    target_crd : object
        The SkyCoord object around which to search for sources.
    mag_lim : float, optional
        The magnitude limit for filtering sources. Default is 20.
    tpf_shape : int, optional
        The shape of the target pixel file being used. Default is 15.
    pix_scale : float, optional
        The pixel scale used in the radius calculation. Default is 21.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the catalog of sources that meet the criteria.
    """
    # Define the Vizier catalog with the desired columns and filters
    Catalog = Vizier(columns=["ID", "Tmag", "pmRA", "pmDEC", "ra", "dec", "RA_orig", "Dec_orig"],
                     column_filters={"Tmag": f"<{mag_lim}"}, row_limit=-1)

    # Query the region around the target coordinate
    source_catalog = Catalog.query_region(
        target_crd,
        radius=Angle((tpf_shape / 1.5 + 5) * pix_scale, "arcsec"),
        catalog='IV/39/tic82'
    )[0].to_pandas()

    source_catalog.columns = ["ID", "Tmag", "pmRA", "pmDEC", "ra", "dec", "RA_orig", "Dec_orig"]
    return source_catalog





def get_source_catalog_ticid(tic_id: int,
                       mag_lim: float = 20,
                       tpf_shape: int = 25,
                       pix_scale: float = 21, 
                        row_limit: int=int(1e6)) -> pd.DataFrame:
    """
    Queries the Vizier catalog for sources around a given target coordinate.

    Parameters:
    -----------
    target_crd : object
        The SkyCoord object around which to search for sources.
    mag_lim : float, optional
        The magnitude limit for filtering sources. Default is 20.
    tpf_shape : int, optional
        The shape of the target pixel file being used. Default is 15.
    pix_scale : float, optional
        The pixel scale used in the radius calculation. Default is 21.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the catalog of sources that meet the criteria.
    """
    # Define the Vizier catalog with the desired columns and filters
    Catalog = Vizier(columns=["TIC", "Tmag", "pmRA", "pmDEC", "ra", "dec", "RA_orig", "Dec_orig", "BPmag", "RPmag", "Gmag"],
                     column_filters={"Tmag": f"<{mag_lim}"}, row_limit=-1)


    # Query the region around the target coordinate
    source_catalog = Catalog.query_region(
        "TIC {}".format(tic_id),
        radius=Angle((tpf_shape / 1.5 + 5) * pix_scale, "arcsec"),
        catalog='IV/39/tic82'
    )[0].to_pandas()

    source_catalog.columns = ["ID", "Tmag", "pmRA", "pmDEC", "ra", "dec", "RA_orig", "Dec_orig","gaiabp", "gaiarp", "GAIAmag"]
    return source_catalog






def get_source_catalog( target_crd=None, 
                        tic_id=None,
                       sort_dist=True,
                       **kwargs) -> pd.DataFrame:
    """
    Queries the Vizier catalog for sources around a given target coordinate.

    Parameters:
    -----------
    target_crd : object
        The SkyCoord object around which to search for sources.
    mag_lim : float, optional
        The magnitude limit for filtering sources. Default is 20.
    tpf_shape : int, optional
        The shape of the target pixel file being used. Default is 15.
    pix_scale : float, optional
        The pixel scale used in the radius calculation. Default is 21.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the catalog of sources that meet the criteria.
    """

    if tic_id is None and not(target_crd is None):
        source_catalog= get_source_catalog_coords(target_crd, **kwargs)
    else:
        source_catalog= get_source_catalog_ticid(tic_id, **kwargs)

    
    if sort_dist:

        if target_crd is None:
            ra_0, de_0 = source_catalog.loc[source_catalog['ID']==tic_id][['ra','dec']].to_numpy()[0]
            target_crd = SkyCoord(ra=ra_0, dec=de_0, unit='degree')

        all_coords = SkyCoord(source_catalog[['ra','dec']].to_numpy(), unit='deg')
        seps = all_coords.separation(target_crd)
        #print(seps.to(u.arcsecond))

        source_catalog['dstArcSec'] = seps.to(u.arcsecond).value

        return source_catalog.sort_values('dstArcSec')
   
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




