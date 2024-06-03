import numpy as np






def matrix_solve(model, data, data_err=None, power=2.):

    A = np.vstack(model).T
    x = data.ravel()[:,None]

    if not(data_err is None):
        x_err = data_err.ravel()[:,None]
        A =1./x_err**power * A
        x = 1./x_err**power * x

    w = np.linalg.solve( A.T.dot(A) , A.T.dot(x) )

    return w






def make_quality_mask(quality_bitmask, qflags, ):

    '''

    0: AttitudeTweak = 1 
    1: SafeMode = 2 
    2: CoarsePoint = 4 
    3: EarthPoint = 8 
    4: Argabrightening = 16 
    5: Desat = 32
    6: ApertureCosmic = 64 
    7: ManualExclude = 128 
    8: Discontinuity = 256
    9: ImpulsiveOutlier = 512 
    10: CollateralCosmic = 1024 
    11: Straylight = 2048

    #: The second stray light flag is set automatically by Ames/SPOC based on background level thresholds.
    12: Straylight2 = 4096
    
    # See TESS Science Data Products Description Document
    13: PlanetSearchExclude = 8192
    14: BadCalibrationExclude = 16384

    # Set in the sector 20 data release notes
    15: InsufficientTargets = 32768
    
    '''
    
    quality_mask = (quality_bitmask & int(sum([2**q for q in qflags]))) == 0

    return quality_mask



def mag_to_flux(mag, zp=20.44):
    return 10.**(-0.4*(mag-zp))

