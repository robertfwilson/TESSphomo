import numpy as np
from .meta import TESS_ZEROPOINT_MAG



def matrix_solve(model, data, data_err=None, power=2.):

    A = np.vstack(model).T
    #A = np.array(model)[:,None].T
    x = data.ravel()[:,None]

    if not(data_err is None):
        x_err = data_err.ravel()[:,None]
        A =1./x_err**power * A
        x = 1./x_err**power * x

    w = np.linalg.solve( A.T.dot(A) , A.T.dot(x) )

    return w



def solve_linear_model(
    A, y, y_err=None, prior_mu=None, prior_sigma=None, k=None, return_errors=False
):
    """
    Solves a linear model with design matrix A and observations y:
        Aw = y
    return the solutions w for the system assuming Gaussian priors.
    Alternatively the observation errors, priors, and a boolean mask for the
    observations (row axis) can be provided.

    Adapted from Luger, Foreman-Mackey & Hogg, 2017
    (https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)

    Parameters
    ----------
    A: numpy ndarray or scipy sparce csr matrix
        Desging matrix with solution basis
        shape n_observations x n_basis
    y: numpy ndarray
        Observations
        shape n_observations
    y_err: numpy ndarray, optional
        Observation errors
        shape n_observations
    prior_mu: float, optional
        Mean of Gaussian prior values for the weights (w)
    prior_sigma: float, optional
        Standard deviation of Gaussian prior values for the weights (w)
    k: boolean, numpy ndarray, optional
        Mask that sets the observations to be used to solve the system
        shape n_observations
    return_errors: boolean
        Whether to return error estimates of the best fitting weights

    Returns
    -------
    w: numpy ndarray
        Array with the estimations for the weights
        shape n_basis
    werrs: numpy ndarray
        Array with the error estimations for the weights, returned if `error` is True
        shape n_basis
    """
    if k is None:
        k = np.ones(len(y), dtype=bool)

    if y_err is not None:
        sigma_w_inv = A[k].T.dot(A[k].multiply(1 / y_err[k, None] ** 2))
        B = A[k].T.dot((y[k] / y_err[k] ** 2))
    else:
        sigma_w_inv = A[k].T.dot(A[k])
        B = A[k].T.dot(y[k])

    if prior_mu is not None and prior_sigma is not None:
        sigma_w_inv += np.diag(1 / prior_sigma ** 2)
        B += prior_mu / prior_sigma ** 2

    if isinstance(sigma_w_inv, (sparse.csr_matrix, sparse.csc_matrix)):
        sigma_w_inv = sigma_w_inv.toarray()
    if isinstance(sigma_w_inv, np.matrix):
        sigma_w_inv = np.asarray(sigma_w_inv)

    w = np.linalg.solve(sigma_w_inv, B)
    if return_errors:
        w_err = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
        return w, w_err
    return w



def mag_to_flux(mag, zp=TESS_ZEROPOINT_MAG):
    return 10.**(-0.4*(mag-zp))



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


