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




def make_corr_lc(raw_fluxes, systematics, raw_flux_errs=None, nterms=4, err_exponent=2.):

    #prf_flux = lc['raw_prf_flux']
    #cap_flux = lc['raw_capflux']
    #systematics = np.array([lc['zp_flux_scale'], 
    #                        lc['bkg_sapflux'],
    #                       lc['col_offset'], 
    #lc['row_offset']])

    corr_fluxes = []

    dm = DesignMatrix(np.vstack(systematics).T)
    dm = dm.pca(nterms).append_constant()

    X = dm.X
    
    for i,f in enumerate(raw_fluxes):

        if not(flux_err is None):

            flux_err = np.ones_like(lc['prf_flux'])
            Xw = X.T.dot(X / flux_err[:, None] ** err_exponent)    
            B_prf = np.dot(X.T, lc['prf_flux']/flux_err )

    w_prf = np.linalg.solve(Xw, B_prf).T
    
    prf_flux_corr = prf_flux/X.dot(w_prf)
    cap_flux_corr = cap_flux/X.dot(w_cap)

    return prf_flux_corr, cap_flux_corr


