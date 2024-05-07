import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches

import astropy.units as u
import numpy as np


def plot_aperture(ax, aperture_mask, mask_color='C3'):
    
    for i in range(aperture_mask.shape[0]):
        for j in range(aperture_mask.shape[1]):
            if aperture_mask[i, j]>0:
                xy = (j - 0.5, i - 0.5)
                rect = patches.Rectangle(
                            xy=xy,
                            width=1,
                            height=1,
                            color=mask_color,
                            fill=False,
                            hatch="",
                        )
                ax.add_patch(rect)




def plot_ne_arrow(ax, wcs, x_0=2.,y_0=12, len_pix=1.5, pix_scale=21.*u.arcsec, color='C1', ):

    pix_scale_deg = pix_scale.to(u.degree).value
    
    ra_0, dec_0 =wcs.all_pix2world(x_0,y_0,0)

    ra_1 = ra_0 + len_pix * pix_scale_deg/np.cos(np.deg2rad(dec_0) )
    dec_1 =dec_0+ len_pix * pix_scale_deg

    xpix_n, ypix_n = wcs.all_world2pix([ra_0, ra_0], [dec_0,dec_1], 0)
    xpix_e, ypix_e = wcs.all_world2pix([ra_0, ra_1], [dec_0,dec_0], 0)

    dx_n = xpix_n[1]-xpix_n[0]
    dy_n = ypix_n[1]-ypix_n[0]

    dx_e = xpix_e[1]-xpix_e[0]
    dy_e = ypix_e[1]-ypix_e[0]


    x_min, x_max = min([x_0, x_0+dx_n*(len_pix+1)/len_pix, x_0+dx_e*(len_pix+1)/len_pix]), max([x_0, x_0+dx_n*(len_pix+1)/len_pix, x_0+dx_e*(len_pix+1)/len_pix])
    y_min, y_max = min([y_0, y_0+dy_n*(len_pix+1)/len_pix, y_0+dy_e*(len_pix+1)/len_pix]), max([y_0, y_0+dy_n*(len_pix+1)/len_pix, y_0+dy_e*(len_pix+1)/len_pix])


    # Shifts arrow so that minimum always occurs at (1,1)
    x = x_0 + x_0 - (x_max+x_min)/2. 
    y = 2*y_0 - (y_max+y_min)/2.
    
    ax.arrow( x, y, dx_n,dy_n , color=color, width=0.1, head_width=0.5, length_includes_head=True)
    ax.arrow( x, y, dx_e,dy_e , color=color, width=0.1, head_width=0.5, length_includes_head=True)

    # Draw the N and E at the end of arrows
    ax.text(x+dx_n+(0.7*dx_n/len_pix), y+dy_n+(0.5*dy_n/len_pix),'N', color=color, va='center', ha='center' ,
           rotation=np.rad2deg(np.arctan(dy_n/dx_n))-np.sign(dx_n)*90   )
    ax.text(x+dx_e+(0.7*dx_e/len_pix), y+dy_e+(0.5*dy_e/len_pix),'E', color=color, va='center', ha='center' ,
           rotation=np.rad2deg(np.arctan(dy_e/dx_e))-np.sign(dx_e)*90  )
    
    return ra_0,dec_0
