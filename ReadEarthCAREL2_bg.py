import sys, os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
from netCDF4 import Dataset
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

FONTSIZE=15
INFOSIZE=13

WP_min, WP_max = 1e-3, 1e5 
WC_min, WC_max = 1e-6, 10**4.5# 1e-5, 10**4.5
cmap = plt.get_cmap('viridis')  # 'viridis' 'inferno'

def find_nearest_id(array,value):
    idx=(np.abs(array-value)).argmin()
    return idx

def moving_average(x, w=3):
    # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean?noredirect=1&lq=1

    return np.convolve(x, np.ones(w), 'same') / w #'full') / w #'valid') / w



# BG: My interpolation function 
def interpolate(x1, data1_raw, fill1, x2, data2_raw, fill2):
    """
    1) Masks fill values in data1 and data2  
    2) Extracts valid points from data1 and sorts them  
    3) Interpolates data1 onto x2  
    4) Sets out‑of‑bounds interpolation to NaN  
    5) Masks data2 zeros/nans  
    6) Returns only the indices where both interpolated data1 and data2 are valid

    Parameters
    - 1D arrays
    - x1, x2 (target)   : latitudes
    - d1, d2 (target)   : data
    - fill1, fill2      : fill values 


    Returns
    - x_valid               : subset of x2, valid for both data
    - data1_interp_valid    : data1 interpolated onto x2 + masked to x_valid.
    - data2_valid           : data2_raw (with fill2→NaN) + masked to ly x_valid.
    """

    # mask fill values → NaN
    d1 = np.where(data1_raw == fill1, np.nan, data1_raw)
    d2 = np.where(data2_raw == fill2, np.nan, data2_raw)

    # select non‑NaN points in data1
    valid1 = ~np.isnan(d1)
    x1v, d1v = x1[valid1], d1[valid1]

    # sort for np.interp requirements
    order = np.argsort(x1v)
    x1v, d1v = x1v[order], d1v[order]

    # interpolate onto x2
    interp_raw = np.interp(x2, x1v, d1v)

    # mask out‑of‑bounds
    oob = (x2 < x1v[0]) | (x2 > x1v[-1])
    interp = np.where(oob, np.nan, interp_raw)

    # mask invalid or zero in data2
    valid2 = ~np.isnan(d2) & (d2 != 0)

    # final valid mask
    valid = (~np.isnan(interp)) & valid2

    # return only the good points
    x_valid            = x2[valid]
    data1_interp_valid = interp[valid]
    data2_valid        = d2[valid]

    return x_valid, data1_interp_valid, data2_valid

# BG: My add-profile-to-plot function
# BG: Used in solar_both and thermal_both
def add_profile_to_plot(fig, ax, ACMCOM, fsize, legend_list, quantity=None, iacr=151, stacked=True, normalize=False):
    """
    Add a single profile (ex. elevation, lwc, iwc, ...) to the given axis.
    
    Parameters:
        ax       : main matplotlib axis
        pl_list  : list to append plotted Line2D objects
        quantity : one of ['elevation', 'lwc', ....]
        iacr     : nadir column index
    
    Returns:
        ax2      : twin axis used for plotting
    """
    
    # Define thresholds (might need to be adjusted)
    THRESHOLD_lwc = 1e30 
    THRESHOLD_iwc = 1e30
    THRESHOLD_lwp = 400  + 1e30
    THRESHOLD_iwp = 1000 + 1e30

    if quantity is None:
        # No profile to plot
        return None

    x = ACMCOM.latitude_active
    n_lat = x.shape[0]
    profile = []

    if quantity == 'elevation':
        for ia in range(n_lat):
            heights = ACMCOM.height_level[1:, ia]
            pressures = ACMCOM.pressure_level[1:, ia]
            valid_heights = [h for h, p in zip(heights, pressures) if h > 0 and p < 1e10]
            profile.append(min(valid_heights) if valid_heights else np.nan)
        ylabel = 'Surface Elevation [m]'
        color = 'black'
        label = 'Surface Elevation'

    elif quantity == 'lwp':
        for ia in range(n_lat):
            heights = ACMCOM.height_layer[:, ia]
            lwc = ACMCOM.liquid_water_content[:, ia]
            valid_heights = []
            valid_lwc = []
            for h, l in zip(heights, lwc):
                if h >= 0 and h < 1e35 and l < THRESHOLD_lwc: 
                    valid_heights.append(h)
                    valid_lwc.append(l)
            if len(valid_heights) > 1:
                totcol = -np.trapezoid(valid_lwc, x=valid_heights)
                profile.append(totcol if (totcol < THRESHOLD_lwp) else 0.0)
            else:
                profile.append(0.0)
        ylabel = 'LWP [kg/m$^2$]'
        color = (0.2, 0.4, 0.8)   # stronger blue
        label = 'LWP'

    elif quantity == 'iwp':
        for ia in range(n_lat):
            heights = ACMCOM.height_layer[:, ia]
            iwc = ACMCOM.ice_water_content[:, ia]
            valid_heights = []
            valid_iwc = []
            for h, l in zip(heights, iwc):
                if h >= 0 and h < 1e35 and l < THRESHOLD_iwc:
                    valid_heights.append(h)
                    valid_iwc.append(l)
            if len(valid_heights) > 1:
                totcol = -np.trapezoid(valid_iwc, x=valid_heights)
                profile.append(totcol if (totcol < THRESHOLD_iwp) else 0.0)
            else:
                profile.append(0.0)
        ylabel = 'IWP [kg/m$^2$]'
        color = (0.0, 0.6, 0.5)   # greenish teal
        label = 'IWP'

    elif quantity == 'tot_wp':
        profile = [[],[]]
        for ia in range(n_lat):
            heights = ACMCOM.height_layer[:, ia]
            lwc     = ACMCOM.liquid_water_content[:, ia]
            iwc     = ACMCOM.ice_water_content[:, ia]

            
            valid_h = []
            valid_iwc = []
            valid_lwc = []
            for h, l, i in zip(heights, lwc, iwc):
                if 0 <= h < 1e35 and l < THRESHOLD_lwc and i < THRESHOLD_iwc:
                    # replace any non‑finite with zero
                    l_val = l if np.isfinite(l) else 0.0
                    i_val = i if np.isfinite(i) else 0.0

                    valid_h.append(h)
                    valid_iwc.append(i_val)
                    valid_lwc.append(l_val)

            if len(valid_h) > 1:
                totcol_i = -np.trapezoid(valid_iwc, x=valid_h)
                totcol_l = -np.trapezoid(valid_lwc, x=valid_h)
                profile[0].append(totcol_i if (totcol_i < THRESHOLD_iwp) else 0.0)
                profile[1].append(totcol_l if (totcol_l < THRESHOLD_lwp) else 0.0)
            else:
                profile[0].append(0.0)
                profile[1].append(0.0)
        ylabel = 'IWP & LWP \n [kg/m$^2$]' #'Water Path [kg/m$^2$]'
        color  =  'black' # (0.1, 0.5, 0.65) # combo LWP and IWP
        label  = 'LWP & IWP'

    elif quantity == 'tot_wc': # lat vs altitude
        profile = [[],[]]
        ylabel = ['IWC [kg/m$^3$]','LWC [kg/m$^3$]'] #'Water content 
        color  =  'black' 
        label  = 'IWC & LWC'
        iwc = ACMCOM.ice_water_content; lwc = ACMCOM.liquid_water_content
        iwc = np.where((iwc >= 1e30) | (iwc == 0), np.nan, iwc)
        lwc = np.where((lwc >= 1e30) | (lwc == 0), np.nan, lwc)
        vmax = WC_max
        vmin = 0

        profile[0].append(iwc)
        profile[1].append(lwc)



    elif quantity == 'albedo' or quantity == 'SWalbedo' or quantity == 'LWalbedo':
        if 'thermal' in plot_type or quantity == 'LWalbedo':
            # long‑wave albedo averaged over all thermal wavelengths
            nwvl = ACMCOM.wavelengths_thermal_surface_emissivity.shape[0]
            for ia in range(n_lat):
                type_idx = ACMCOM.surface_emissivity_type_index[iacr, ia]
                if type_idx < 0 or type_idx > ACMCOM.surface_emissivity_table.shape[0]:
                    # missing or invalid type
                    profile.append(0.0)
                else:
                    # compute albedo = 1 - emissivity for each wavelength
                    albs = []
                    for iwvl in np.arange(nwvl-1,-1,-1):
                        eps = ACMCOM.surface_emissivity_table[type_idx-1, iwvl]
                        if np.isfinite(eps):
                            albs.append(1.0 - eps)
                    profile.append(np.nanmean(albs) if albs else 0.0)
            ylabel = 'Albedo (LW) []'
            label  = 'LW Albedo'
            color  = (1.00, 0.40, 0.00) # orange
        elif 'solar' in plot_type or quantity == 'SWalbedo':
            # short‑wave albedo = mean(vis, NIR)
            for ia in range(n_lat):
                a1 = ACMCOM.albedo_diffuse_radiation_surface_visible[iacr, ia]
                a2 = ACMCOM.albedo_diffuse_radiation_surface_near_infrared[iacr, ia]
                # filter out sentinel large values and nonfinite
                if a1 >= 1e35 or a2 >= 1e35 or not np.isfinite(a1) or not np.isfinite(a2):
                    profile.append(0.0)
                else:
                    profile.append(np.nanmean([a1, a2]))
            ylabel = 'Albedo (SW) []'
            label  = 'SW Albedo'
            color  = (1.00, 0.84, 0.00) # gold


    elif quantity == 'aerosols':
        for ia in range(n_lat):
            nheights = ACMCOM.aerosol_extinction.shape[0] - 1
            aero_tau_tot = 0.0
            for ih in range(nheights):
                h = ACMCOM.height_level[ih+1, ia]
                dz = (ACMCOM.height_level[ih, ia] - h) / 1000.0
                ext = ACMCOM.aerosol_extinction[ih, ia]
                if ACMCOM.aerosol_classification[ih, ia] >= 0 and ext < 1e30: #np.isfinite(ext) -> do not work
                    tau = ext * dz
                    aero_tau_tot += tau
            if not np.isfinite(aero_tau_tot):
                aero_tau_tot = 0.0
            if 'thermal' in plot_type:
                # scale for thermal: 
                import MakeRTMInputFile_bg as MakeRTM
                aero_tau_tot = aero_tau_tot * 0.6 if MakeRTM.aerosol_thermal_impact_bool(ia, ACMCOM) else 0.0 # Conversion Factor
            profile.append(aero_tau_tot)
        ylabel = 'AOD []' #'Aerosol optical depth []'
        color  = 'gray'
        label  = 'AOD'

    elif quantity == 'surface_temperature':
        # surface_T = min(T for T,v,p in zip(ACMCOM.temperature_level[1:,ia], ACMCOM.height_level[1:,ia], ACMCOM.pressure_level[1:,ia]) if v > 0 and p < 1e10)
        # surface_T: ", ACMCOM.surface_temperature[iacr,ia]) 
        for ia in range(n_lat):
            # T = ACMCOM.surface_temperature[iacr,ia]

            T = min((T for T, v, p in zip(ACMCOM.temperature_level[1:, ia],
                        ACMCOM.height_level[1:, ia],
                        ACMCOM.pressure_level[1:, ia])
                        if v > 0 and p < 1e10 and np.isfinite(T)), default=np.nan)

            if np.isfinite(T): profile.append(T)
            else: profile.append(np.nan)
        ylabel = 'Surf.Temp. [K]' #'Surface Temperature [K]'
        color  = 'red'
        label  = 'Surface Temperature'

    elif quantity == 'DEM_elevation':
        # Import GetElevation() from MakeRTMInpitFile.py to compare DEM-elevation to EarthCARE-elevation
        from MakeRTMInputFile_bg import GetElevation

        for ia in range(n_lat):
            h = GetElevation(ACMCOM.latitude_active[ia], ACMCOM.longitude_active[ia], ia)
            if np.isfinite(h): profile.append(h)
            else: profile.append(np.nan)
        ylabel = 'Surface Elevation (DEM) [m]'
        color  = 'pink'
        label  = 'Surface Elevation (DEM)'

    elif quantity == 'CF':
        THRESHOLD = 0 # kg/m3
        MAX = 1e30    # kg/m3
    
        want_large_buffer =  True     # True False
        Buffer_Along  = 12 if want_large_buffer else 6
        Buffer_Across = 10 if want_large_buffer else 6
        # handy refs
        LWC = ACMCOM.liquid_water_content    # (levels, columns)
        IWC = ACMCOM.ice_water_content       # (levels, columns)
        n_levels, n_cols = LWC.shape
        # output
        profile = np.full(n_cols, np.nan, dtype=float)
        # 1) per-column flags (compute once)
        valid_col = np.any(np.isfinite(LWC) & (LWC < MAX)    &  np.isfinite(IWC) & (IWC < MAX), axis=0)
        cloud_col = np.any(((LWC > THRESHOLD) & (LWC < MAX)) | ((IWC > THRESHOLD) & (IWC < MAX)), axis=0)
        # 2) adjusted index map (once)
        offset = 0
        if   "Orbit_05926C" in SceneName: offset = 2700
        elif "Orbit_06888C" in SceneName: offset = 2527
        elif "Orbit_07277C" in SceneName: offset = 2527
        elif "Orbit_06331C" in SceneName: offset = 2636
        idx = ACM3D.index_construction - offset          # (cross, along)
        acr_lo = ACM3D.nadir_pixel_index - Buffer_Across
        acr_hi = ACM3D.nadir_pixel_index + Buffer_Across + 1
        # 3) slide along
        for ia in range(Buffer_Along, n_cols - Buffer_Along):
            al_lo = ia - Buffer_Along
            al_hi = ia + Buffer_Along + 1
            # Column indices in Buffer
            # Rectangle of column IDs -> 1D 
            cols = idx[acr_lo:acr_hi, al_lo:al_hi].ravel()
            v = valid_col[cols]
            c = cloud_col[cols]
            v_sum = int(v.sum())
            CF = (int(c.sum()) / v_sum) if v_sum > 0 else np.nan
            profile[ia] = CF

        want_compare_to_small_buffer = True
        if want_compare_to_small_buffer: # Same procedure as above
            want_large_buffer =  False   
            Buffer_Along  = 12 if want_large_buffer else 6
            Buffer_Across = 10 if want_large_buffer else 6
            acr_lo = ACM3D.nadir_pixel_index - Buffer_Across
            acr_hi = ACM3D.nadir_pixel_index + Buffer_Across + 1
            for ia in range(Buffer_Along, n_cols - Buffer_Along):
                al_lo = ia - Buffer_Along
                al_hi = ia + Buffer_Along + 1
                cols = idx[acr_lo:acr_hi, al_lo:al_hi].ravel()
                v = valid_col[cols]
                c = cloud_col[cols]
                v_sum = int(v.sum())
                CF = (int(c.sum()) / v_sum) if v_sum > 0 else np.nan
                profile[ia] -= CF
            profile[:int(np.flatnonzero(valid_col)[0])], profile[int(np.flatnonzero(valid_col)[-1]):] = np.nan, np.nan # Eliminate boundary effects

        ylabel = 'Cloud Fraction []' if not want_compare_to_small_buffer else 'CF Diff.\n(Large - Small Buffer) []'
        color  = 'black'
        label  = 'CF' if not want_compare_to_small_buffer else 'CF Diff.'


    else:
        raise ValueError("Invalid quantity. Choose from: 'elevation', 'lwp', 'iwp', 'tot_wp', 'albedo', 'aerosols")

    alpha = 0.25
    # -------------------------------------- Handling Stacked Figures --------------------------------------------------------
    if stacked: # Stack the Add_Profile to a subplot under OG-figure (stacked on top)
        if not hasattr(fig, "_stacked_inited"): # check if first time call this function
            fig.clf()  # clear figure to rebuild layout
            if 'CF' in quantity_list and len(quantity_list) != 1: #-> if only CF -> only 2 stacked figures
                fig.set_figheight( fig.get_figheight() * 2) #increase fig-height for stacked figures
                gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1.5], hspace=0.0)
                ax = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
                ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
               
                fig._ax_top = ax
                fig._ax_mid = ax2
                fig._ax_bot = ax3
            else: 
                fig.set_figheight( fig.get_figheight() * 1.5) #increase fig-height for stacked figures
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.0)
                ax = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        
                fig._ax_top = ax
                fig._ax_bot = ax2

        if 'CF' in quantity_list: # 3-Figures Stacked
            if 'CF' in quantity: # Plotting CF into _ax_bot
                ax  = fig._ax_top
                ax2 = fig._ax_bot
            else: # Plot other quantities into _ax_mid
                ax  = fig._ax_top
                ax2 = fig._ax_mid
        else:
            ax  = fig._ax_top
            ax2 = fig._ax_bot
        alpha = 0.6

        # Axes background (warm light grey)
        ax2.set_facecolor('#f0f0f0')
        # Grid: major dashed, minor dotted
        ax2.grid(which='major', linestyle='--', alpha=0.4)
        ax2.grid(which='minor', linestyle=':',  alpha=0.2)
        ax2.spines['right'].set_visible(False)
        
        plt.setp(ax.get_xticklabels(), visible=False) # same x-ticks
        # -------------------------------------------

        # Make twin axis
        if ('wc' not in quantity) and ('CF' not in quantity or len(quantity_list) == 1): 
            ax2.tick_params(left=False, labelleft=False) # remove left y-ticks
            ax2.spines['left'].set_visible(False)
            ax2.tick_params(axis='y', which='both', left=True, labelleft=False, length=0)
            ax2 = ax2.twinx()
            ax2.spines['left'].set_visible(False)
           
        fig._stacked_inited = True # Set True -> function run at least once
        # -----------------------------------------------------------------------------------------------------------------------
    else:
        # Make twin axis
        ax2 = ax.twinx()




    #------------ Modify for multiple add-profiles-verticle spine position ---------
    # per-parent-axis counter
    n_right = getattr(ax, "right_twin_count", 0)
    # only push outward for the 2nd, 3rd, ... twins
    if n_right > 0 and 'CF' not in quantity:
        offset_step = 80 #60  # px; adjust to taste
        ax2.spines["right"].set_position(("outward", offset_step * n_right))

    # update counter for next call
    if 'CF' not in quantity: setattr(ax, "right_twin_count", n_right + 1)
    # ---------------------------------------------------------------------------

    ax2.set_ylabel(ylabel, fontsize=fsize*0.8, color=color, labelpad=5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.spines["right"].set_edgecolor(color)

    # PLOT
    if 'wp' in quantity or 'wc' in quantity:
        profile = np.asarray(profile, float)
        # break the line at non-positive values (important for log)
        profile[profile <= 0] = np.nan  # no zeros/negatives on log y
        if quantity == 'tot_wc': 
            profile = profile.squeeze() 
            y = ACMCOM.height_layer[:,1000]/1000  # Convert from m to km
            label_list = ylabel
            color_list = [0, 0]
            ax2.set_ylabel('Altitude [km]', fontsize=INFOSIZE)
            fig.subplots_adjust(right=1.05)   # Adjuct without tight_layout
        elif quantity == 'tot_wp': label_list = ['IWP','LWP']; color_list = [(0.0, 0.6, 0.5), (0.2, 0.4, 0.8)]
        else: profile = [profile]; label_list = [label]; color_list = [color]

        for i, (p, l, c) in enumerate(zip(profile, label_list, color_list)):
            if 'wc' in quantity: # Plot Colormesh + Colorbar
                if   'I' in l: cmap = plt.get_cmap('viridis')
                elif 'L' in l: cmap = plt.get_cmap('magma')
                else: print("ERROR in cmap")
        
                cs = ax2.pcolormesh(x,y,p, shading='auto', cmap=cmap, norm=LogNorm(WC_min, WC_max))
                ax2.set_ylabel('')        # clears the label text
                ax2.tick_params(axis='y', which='both', right=True, labelright=False, length=0) # clears ticks

                # Placement of Colorbar Axis (cax)
                cb = fig.colorbar(  cs, ax=[ax,ax2], #location='right', use_gridspec=True,
                                    shrink=0.4, pad=0.0,
                                    # extend='both',
                                    # extendrect=True,
                                    anchor=(0, 0),   # (x, y) for 'right' location -> y shifts vertically:
                                                        # 0 = bottom, 0.5 = center, 1 = top
                                    )
                cb.set_label(l, size=INFOSIZE)
                cb.ax.tick_params(labelsize=INFOSIZE*.8)
                cb.outline.set_visible(False)             # remove black frame
                cb.ax.set_facecolor('#f7f7f7')            # subtle bg behind ramp

                fig.tight_layout = lambda *args, **kwargs: None # turnes off tight_layout -> wont work later in code
                        
                
            else: # Plot Line(s)
                line, = ax2.plot(x, p,  
                            label=l, 
                            color=c,
                            linewidth=0.8,
                            markersize=1, marker='o',
                            alpha=alpha)
                legend_list.append(line) 
    else: # Add property
        line, = ax2.plot(x, profile,  
                    label=label, 
                    color=color,
                    # linestyle='--',
                    linewidth=0.8,
                    alpha=alpha)
    if 'wc' in quantity:
        pass  # no legend for tot_wc
    else:
        if 'CF' not in quantity or len(quantity_list) == 1:
            legend_list.append(line)
        else: 
            # CF gets its own legend on ax2 (only if there is something to show)
            ax2.legend( loc='upper right', framealpha=0.7, 
                        borderaxespad=0.0,                  # space to axes
                        borderpad=0.25, labelspacing=0.25,   # compact box)
                        fontsize=INFOSIZE*.8)
    
    # Set y-limits:
    if 'wp' in quantity:
        ax2.set_yscale('log')
        ax2.set_ylim(WP_min, WP_max)
    elif 'wc' in quantity: ax2.set_ylim(ymin=0,ymax=19) # altitude
    else:
        ymax, ymin = np.nanmax(profile), np.nanmin(profile)
        pad = 0.15 * (ymax-ymin)
        ax2.set_ylim(ymin, ymax + pad)
    
    return ax, ax2

    
def calculate_cloud_fraction(ACMCOM, ACM3D, want_2D=True, want_ice=False):
    # CF of whole swat
    THRESHOLD = 0 # kg/m3
    MAX = 1e30    # kg/m3
    THRESHOLD_lwp = 400  + 1e30 
    THRESHOLD_iwp = 1000 + 1e30


    
    # ----------- Quality-Status -----------------
    # Create a combined mask: quality 0 or 1
    quality = np.asarray(ACMCOM.quality_status[:])
    quality_status_mask = np.isin(quality, [0, 1])
    # ------------------------------------------------¨

    if want_2D:
        # flatten horizontal mapping
        irec_flat = ACM3D.index_construction.ravel().copy()   # shape (nx*ny)
        # Modify for cutted swats:
        if "Orbit_06888C" in SceneNames[0]: # Svaldbard swat
            irec_flat -= 2527 
            # keep only non-negative values
            irec_flat = irec_flat[irec_flat >= 0]
        elif SceneNames[0] in ["Orbit_06662C", "Orbit_06600C"]: # Greenland swats
            # keep only in-bounce values from cutting swat 
            irec_flat = irec_flat[irec_flat < ACM3D.index_construction.shape[1]] # length of along-track

        # gather arrays for those columns: shapes -> (layer_number, along_track)
        lwc = ACMCOM.liquid_water_content[:, irec_flat]     
        iwc = ACMCOM.ice_water_content[:, irec_flat]        

        # per-level valid / cloud masks (2D: levels x npoints)
        if want_ice:
            valid_level = np.isfinite(iwc) & (iwc < MAX)  
            cloud_level = (iwc > THRESHOLD) & (iwc < MAX)  
        else:
            valid_level = np.isfinite(lwc) & (lwc < MAX) & np.isfinite(iwc) & (iwc < MAX)
            cloud_level = (lwc > THRESHOLD) & (lwc < MAX) | (iwc > THRESHOLD) & (iwc < MAX)
                    
        # collapse to 1D per horizontal pixel: True if ANY level satisfies condition
        valid_mask_1d = valid_level.any(axis=0)   # shape (npoints,), boolean
        cloud_mask_1d = cloud_level.any(axis=0)   # shape (npoints,), boolean

        # apply quality filter on the selected columns
        quality_status_mask = quality_status_mask[irec_flat]             # align with selected columns
        valid_mask_1d &= quality_status_mask
        cloud_mask_1d &= quality_status_mask

        # global counts over the whole 2D swath (number of horizontal pixels)
        valid_count = int(valid_mask_1d.sum())   # how many horizontal pixels have any valid data
        cloud_count = int(cloud_mask_1d.sum())   # how many horizontal pixels have any cloud

        CF = cloud_count / valid_count if valid_count > 0 else np.nan 
            
    # CF of nadir swat (1D)
    else:
        lwc = ACMCOM.liquid_water_content    # shape (layer_number, along_track)
        iwc = ACMCOM.ice_water_content       # shape (layer_number, along_track)

        # valid pixel mask: both lwc and iwc finite                                                              
        # cloud mask: either lwc>0 or iwc>0, but only where values are valid  
        if want_ice:
            valid_level = (np.isfinite(iwc) & (iwc < MAX))  # Dimention = 2D
            cloud_level = (iwc > THRESHOLD) & (iwc < MAX)   
        else:
            valid_level = (np.isfinite(lwc) & (lwc < MAX)) & (np.isfinite(iwc) & (iwc < MAX))   # Dimention = 2D
            cloud_level = ((lwc > THRESHOLD) & (lwc < MAX)) | ((iwc > THRESHOLD) & (iwc < MAX))

        valid_mask = valid_level.any(axis=0)    # Dimention = 1D
        cloud_mask = cloud_level.any(axis=0)                                            
        # Explenation: .any(axis=0) returns True if any element is True -> at least one element True for all layer_number
        
        # apply quality filter
        valid_mask &= quality_status_mask
        cloud_mask &= quality_status_mask

        # Sum up valid- or CP-pixel mask:
        # collapse to 1D: counts per along-track column
        valid_count = valid_mask.sum(axis=0).astype(int)   # shape (along_track,)
        cloud_count = cloud_mask.sum(axis=0).astype(int)   # shape (along_track,)

        CF = cloud_count / valid_count if valid_count > 0 else np.nan

        # Caclulate All-Sky mean LWP and IWP: 
        # -------------------------------------------------------------------------------
        if not want_ice:
            z = np.asarray(ACMCOM.height_layer)
            L,N = lwc.shape
            LWP = np.full(N, 0.0)
            IWP = np.full(N, 0.0)
            for i in range(N):
                if not valid_mask[i]:
                    continue # -> invalid column, exclude from mean
                
                zi = z[:,i]
                # Create mask for valid heights
                mL = (lwc[:,i] < MAX) & np.isfinite(lwc[:, i]) & np.isfinite(zi)
                mI = (iwc[:,i] < MAX) & np.isfinite(iwc[:, i]) & np.isfinite(zi)

                # Integrate; abs() handles increasing or decreasing height
                LWP[i] = np.abs(np.trapezoid(lwc[mL, i], x=zi[mL])) if mL.any() else 0.0
                IWP[i] = np.abs(np.trapezoid(iwc[mI, i], x=zi[mI])) if mI.any() else 0.0


            # All-sky means: clear valid columns contribute 0; invalid columns are NaN
            # Update valid mask to lwp- and iwp-threshold
            valid_mask_l = valid_mask & (LWP < THRESHOLD_lwp)
            valid_mask_i = valid_mask & (IWP < THRESHOLD_iwp)
            LWP_all = np.where(valid_mask_l, LWP, np.nan)
            IWP_all = np.where(valid_mask_i, IWP, np.nan)

            LWP_mean = float(np.nanmean(LWP_all)) if np.any(valid_mask_l) else np.nan
            IWP_mean = float(np.nanmean(IWP_all)) if np.any(valid_mask_i) else np.nan

            print(f"Mean All-Sky (1D):   <LWP> = {LWP_mean:.2f} kg m^-2   <IWP> = {IWP_mean:.2f} kg m^-2")
            print(f'     All-Sky (1D): Max(LWP) = {np.nanmax(LWP_all):.2f} kg m^-2   Max(IWP) = {np.nanmax(IWP_all):.2f} kg m^-2')
            # -----------------------------------------------------------------------------

    return CF

def get_SZA(BMAFLX, property='max'):
    SZA_list = BMAFLX.solar_zenith_angle[:,1]  # Nadir view is in element one

    if 'wp' in additional_spesifications:
        lat = BMAFLX.latitude
        if SceneName == 'Orbit_06497E': 
            target_start_lat, target_end_lat = 3, 5
        elif SceneName == 'Orbit_06518D':
            target_start_lat, target_end_lat = 40,42
        elif SceneName == 'Orbit_06888C': 
            target_start_lat, target_end_lat = 68,69
        elif SceneName == 'Orbit_06331C': 
            target_start_lat, target_end_lat = 76,78
        
        start, end = int(np.nanargmin(np.abs(lat - target_start_lat))),  int(np.nanargmin(np.abs(lat - target_end_lat)))
        if start > end: tmp = start; start = end; end = tmp
        SZA_list = SZA_list[start:end]
    elif any(k in additional_spesifications for k in ('GHM', 'SC', 'RA')):
        lat = BMAFLX.latitude
        if SceneName == 'Orbit_06888C': 
             target_start_lat, target_end_lat = 68, 71
        elif SceneName == 'Orbit_06518D': 
            target_start_lat, target_end_lat = 38, 46
        elif SceneName == 'Orbit_06886E':
            target_start_lat, target_end_lat = 1,15
        elif SceneName == 'Orbit_07277C': 
            target_start_lat, target_end_lat = 70,73
        elif SceneName == 'Orbit_06907D': 
            target_start_lat, target_end_lat = 40,46
        elif SceneName == 'Orbit_06497E': 
            target_start_lat, target_end_lat = 5,15

        start, end = int(np.nanargmin(np.abs(lat - target_start_lat))),  int(np.nanargmin(np.abs(lat - target_end_lat)))
        if start > end: tmp = start; start = end; end = tmp
        SZA_list = SZA_list[start:end]
    if property == 'max':
        SZA = np.nanmax(SZA_list)
    elif property == 'mean':
        SZA = np.nanmean(SZA_list)

    return SZA






class Scene:
    def __init__(self,  Name='',  verbose=False):
        self.Name=Name
        self.verbose=verbose

        return


    def Plot2D(self, plot_type='', pngfile='', verbose=False, ACM3D=None, libRad=None):

        fsize=FONTSIZE

        projection=ccrs.PlateCarree()

        fig = plt.figure(figsize=(6.0,9.0))
        if 'C' in SceneName: fig = plt.figure(figsize=(6.0,2.0))
        if 'integrated' in plot_type: fig =  plt.figure(figsize=(5,4)) #plt.figure(figsize=(5,4))
        fig.subplots_adjust(left=0.11, right=0.88,bottom=0.06)

        ax = fig.add_subplot(1,1,1, projection=projection)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle='-')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        gl=ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels=False
        ax.set_extent([self.extent_left, self.extent_right, self.extent_bottom,  self.extent_top], crs=ccrs.PlateCarree())

        x = self.longitude
        y = self.latitude
        cblabel = plot_type
        title = ''

        cmap = plt.get_cmap('viridis')  #cmap = plt.get_cmap('jet')
        THRESHOLD_lwc = 1e30 
        THRESHOLD_iwc = 1e30
        THRESHOLD_lwp = 400 
        THRESHOLD_iwp = 1000

        if plot_type=='index_construction':
            data = self.index_construction
            vmin=data.min()
            vmax=data.max() 
        elif plot_type=='integrated_iwc':
            title = 'Ice Water Path'
            cblabel = r"IWP [$kg/m^2$]"
            data = ACM3D.index_construction * 0.0
            nx = data.shape[0]
            ny = data.shape[1]

            ix=0
            while ix < nx:
                iy=0
                while iy < ny:
                    il = ACM3D.index_construction[ix,iy]

                    if "Orbit_05926C" in SceneName:
                        il -= 2700 
                        if il < 0: il = 0
                    elif "Orbit_06888C" in SceneName: # Svaldbard swat
                        il -= 2527 
                        # keep only non-negative values
                        if il < 0:
                            iy += 1
                            continue
                    elif "Orbit_06662C" in SceneName or "Orbit_06600C" in SceneName: # Greenland swats
                        # keep only in-bounce values from cutting swat 
                        if il >= ny:
                            iy += 1
                            continue
                    
                    tmpx = self.height_layer[:,il]  #height_level(level_number, along_track)
                    tmpy = self.ice_water_content[:,il] #liquid_water_content(atmosphere, layer_number, along_track)
                    # indy = np.where((tmpx>=0)&(tmpx<1.0e+35)&(tmpy!=np.inf))
                    indy = np.where((tmpx>=0)&(tmpx<1.0e+35)&(tmpy!=np.inf) & (tmpy < THRESHOLD_iwc))
                    totcol = -np.trapz(tmpy[indy], x=tmpx[indy])  # negative since x (height) is decreasing
                    if not np.isfinite(totcol) or totcol > THRESHOLD_iwp: totcol=0.0
                    data[ix,iy] = totcol

                    iy=iy+1
                ix=ix+1
            vmin=0.1 #data.min()
            vmax=THRESHOLD_iwp  #5000 #data.max()


            # ----------- Add nadir-swat -----------------
            nadir_idx = 151
            ax.plot(x[nadir_idx,:], y[nadir_idx,:],'--',                
                    linewidth=1,
                    color='r',
                    label='Satellite trajectory',
                    transform=projection,   # e.g. ccrs.PlateCarree()
                    zorder=10)
            # ---------------------------------------------
            # # ---------- Add 3D-Buffers  ------------------
            # nx, ny = data.shape
            # i_c = nadir_idx # center: nadir row,
            # j_c = int(ny/1.63) # int(ny / 1.52) # ny // 2 + ny // 3 # center: middle column
            #                     # Small Buffer       Large Buffer
            # for (H, W, color, label) in [(13, 13, 'black',  'Small Buffer'),(25, 21, 'dimgray', 'Large Buffer')]:  # 'saddlebrown' and 'darkred' works
            #     i0 = i_c - H // 2  
            #     j0 = j_c - 9*W // 2
            #     edges = [
            #         (x[i0,        j0:j0+W],   y[i0,        j0:j0+W]),   # top
            #         (x[i0:i0+H,   j0+W-1],    y[i0:i0+H,   j0+W-1]),    # right
            #         (x[i0+H-1,    j0:j0+W],   y[i0+H-1,    j0:j0+W]),   # bottom
            #         (x[i0:i0+H,   j0],        y[i0:i0+H,   j0])         # left
            #     ]
            #     for k, (X, Y) in enumerate(edges):
            #         ax.plot(X, Y, '-', color=color, lw=1.5, transform=projection, zorder=20,
            #                     label=label if k == 0 else '_nolegend_')
            # ax.legend()
            # # ---------------------------------------------
        

        elif plot_type=='integrated_lwc':
            title = 'Liquid Water Path'
            cblabel = r"LWC [$kg/m^2$]"
            data = ACM3D.index_construction * 0.0
            nx = data.shape[0]
            ny = data.shape[1]
            ix=0
            while ix < nx:
                iy=0
                while iy < ny:
                    il = ACM3D.index_construction[ix,iy]

                    if "Orbit_05926C" in SceneName:
                        il -= 2700 
                        if il < 0: il = 0
                    elif "Orbit_06888C" in SceneName: # Svaldbard swat
                        il -= 2527 
                        # keep only non-negative values
                        if il < 0:
                            iy += 1
                            continue
                    elif "Orbit_06662C" in SceneName or "Orbit_06600C" in SceneName: # Greenland swats
                        # keep only in-bounce values from cutting swat 
                        if il >= ny:
                            iy += 1
                            continue
                    

                    tmpx = self.height_layer[:,il]  #height_level(level_number, along_track)
                    tmpy = self.liquid_water_content[:,il] #liquid_water_content(atmosphere, layer_number, along_track)
                    indy = np.where((tmpx>=0)&(tmpx<1.0e+35)&(tmpy!=np.inf) & (tmpy!=np.nan) & (tmpy < THRESHOLD_lwc))
                    totcol = -np.trapz(tmpy[indy], x=tmpx[indy])  # negative since x (height) is decreasing
                    if (not np.isfinite(totcol)) or totcol > THRESHOLD_lwp: totcol=0.0
                    data[ix,iy] = totcol
                    iy=iy+1
                ix=ix+1
            vmin=0.0001 #data.min()
            vmax=THRESHOLD_lwp # 250 #2000 #data.max()

            # ----------- Add nadir-swat -----------------
            nadir_idx = 151
            ax.plot(x[nadir_idx,:], y[nadir_idx,:],'--',                
                    linewidth=1,
                    color='r',
                    label='Satellite trajectory',
                    transform=projection,   # e.g. ccrs.PlateCarree()
                    zorder=10)
            # ---------------------------------------------
            # # ---------- Add 3D-Buffers  ------------------
            # nx, ny = data.shape
            # i_c = nadir_idx # center: nadir row,
            # j_c = int(ny/1.63) # int(ny / 1.52) # ny // 2 + ny // 3 # center: middle column
            #                     # Small Buffer       Large Buffer
            # for (H, W, color, label) in [(13, 13, 'black',  'Small Buffer'),(25, 21, 'dimgray', 'Large Buffer')]:  # 'saddlebrown' and 'darkred' works                i0 = i_c - H // 2  
            #     i0 = i_c - H // 2  
            #     j0 = j_c - 9*W // 2
            #     edges = [
            #         (x[i0,        j0:j0+W],   y[i0,        j0:j0+W]),   # top
            #         (x[i0:i0+H,   j0+W-1],    y[i0:i0+H,   j0+W-1]),    # right
            #         (x[i0+H-1,    j0:j0+W],   y[i0+H-1,    j0:j0+W]),   # bottom
            #         (x[i0:i0+H,   j0],        y[i0:i0+H,   j0])         # left
            #     ]
            #     for k, (X, Y) in enumerate(edges):
            #         ax.plot(X, Y, '-', color=color, lw=1.5, transform=projection, zorder=20,
            #                     label=label if k == 0 else '_nolegend_')
            # ax.legend()
            # # ---------------------------------------------

        elif plot_type=='albedo_direct_radiation_surface_visible':
            data = self.albedo_direct_radiation_surface_visible
            vmin=0 #data[indx].min()#220 #
            vmax=1 #data[indx].max()#320 #
        elif plot_type=='albedo_direct_radiation_surface_near_infrared':
            data = self.albedo_direct_radiation_surface_near_infrared
            vmin=0 #data[indx].min()#220 #
            vmax=1 #data[indx].max()#320 #
        elif plot_type=='albedo_diffuse_radiation_surface_visible':
            data = self.albedo_direct_radiation_surface_visible
            vmin=0 #data[indx].min()#220 #
            vmax=1 #data[indx].max()#320 #
        elif plot_type=='albedo_diffuse_radiation_surface_near_infrared':
            data = self.albedo_direct_radiation_surface_near_infrared
            vmin=0 #data[indx].min()#220 #
            vmax=1 #data[indx].max()#320 #
        elif plot_type=='surface_temperature':
            data = self.surface_temperature
            indx = np.where(data<1e+35)
            vmin=data[indx].min()#220 #
            vmax=data[indx].max()#320 #
        
        elif plot_type == 'plot_swat':
            data = self.surface_temperature * 0
            data[147:155,:] = 1 # i = 151 = Nadir-pixel-index
            plot_type = ''
            vmin = -1.5; vmax = 1
            plt.title(self.Name, size=fsize)

            cmap = plt.get_cmap('coolwarm') 

        else:
            print("Unknown plot_type: "+plot_type+", exiting")
            exit()

                ############################ OLD #################################
                # print('data', vmin, vmax, data.min(), data.max())
                # cs = ax.pcolor(x, y, data, cmap=cmap, transform=projection, vmin=vmin, vmax=vmax, linewidth=0)
                # if plot_type!='':
                #     cb = plt.colorbar(cs, shrink=1.,  fraction=0.02, pad=0.03) # BG: change fraction to change cb-height
                #     cblabel=cblabel
                #     cb.ax.tick_params(labelsize=fsize)
                #     cb.set_label(cblabel, size=fsize)
                #     title = self.Name + title
                #     plt.title(title, size=fsize)
                #     cmap.set_under('gray')
                # if plot_type=='albedo_diffuse_radiation_surface_visible' or plot_type=='albedo_diffuse_radiation_surface_near_infrared':
                #         cmap.set_over('gray')
                # # BG: how to add points on 2D plot
                # # ax.plot(x[151, 2000], y[151, 2000], 'rx', markersize=20, markeredgewidth=2, transform=ccrs.PlateCarree())

                #        if plot_type=='index_construction':
        



        if libRad != None:
            valid_mask = (libRad.solar_eup != 0)

            # subset columns
            x_valid    = x[:, valid_mask]
            y_valid    = y[:, valid_mask]
            data_valid = data[:, valid_mask]

            # set both lat & lon limits from the subset
            xmin = np.nanmin(x_valid); xmax = np.nanmax(x_valid)
            ymin = np.nanmin(y_valid); ymax = np.nanmax(y_valid)
            padx, pady = 0.2, 0.2

            # Orbit 06518D specs:
            # xmin, xmax, ymin, ymax = -99.5, -97.5, 39, 41

            ax.set_xlim(xmin - padx, xmax + padx)   # longitude
            ax.set_ylim(ymin - pady, ymax + pady)   # latitude

            print('Mean data over valid-region = ', data_valid.mean())

        
        ax.set_aspect('auto')  # let x/y stretch freely

        if 'wc' in plot_type:
            cs = ax.pcolormesh(x, y, data, shading='auto', cmap=cmap, norm=LogNorm(WP_min, WP_max))
        else: 
            cs = ax.pcolormesh(x, y, data, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        # Under/bad colors for cleaner look
        cs.cmap.set_under('#f0f0f0')
        cs.cmap.set_bad('#f0f0f0') # NaNs

        # Colorbar
        cb = fig.colorbar(
            cs, ax=ax, 
            shrink=1, pad=0.0, 
            extend='both',
            extendrect=True)
        cb.set_label(cblabel, size=INFOSIZE)
        cb.ax.tick_params(labelsize=INFOSIZE)

        # soften colorbar box + background
        cb.outline.set_visible(False)             # remove black frame
        cb.ax.set_facecolor('#f7f7f7')            # subtle bg behind ramp

        ax.set_xlabel(r'Latitude [N$^\circ$]', fontsize=INFOSIZE)
        ax.set_ylabel('Altitude [km]', fontsize=INFOSIZE)
        
        # ax.xaxis.set_major_locator(MultipleLocator(2.0))  # one tick every 2°
        # ax.xaxis.set_minor_locator(MultipleLocator(2.0))
        
        #----------------------------------- Plot Settings -------------------------
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
        ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
        fig.suptitle(title, fontsize=FONTSIZE, y=0.98, fontweight='bold')
        ax.set_title(f"{self.Name} - {date}  {time} (UTC)", fontsize=INFOSIZE)
        plt.figtext(0.001, 0.003, f"Baselines: AC = ({AC_baseline})  |  BA = ({BA_baseline})", fontsize=FONTSIZE*0.45)
        # Orbit nr: self.Name
        

        # BG: ----- plot-adjustments for nicer looking plots -----------
        # Axes background (warm light grey)
        ax.set_facecolor('#f0f0f0')
        # Grid: major dashed, minor dotted
        ax.grid(which='major', linestyle='--', alpha=0.4)
        ax.grid(which='minor', linestyle=':',  alpha=0.2)
        # ax.minorticks_on()

    
        # remove top/right border
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        
        fig.tight_layout()
        # ----------------------------------------------------------------------------

       

        if pngfile==None or pngfile=='':
            plt.show()
        else:
            if verbose:
                print("pngfile", pngfile)
            plt.savefig(pngfile)
            plt.close()
        return



    def PlotCurtain(self, plot_type='', pngfile='', verbose=False, libRad=None):

        fsize=FONTSIZE

        fig = plt.figure(figsize=(10,4))
        # fig.subplots_adjust(left=0.11, right=0.88,bottom=0.06)

        ax = fig.add_subplot(1,1,1)
        # ax.set_extent([self.extent_left, self.extent_right, self.extent_bottom,  self.extent_top], crs=ccrs.PlateCarree())

        x = self.latitude_active
        y = self.height_layer[:,1000]/1000  # Convert from m to km
        if verbose:
            print('height_layer min/max',self.height_layer.min(), self.height_layer.max())

        cblabel = plot_type
        xmin = self.latitude.min()
        xmax = self.latitude.max()
        units=''
        if plot_type=='aerosol_extinction':
            data = self.aerosol_extinction
            vmin=0.000000001 #data.min()
            vmax=0.1 #data.max()
            units=self.aerosol_extinction_units
        elif plot_type=='ice_effective_radius':
            data = self.ice_effective_radius
            vmin=0.0001 #data.min()
            vmax=100 #data.max()
            units=self.ice_effective_radius_units

        elif plot_type=='ice_water_content':
            title = 'Ice Water Content'
            cblabel = r"IWC [$kg/m^3$]"
            data = self.ice_water_content
            data = np.where((data >= 1e30) | (data == 0), np.nan, data)
            vmax = np.nanmax(data) #+ 0.2  # np.nanmax(data)
            vmin = 0 # 0.0001 #data.min()
        elif plot_type=='liquid_water_content':
            title = 'Liquid Water Content'
            cblabel = r"LWC [$kg/m^3$]"
            data = self.liquid_water_content
            # Mask values that are invalid
            data = np.where((data >= 1e30) | (data == 0), np.nan, data)
            vmax = np.nanmax(data) #+ 3 #np.nanmax(data)
            vmin = 0 # 0.0001 

        elif plot_type=='specific_humidity_layer_mean':
            data = np.log10(self.specific_humidity_layer_mean)
            vmin=-6 #data.min()
            vmax=-1 #data.max()
            units='log$_{10} $'+self.specific_humidity_layer_mean_units.decode('UTF-8')
        else:
            print("Unknown plot_type: "+plot_type+", exiting")
            exit()

            # print(x.shape, y.shape, data.shape, self.latitude_active.shape)
            # print('xmin, xmax ymin, ymax, datamin, datamax', xmin, xmax, x.min(), x.max(), y.min(), y.max(), data.min(), data.max())
            # BG: mark out given old system
            # if units != '':
            #     cblabel = cblabel + ' ('+units+')'
        
            # cs = ax.pcolor(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0)

            # cmap.set_under('white')
            # cb = plt.colorbar(cs, shrink=0.75, extend='both', fraction=0.07, pad=0.03)
            # cblabel=cblabel
            # cb.set_label(cblabel, size=fsize)
            # cb.ax.tick_params(labelsize=fsize)

        # Set x- and y-limits
        if libRad != None:
            libRad_lat  = libRad.latitude
            libRad_flux = libRad.solar_eup
            mask = (libRad_flux != 0)
            xmax, xmin = libRad_lat[mask].max(), libRad_lat[mask].min()
            pad = 0.05 * (xmax-xmin)
            xmin = xmin - pad
            xmax = xmax + pad
            print('xmin, xmax =', xmin, xmax)
        ax.set_xlim(xmin=xmin,xmax=xmax)
        ax.set_ylim(ymin=0,ymax=20) # altitude

        
    
        # Pcolormesh - turn on shading to avoid grid lines
        if 'water_content' in plot_type:
            cs = ax.pcolormesh(x, y, data, shading='auto', cmap=cmap, norm=LogNorm(WC_min, WC_max))
        else: 
            cs = ax.pcolormesh(x, y, data, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        # Under/over/bad colors for cleaner look
        cs.cmap.set_under('#f0f0f0')
        cs.cmap.set_bad('#f0f0f0') # NaNs

        cb = fig.colorbar(
            cs, ax=ax, 
            shrink=1, pad=0.0, 
            extend='both',
            extendrect=True)
        cb.set_label(cblabel, size=INFOSIZE)
        cb.ax.tick_params(labelsize=INFOSIZE)
           
        # soften colorbar box + background
        cb.outline.set_visible(False)             # remove black frame
        cb.ax.set_facecolor('#f7f7f7')            # subtle bg behind ramp

        ax.set_xlabel(r'Latitude [N$^\circ$]', fontsize=INFOSIZE)
        ax.set_ylabel('Altitude [km]', fontsize=INFOSIZE)
            
        
        ax.xaxis.set_major_locator(MultipleLocator(1.0))  # one tick every 1°
        #----------------------------------- Plot Settings -------------------------
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax.tick_params(axis='both', which='major', labelsize=INFOSIZE)
        ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE)
        fig.suptitle(title, fontsize=FONTSIZE, y=0.98)
        ax.set_title(f"{self.Name} - {date}  {time} (UTC)", fontsize=INFOSIZE)
        plt.figtext(0.001, 0.003, f"Baselines: AC = ({AC_baseline})  |  BA = ({BA_baseline})", fontsize=FONTSIZE*0.45)
        # Orbit nr: self.Name
        

        # BG: ----- plot-adjustments for nicer looking plots -----------
        fig.tight_layout()
        # Axes background (warm light grey)
        ax.set_facecolor('#f0f0f0')
        # Grid: major dashed, minor dotted
        ax.grid(which='major', linestyle='--', alpha=0.4)
        ax.grid(which='minor', linestyle=':',  alpha=0.2)
        ax.minorticks_on()

        # remove top/right border
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        # ----------------------------------------------------------------------------
        
        

        if pngfile==None or pngfile=='':
            plt.show()
        else:
            if verbose:
                print("pngfile", pngfile)
            plt.savefig(pngfile)
            plt.close()

        return




    def PlotLine(self, plot_type='', pngfile='', verbose=False, Scene2 = None, Scene3 = None, Scene4 = None):

        fsize=FONTSIZE

        # fig = plt.figure(figsize=(15.0,5.0))
        # BG: test different figszie
        fig = plt.figure(figsize=(10,4))
        # if plot_type=='plot_info':
        #     fig = plt.figure(figsize=(13,4))
        if 'correlation' in plot_type:
            fig = plt.figure(figsize=(7,6))
        fig.subplots_adjust(left=0.11, right=0.88,bottom=0.06)

        ax = fig.add_subplot(1,1,1)
        #        ax.set_extent([self.extent_left, self.extent_right, self.extent_bottom,  self.extent_top], crs=ccrs.PlateCarree())

        # BG: define window-size for average-meassurements
        w_size = 20

        x = self.latitude
        xlabel = "Latitude"; xlabel_specs = r" [N$^\circ$]"
        ylabel= r"$F_{\mathrm{TOA}}^{\uparrow}$"; ylabel_specs = r" [W/m$^2$]"

    

        title=''    
        cblabel = plot_type
        cmap = plt.get_cmap('viridis')  # 'viridis' 'inferno'     cmap = plt.get_cmap('jet')
        xmin = self.latitude.min() #-90 #
        xmax = self.latitude.max() #90 #
        units=''
        
        pl_list = []
        if plot_type=='solar_zenith_angle':
            data = self.solar_zenith_angle
            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =0
            ymax =100
            units=self.solar_zenith_angle_units
        elif plot_type=='solar_azimuth_angle':
            data = self.solar_azimuth_angle
            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =0
            ymax =360
            units=self.solar_zenith_angle_units
        elif plot_type=='solar_combined_top_of_atmosphere_flux':
            data = self.solar_combined_top_of_atmosphere_flux
            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =0
            ymax =500 #tms ori
            ymax =700
            units=self.solar_combined_top_of_atmosphere_flux_units
        elif plot_type=='solar_combined_top_of_atmosphere_flux_quality_status':
            data = self.solar_combined_top_of_atmosphere_flux_quality_status
            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =0
            ymax =3
        elif plot_type=='albedo_direct_radiation_surface_visible':
            title=': surface albedo'    
            cblabel='albedo_direct_radiation_surface_visible'
            x = self.latitude[0,:]
            data = self.albedo_direct_radiation_surface_visible[0,:]
            data2 = self.albedo_diffuse_radiation_surface_visible[0,:]            
            p,=ax.plot(x, data2, color='black', label='albedo_diffuse_radiation_surface_visible')
            pl_list.append(p)
            ymin=0 #data[indx].min()#220 #
            ymax=1 #data[indx].max()#320 #
            ax.set_ylabel('Albedo', fontsize=fsize)
        elif plot_type=='aerosol_column':
            il=0
            totcols=np.zeros(self.height_layer.shape[1])
            while il<self.height_layer.shape[1]:
                tmpx = self.height_layer[:,il]  #height_level(level_number, along_track)
                tmpy = self.aerosol_extinction[:,il]/1000 # Convert from 1/km to 1/m
#                indy = np.where(tmpx>=0)
                indy = np.where((tmpx>=0)&(tmpx<1.0e+35)&(tmpy!=np.inf))
                totcol = -np.trapz(tmpy[indy], x=tmpx[indy])  # negative since x (height) is decreasing
                if not np.isfinite(totcol): totcol=0.0
#                print(il, totcol, tmpx, tmpy)
                totcols[il] = totcol
                il=il+1

            x = self.latitude[0,:]
            ax.set_ylabel('Aerosol extinction', fontsize=fsize)
#            data = np.where(totcols>0, np.log10(totcols), -999990.0)
            data = totcols
            print('mabba', data.min(), data.max())
            ymin=data.min()#-1 #data.min()#0.0001 #
            ymax=data.max()#4 #
        elif plot_type=='ice_water_column':
            il=0
            totcols=np.zeros(self.height_layer.shape[1])
            while il<self.height_layer.shape[1]:
                tmpx = self.height_layer[:,il]  #height_level(level_number, along_track)
                tmpy = self.ice_water_content[:,il] #liquid_water_content(atmosphere, layer_number, along_track)
                indy = np.where((tmpx>=0)&(tmpx<1.0e+35)&(tmpy!=np.inf))
                totcol = -np.trapz(tmpy[indy], x=tmpx[indy])  # negative since x (height) is decreasing
                if not np.isfinite(totcol): totcol=0.0
                # print('gabba', il, totcol, self.latitude[0,il]) #, tmpx, tmpy)
                # print( il, totcol, self.latitude[0,il], tmpx, tmpy)
                totcols[il] = totcol
                il=il+1

            x = self.latitude[0,:]
            log10=True #False# 
            if log10:
                data = np.where(totcols>0, np.log10(totcols), -999990.0)
                ax.set_ylabel('Ice water column (log$_{10}$ g/m$^2$)', fontsize=fsize)
                ymin=-1 #data.min()#0.0001 #
                ymax=data.max()#4 #
            else:
                ax.set_ylabel('Ice water column (g/m$^2$)', fontsize=fsize)
                data = totcols
                ymin= 0
                ymax=data.max()#4 #

            print('mabba', data.min(), data.max())
        elif plot_type=='liquid_water_column':
            il=0
            totcols=np.zeros(self.height_layer.shape[1])
            while il<self.height_layer.shape[1]:
                tmpx = self.height_layer[:,il]  #height_level(level_number, along_track)
                tmpy = self.liquid_water_content[:,il] #liquid_water_content(atmosphere, layer_number, along_track)
                #                indy = np.where(tmpx>=0)
                indy = np.where((tmpx>=0)&(tmpx<1.0e+35)&(tmpy!=np.inf))
                totcol = -np.trapz(tmpy[indy], x=tmpx[indy])  # negative since x (height) is decreasing
                if not np.isfinite(totcol): totcol=0.0
                # print('gabba', il, totcol, self.latitude[0,il]) #, tmpx, tmpy)
                # print( il, totcol, self.latitude[0,il], tmpx, tmpy)
                totcols[il] = totcol
                il=il+1

            x = self.latitude[0,:]
            log10=True #False
            if log10:
                ax.set_ylabel('Liquid water column (log$_{10}$ g/m$^2$)', fontsize=fsize)
                data = np.where(totcols>0, np.log10(totcols), -999990.0)
            else:
                ax.set_ylabel('Liquid water column ( g/m$^2$)', fontsize=fsize)
                data = totcols
                
            # print('mabba', data.min(), data.max())
            ymin= -1 #data.min()#0.0001 #
            ymax= 4 # data.max()#4 #
        elif plot_type=='solar_diff':
            data = self.solar_combined_top_of_atmosphere_flux[Scene2.BMAFLXindlats] - Scene2.solar_eup
            x = self.latitude[Scene2.BMAFLXindlats]
            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =-150
            ymax =150
            ax.plot(x, data*0.0, color='black')

        # BG: these brantches used often ------------------------------------------------------------------------------------------------------
        elif plot_type=='plot_info':
            title='Atmospheric and Surface Properties'    
            
            data = self.solar_combined_top_of_atmosphere_flux * np.nan
            x = ACMCOM.latitude_active

            for quantity in quantity_list:
                ax2 = add_profile_to_plot(fig, ax, ACMCOM, fsize=fsize, legend_list=pl_list, quantity=quantity, stacked=False)
                
           

            vmin = 0 #0.0001 #
            vmax = 1
            ymin = 0
            ymax = 1# max((v for v in profile if np.isfinite(v))) #1
            

        elif plot_type=='solar_both':
            # INFO: 
            #   self   = BMAFLX
            #   Scene2 = librad
            #   Scene3 = ACMRT
            #   Scene4 = librad2
            title='Solar TOA flux' #+ f' - Atmosphere {atmosphere} (ACM-COM)'
            cblabel='BMA_FLX: solar_combined_top_of_atmosphere_flux'
                

            # ---------- BG: Get additional data to plot (twin or stacked-axis) ----------------
            if quantity_list:
                add_profile_list = [] # List for ax2-legends
                for quantity in quantity_list:
                    ax, ax2 = add_profile_to_plot(fig, ax, ACMCOM, fsize=fsize, legend_list=add_profile_list, quantity=quantity, stacked=True)
                if add_profile_list: ax2.legend(handles=add_profile_list, 
                                                loc='upper right', framealpha=0.7, 
                                                borderaxespad=0.0,                  # space to axes
                                                borderpad=0.25, labelspacing=0.25,   # compact box)
                                                fontsize=INFOSIZE*.8)
            # ------------------------------------------------------------------------


            x = self.latitude 
            x2 = Scene2.latitude

            data = self.solar_combined_top_of_atmosphere_flux

            # ----------- Quality-Status -----------------
            quality = self.quality_status[:]
            # boolean mask: True where quality is 0 or 2
            quality_mask = np.isin(quality, [0, 2]) 

            # Apply the mask to data
            x[~quality_mask], data[~quality_mask] = np.nan, np.nan
            # ------------------------------------------------


            # bbr_direction=1
            # data4 = self.solar_top_of_atmosphere_flux[:,bbr_direction]
            data2 = Scene2.solar_eup

            # ----------- Quality-Status -----------------
            # BG: modification to only include calculated results with valid quality
            # Create a combined mask: non-zero and quality 0 or 1
            quality = ACMCOM.quality_status[:]
            mask = (data2 != 0) & np.isin(quality, [0, 1])

            # Apply the mask to data
            x2[~mask], data2[~mask] = np.nan, np.nan
            # ------------------------------------------------

            # Print info:
            print(f'\n|--------{plot_type:.14} FLUX MYSTIC---------------|\n'
                    f'|mean, min, max  = {np.nanmean(data2):.2f} & {np.nanmin(data2):.2f} & {np.nanmax(data2):.2f}')

            p,=ax.plot(x2, data2, color='blue', label='libRadtran, '+librad_version,
                        # marker=".", markersize = 4, linestyle="None", 
                        alpha=.9,
                        linewidth=1.2,
                        zorder=10)
            pl_list.append(p)

            
            if want_average_line:
                # BG: original w = 21 -> change to 3
                # BG: if calculated fewer than 3 pixels -> must change data3
                data3=  moving_average(data2, w=w_size)  # Assessment_domain_along_size = 21
                p,=ax.plot(x2[w_size:-w_size], data3[w_size:-w_size],
                        color='#007FFF',
                        linewidth=2, 
                        zorder=9,
                        alpha=.7,
                        label='libRadtran, averaged')
                pl_list.append(p) 

            
            if Scene4 != None:
                if librad_version == 'disort_1D': librad_version2 = 'montecarlo_3D'
                else: librad_version2 = 'disort_1D'

                x4 = Scene4.latitude 
                data4 = Scene4.solar_eup
                # ----------- Quality-Status -----------------
                # BG: modification to only include calculated results with valid quality
                # Create a combined mask: non-zero and quality 0 or 1
                quality = ACMCOM.quality_status[:]
                mask = (data4 != 0) & np.isin(quality, [0, 1])
                
                # Apply the mask to data
                x4[~mask], data4[~mask] = np.nan, np.nan
                # ------------------------------------------------
                p,=ax.plot(x4, data4, color='green', label='libRadtran, '+librad_version2,
                    #    marker=".",
                    #    markersize = 2, 
                    #    linestyle="None", 
                       alpha=.5,
                       zorder=5)
                pl_list.append(p)
            if Scene3 != None: 
                x3 = Scene3.latitude_active
                itoa=0
                data3 = Scene3.flux_up_solar_1d_all_sky[:,itoa]
                data4 = Scene3.flux_up_solar_3d_all_sky[:,itoa]

                # ----------- Quality-Status -----------------
                quality = Scene3.quality_status[0,:]
                # boolean mask: True where quality is 0 or 1
                quality_mask = np.isin(quality, [0, 1]) 

                data3 = data3[quality_mask]
                data4 = data4[quality_mask]
                x3 = x3[quality_mask]
                # ------------------------------------------------

                p,=ax.plot(x3, data3, color='yellow', label='ACM_RT 1d: flux_up_solar_1d_all_sky, TOA')
                pl_list.append(p)

                p,=ax.plot(x3, data4, 
                        color='black', # '#00A028'
                        marker = 'x', 
                        markersize=8,
                        linestyle="None", 
                        zorder=8, 
                        label='ACM_RT 3d: flux_up_solar_3d_all_sky, TOA')
                pl_list.append(p) 
            


            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =0
            ymax =500
            ymax =1000
            # ax.set_ylabel(r"$F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]", fontsize=INFOSIZE) #('TOA upward flux [W/m$^2$]', fontsize=fsize)


        elif plot_type=='thermal_both':
            title='Thermal TOA flux' #+ f' - Atmosphere {atmosphere} (ACM-COM)'
            cblabel='BMA_FLX: thermal_combined_top_of_atmosphere_flux'

            # ---------- BG: Get additional data to plot (twin or stacked-axis) ----------------
            if quantity_list:
                add_profile_list = [] # List for ax2-legends
                for quantity in quantity_list:
                    ax, ax2 = add_profile_to_plot(fig, ax, ACMCOM, fsize=fsize, legend_list=add_profile_list, quantity=quantity, stacked=True)
                if add_profile_list: ax2.legend(handles=add_profile_list, 
                                                loc='upper right', framealpha=0.7, 
                                                borderaxespad=0.0,                  # space to axes
                                                borderpad=0.25, labelspacing=0.25,   # compact box)
                                                fontsize=INFOSIZE*.8)
            # ------------------------------------------------------------------------


            x = self.latitude
            x2 = Scene2.latitude
            bbr_direction=1
            data = self.thermal_combined_top_of_atmosphere_flux        

            # ----------- Quality-Status -----------------
            quality = self.quality_status[:]
            # boolean mask: True where quality is 0 or 2
            quality_mask = np.isin(quality, [0, 1]) 

            # Apply the mask to data
            x[~quality_mask], data[~quality_mask] = np.nan, np.nan
            # ------------------------------------------------
  


            data2 = Scene2.thermal_eup
            # ----------- Quality-Status -----------------
            # BG: modification to only include calculated results with valid quality
            # Create a combined mask: non-zero data2 and quality 0 or 1
            quality = ACMCOM.quality_status[:]
            mask = (data2 != 0) & np.isin(quality, [0, 1])

            # Apply the mask to data
            x2[~mask], data2[~mask] = np.nan, np.nan
            # ------------------------------------------------
            # Print info:
            print(f'|--------{plot_type:.14} FLUX MYSTIC-------------|\n'
                  f'|mean, min, max  = {np.nanmean(data2):.2f} & {np.nanmin(data2):.2f} & {np.nanmax(data2):.2f}\n')

            p,=ax.plot(x2, data2, color='blue', label='libRadtran, '+librad_version,
                        # marker=".", markersize = 4, linestyle="None", 
                        alpha=.9,
                        linewidth=1.2,
                        zorder=10)
            pl_list.append(p)

            
            if want_average_line:
                # BG: original w = 21 -> change to 3
                # BG: if calculated fewer than 3 pixels -> must change data3
                data3=  moving_average(data2, w=w_size)  # Assessment_domain_along_size = 21
                p,=ax.plot(x2[w_size:-w_size], data3[w_size:-w_size],
                        color='#007FFF',
                        linewidth=2, 
                        zorder=9,
                        alpha=.7,
                        label='libRadtran, averaged')
                pl_list.append(p) 
            

            if Scene4 != None:
                if librad_version == 'disort_1D': librad_version2 = 'montecarlo_3D'
                else: librad_version2 = 'disort_1D'

                x4 = Scene4.latitude 
                data4 = Scene4.thermal_eup
                # ----------- Quality-Status -----------------
                # BG: modification to only include calculated results with valid quality
                # Create a combined mask: non-zero and quality 0 or 1
                quality = ACMCOM.quality_status[:]
                mask = (data4 != 0) & np.isin(quality, [0, 1])

                # Apply the mask to data
                x4[~mask], data4[~mask] = np.nan, np.nan
                # ------------------------------------------------
                p,=ax.plot(x4, data4, color='green', label='libRadtran, '+librad_version2,
                    #    marker=".",
                    #    markersize = 2, 
                    #    linestyle="None", 
                       alpha=.5,
                       zorder=5)
                pl_list.append(p)
            if Scene3 != None:
                x3 = Scene3.latitude_active
                itoa=0
                data3 = Scene3.flux_up_thermal_1d_all_sky[:,itoa]
                data4 = Scene3.flux_up_thermal_3d_reference_height_all_sky[:]

                # ----------- Quality-Status -----------------
                quality = Scene3.quality_status[0,:]
                # boolean mask: True where quality is 0 or 1
                quality_mask = np.isin(quality, [0, 1]) 

                data3 = data3[quality_mask]
                data4 = data4[quality_mask]
                x3 = x3[quality_mask]
                # ------------------------------------------------

                p,=ax.plot(x3, data3, color='yellow', label='ACM_RT 1d: flux_up_thermal_1d_all_sky, TOA')
                pl_list.append(p)
            
                
                p,=ax.plot(x3, data4, 
                        color='black', # '#00A028'
                        marker = 'x', 
                        markersize=8,
                        linestyle="None", 
                        zorder=8, 
                        label='ACM_RT 3d: flux_up_thermal_3d_all_sky, TOA')
                pl_list.append(p)
                

            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =0
            ymax =500
            ymax =400
            # ax.set_ylabel(r"$F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]", fontsize=INFOSIZE) #('TOA upward flux [W/m$^2$]', fontsize=fsize)     
    

        
        # BG: made changes to this branch
        elif plot_type=='librad_solar_flx_diff':  
            if 'disort_pseudospherical' in Scene2.fn and 'montecarlo' in self.fn:
                label = '[MYSTIC - DISORT] (ps)'
            elif 'montecarlo' in Scene2.fn and 'disort_pseudospherical' in self.fn:
                label = 'DISORT (ps) - MYSTIC'
            elif 'disort' in Scene2.fn and 'montecarlo' in self.fn:
                label = '[MYSTIC - DISORT]'
                cblabel = r"$F_{\mathrm{3D}} - F_{\mathrm{1D}}$"
            elif 'montecarlo' in Scene2.fn and 'disort' in self.fn:
                label = '[DISORT - MYSTIC]'
                cblabel = r"$F_{\mathrm{1D}} - F_{\mathrm{3D}}$"
            # ------- Sensitivity analysis specs ------------------
            elif 'SmallBuffer' in self.fn:
                label = 'MYSTIC - 3D Buffer Size Sensitivity'
                cblabel = r"$F_{\mathrm{SmallBuffer}} - F_{\mathrm{LargeBuffer}}$"
            elif 'FullBuffer' in self.fn:
                label = 'MYSTIC - 3D Buffer Size Sensitivity'
                cblabel = r"$F_{\mathrm{LargeBuffer}} - F_{\mathrm{SmallBuffer}}$"
            elif 'GHM' in self.fn or 'RA' in self.fn or 'SC' in self.fn:
                if 'montecarlo' in self.fn:
                    label = 'MYSTIC - '     
                else:
                    label = 'DISORT - ' 
                if 'RA' in self.fn:
                    label += 'Ice-Habit Sensitivity [RA vs GHM]' 
                    cblabel = r"$F_{\mathrm{RA}} - F_{\mathrm{GHM}}$"
                elif 'SC' in self.fn:
                    label += 'Ice-Habit Sensitivity [SC vs GHM]' 
                    cblabel = r"$F_{\mathrm{SC}} - F_{\mathrm{GHM}}$"
            # -------------------------------
            else:
                label= 'something went wrong'

            # ---------- BG: Get additional data to plot (twin or stacked-axis) ----------------
            if quantity_list:
                add_profile_list = [] if stacked else pl_list # List for ax2-legends
                for quantity in quantity_list:
                    ax, ax2 = add_profile_to_plot(fig, ax, ACMCOM, fsize=fsize, legend_list=add_profile_list, quantity=quantity, stacked=stacked)
                if stacked and add_profile_list: ax2.legend(handles=add_profile_list, 
                                                            loc='upper right', framealpha=0.7, 
                                                            borderaxespad=0.0,                  # space to axes
                                                            borderpad=0.25, labelspacing=0.25,   # compact box)
                                                            fontsize=INFOSIZE*.8)
            # ------------------------------------------------------------------------

            title = ''
            title += label
            title += ' - Solar TOA Flux Difference'   
        
            x1 = self.latitude
            x2 = Scene2.latitude

            data1 = self.solar_eup # librad
            data2 = Scene2.solar_eup # librad2

            # ----------- Quality-Status -----------------
            # BG: modification to only include calculated results with valid quality
            # Create a combined mask: non-zero and quality 0 or 1
            quality = ACMCOM.quality_status[:]
            mask = (data1 != 0) & (data2 != 0) & np.isin(quality, [0, 1])

            # Apply the mask to data
            x1[~mask], x2[~mask], data1[~mask], data2[~mask] = np.nan, np.nan, np.nan, np.nan
            # ------------------------------------------------

            data = data1 - data2 
            if want_ratio: data = data1 / data2
            data_mean = moving_average(data1, w=w_size) / moving_average(data2, w=w_size)

            x = x1
            if want_average_line:
                p,=ax.plot(x, data_mean,
                        color='b',
                        linewidth=2, 
                        zorder=9,
                        alpha=.8,
                        label=cblabel+', averaged') 
                pl_list.append(p)

            # Add mean of the error to plot
            mean_err = np.nanmean(data)
            std  = np.nanstd(data)
            data_str = fr"⟨$\Delta F$⟩ = {mean_err:.3f} ± {std:.3f}" + r" W/m$^2$"
            print('-----------------------------------------------\n'
                  'solar: ', data_str, '    (n = ', len(data), ')\n'
                  '-----------------------------------------------')
            ax.text(
                0.99, 0.88, data_str,
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=fsize*.8, color='k',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.9))


            baseline = 0; 
            padding = 10 #1
            if want_ratio: baseline = 1; padding = 0.1
            
            ax.axhline(baseline, color='black', linestyle='--', linewidth=1.5, label=f'Baseline (y={baseline})')
            # Calculate the maximum absolute deviation from the baseline (y=1)
            max_deviation = np.max(np.abs(data - baseline)) 
            ymin = baseline - max_deviation - padding 
            ymax = baseline + max_deviation + padding

            if any(tag in additional_spesifications for tag in ("GHM", "RA", "SC")):
                ymin, ymax = -65, 65        # Ice-habit
            elif any(tag in additional_spesifications for tag in ("SmallBuffer", "FullBuffer")):
                ymin, ymax = -160, 160      # 3D-Buffer
            else:
                ymin, ymax = np.nanmin(data) - 10, np.nanmax(data) + 10

            # ax.set_ylabel(r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]", fontsize=INFOSIZE) # ('TOA upward flux diff [W/m$^2$]', fontsize=fsize)
            ylabel = r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$"

        # BG: created by me
        elif plot_type=='librad_thermal_flx_diff':  
            if 'disort_pseudospherical' in Scene2.fn and 'montecarlo' in self.fn:
                cblabel = 'MYSTIC - DISORT (ps)'
            elif 'montecarlo' in Scene2.fn and 'disort_pseudospherical' in self.fn:
                cblabel = 'DISORT (ps) - MYSTIC'
            elif 'disort' in Scene2.fn and 'montecarlo' in self.fn:
                cblabel = 'MYSTIC - DISORT'
            elif 'montecarlo' in Scene2.fn and 'disort' in self.fn:
                cblabel = 'DISORT - MYSTIC'
            
            # ------- Sensitivity analysis specs ------------------
            elif 'SmallBuffer' in self.fn:
                label = 'MYSTIC - 3D Buffer Size Sensitivity'
                cblabel = r"$F_{\mathrm{SmallBuffer}} - F_{\mathrm{LargeBuffer}}$"
            elif 'GHM' in self.fn or 'RA' in self.fn or 'SC' in self.fn:
                if 'montecarlo' in self.fn:
                    label = 'MYSTIC - '     
                else:
                    label = 'DISORT - ' 

                if 'RA' in self.fn:
                    label += 'Ice-Habit Sensitivity [RA vs GHM]' 
                    cblabel = r"$F_{\mathrm{RA}} - F_{\mathrm{GHM}}$"
                elif 'SC' in self.fn:
                    label += 'Ice-Habit Sensitivity [SC vs GHM]' 
                    cblabel = r"$F_{\mathrm{SC}} - F_{\mathrm{GHM}}$"
            elif 'FullBuffer' in self.fn:
                label = 'MYSTIC - 3D Buffer Size Sensitivity'
                cblabel = r"$F_{\mathrm{LargeBuffer}} - F_{\mathrm{SmallBuffer}}$"
            # -------------------------------
            else:
                # label= 'MYSTIC - Thermal TOA Flux Difference\n[absorbing ×0.60 - No Aerosols]\n' #'something went wrong' 
                # cblabel = r"$F_{\mathrm{abs}\times 0.60} - F_{\mathrm{no\;aer}}$"  #''   
                label = 'Something Went Wrong'
                clabel = ''

            title = ''
            title += label
            title += ' - Thermal TOA Flux Difference'   

            # ---------- BG: Get additional data to plot (twin or stacked-axis) ----------------
            if quantity_list:
                add_profile_list = [] if stacked else pl_list # List for ax2-legends
                for quantity in quantity_list:
                    ax, ax2 = add_profile_to_plot(fig, ax, ACMCOM, fsize=fsize, legend_list=add_profile_list, quantity=quantity, stacked=stacked)
                if stacked and add_profile_list: ax2.legend(handles=add_profile_list, 
                                                            loc='upper right', framealpha=0.7, 
                                                            borderaxespad=0.0,                  # space to axes
                                                            borderpad=0.25, labelspacing=0.25,   # compact box)
                                                            fontsize=INFOSIZE*.8)
            # ------------------------------------------------------------------------


            x1 = self.latitude
            x2 = Scene2.latitude

            data1 = self.thermal_eup 
            data2 = Scene2.thermal_eup 

            # ----------- Quality-Status -----------------
            # BG: modification to only include calculated results with valid quality
            # Create a combined mask: non-zero and quality 0 or 1
            quality = ACMCOM.quality_status[:]
            mask = (data1 != 0) & (data2 != 0) & np.isin(quality, [0, 1])

            # Apply the mask to data
            x1[~mask], x2[~mask], data1[~mask], data2[~mask] = np.nan, np.nan, np.nan, np.nan
            # ------------------------------------------------


            data = data1 - data2 
            if want_ratio: data = data1 / data2
            data_mean = moving_average(data1, w=w_size) / moving_average(data2, w=w_size)

            x = x1
            if want_average_line:
                p,=ax.plot(x, data_mean,
                        color='b',
                        linewidth=2, 
                        zorder=9,
                        alpha=.8,
                        label=label) 
                pl_list.append(p)
            
            # Add mean of the error to plot
            mean_err = np.nanmean(data)
            std  = np.nanstd(data)
            data_str = fr"⟨$\Delta F$⟩ = {mean_err:.3f} ± {std:.3f}" + r" W/m$^2$"
            print('-----------------------------------------------\n'
                  'thermal: ', data_str, '    (n = ', len(data), ')\n'
                  '-----------------------------------------------')
            ax.text(
                0.99, 0.88, data_str,
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=fsize*.8, color='k',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.9))


            baseline = 0; 
            padding = 10 # 1
            if want_ratio: baseline = 1; padding = 0.1
            
            ax.axhline(baseline, color='black', linestyle='--', linewidth=1.5, label=f'Baseline (y={baseline})')
            # Calculate the maximum absolute deviation from the baseline (y=1)
            max_deviation = np.max(np.abs(data - baseline)) 
            ymin = baseline - max_deviation - padding 
            ymax = baseline + max_deviation + padding

            if any(tag in additional_spesifications for tag in ("GHM", "RA", "SC")):
                ymin, ymax = -6, 6          # Ice-habit
            elif any(tag in additional_spesifications for tag in ("SmallBuffer", "FullBuffer")):
                ymin, ymax = -70, 70        # 3D-Buffer
            else:
                ymin, ymax = np.nanmin(data) - 10, np.nanmax(data) + 10
            
            # ax.set_ylabel(r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]", fontsize=INFOSIZE) # ('TOA upward flux diff [W/m$^2$]', fontsize=fsize)
            ylabel = r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$"

        # BG: created by me
        elif plot_type=='solar_flx_ratio':
            title='Solar TOA flux ratio'    
            if 'montecarlo' in Scene2.fn:
                cblabel = 'MYSTIC / BMA_FLX '
                label = 'MYSTIC / BMA_FLX, averaged'
            elif 'disort' in Scene2.fn:
                cblabel = 'DISORT / BMA_FLX'
                label = 'DISORT / BMA_FLX, averaged'
            else:
                print("BG: error, correct files?")
            title += '     [ ' + cblabel + ' ] '


            x1 = self.latitude
            x2 = Scene2.latitude
    
            data1 = self.solar_combined_top_of_atmosphere_flux
            data2 = Scene2.solar_eup

            # Performe interpolation for comparison
            fill_value_data1 = 9.96921e+36
            fill_value_data2 = 0
            x, data1_interp, data2 = interpolate(x1, data1, fill_value_data1, x2, data2, fill_value_data2) 

            # Calculate ratio
            data = data2 / data1_interp

            data_mean = moving_average(data2, w=w_size) / moving_average(data1_interp, w=w_size)
            p,=ax.plot(x, data_mean,
                       color='b',
                       linewidth=2, 
                       zorder=9,
                       alpha=.8,
                       label=label) 
            pl_list.append(p)

           
            baseline = 1
            padding = 0.1
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (y=1)')
            # Calculate the maximum absolute deviation from the baseline (y=1)
            max_deviation = np.max(np.abs(data - baseline)) 
            # BG: adjust ymin/ymax for equal comparison
            max_deviation = 1.3

            ymin = baseline - max_deviation - padding 
            ymax = baseline + max_deviation + padding



            ax.set_ylabel('TOA upward flux ratio', fontsize=fsize)

        # BG: created by me
        elif plot_type=='thermal_flx_ratio':
            title='Thermal TOA flux ratio'    
            if 'montecarlo' in Scene2.fn:
                cblabel = 'MYSTIC / BMA_FLX '
                label = 'MYSTIC / BMA_FLX, averaged'
            elif 'disort' in Scene2.fn:
                cblabel = 'DISORT / BMA_FLX'
                label = 'DISORT / BMA_FLX, averaged'
            else:
                print("BG: error, correct files?")
            title += '     [ ' + cblabel + ' ] '


            x1 = self.latitude
            x2 = Scene2.latitude
    
            data1 = self.thermal_combined_top_of_atmosphere_flux
            data2 = Scene2.thermal_eup

            # Performe interpolation for comparison
            fill_value_data1 = 9.96921e+36
            fill_value_data2 = 0
            x, data1_interp, data2 = interpolate(x1, data1, fill_value_data1, x2, data2, fill_value_data2) 

            # Calculate ratio
            data = data2 / data1_interp

            data_mean = moving_average(data2, w=6) / moving_average(data1_interp, w=6)
            p,=ax.plot(x, data_mean,
                       color='b',
                       linewidth=2, 
                       zorder=9,
                       alpha=.8,
                       label=label) 
            pl_list.append(p)

           
            baseline = 1
            padding = 0.1
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (y=1)')
            # Calculate the maximum absolute deviation from the baseline (y=1)
            max_deviation = np.max(np.abs(data - baseline)) 
            # BG: adjust ymin/ymax for equal comparison
            max_deviation = 1.3
            
            ymin = baseline - max_deviation - padding 
            ymax = baseline + max_deviation + padding

            ax.set_ylabel('TOA upward flux ratio', fontsize=INFOSIZE)

        # BG: created by me
        elif plot_type=='solar_flx_diff':
            title='Solar TOA flux difference'    
            if 'montecarlo' in Scene2.fn:
                cblabel = 'MYSTIC - BMA_FLX '
                label = 'MYSTIC - BMA_FLX, averaged'
            elif 'disort_pseudospherical' in Scene2.fn:
                cblabel = 'DISORT (ps) - BMA_FLX'
                label = 'DISORT (ps) - BMA_FLX, averaged'
            elif 'disort' in Scene2.fn:
                cblabel = 'DISORT - BMA_FLX'
                label = 'DISORT - BMA_FLX, averaged'
            else:
                print("BG: error, correct files?")
            title += '     [ ' + cblabel + ' ] '


            x1 = self.latitude
            x2 = Scene2.latitude
    
            data1 = self.solar_combined_top_of_atmosphere_flux
            data2 = Scene2.solar_eup

            # ----------- Quality-Status -----------------
            quality = self.quality_status[:]
            # boolean mask: True where quality is 0 or 2
            quality_mask = np.isin(quality, [0, 2]) 

            data1 = data1[quality_mask]
            x1 = x1[quality_mask]

            # ----------- Quality-Status -----------------
            quality = ACMCOM.quality_status[:]
            quality_mask = np.isin(quality, [0, 1])

            x2 = x2[quality_mask]
            data2 = data2[quality_mask]
            # ------------------------------------------------

            # Performe interpolation for comparison
            fill_value_data1 = 9.96921e+36
            fill_value_data2 = 0
            x, data1_interp, data2 = interpolate(x1, data1, fill_value_data1, x2, data2, fill_value_data2) 

            # Calculate ratio
            data = (data2 - data1_interp) #/ data1_interp

            if want_average_line:
                data_mean = (moving_average(data2, w=w_size) - moving_average(data1_interp, w=w_size)) / moving_average(data1_interp, w=w_size)
                p,=ax.plot(x, data_mean,
                        color='b',
                        linewidth=2, 
                        zorder=9,
                        alpha=.8,
                        label=label) 
                pl_list.append(p)


            # Add mean of the relative‑error to plot
            mean_err = np.nanmean(data)
            std  = np.nanstd(data)
            data_str = fr"⟨$\Delta F$⟩ = {mean_err:.3f} ± {std:.3f}" + r" W/m$^2$"
            print('-----------------------------------------------\n'
                  'solar: ', data_str, '    (n = ', len(data), ')\n'
                  '-----------------------------------------------')
            ax.text(.98, .10, data_str,
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fsize*.8, color='k',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.8))


           
            baseline = 0
            padding = 0.1
            ax.axhline(0.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (y=0)')
            # Calculate the maximum absolute deviation from the baseline (y=1)
            max_deviation = np.max(np.abs(data - baseline)) 
      
            
            ymin = baseline - max_deviation - padding 
            ymax = baseline + max_deviation + padding

            # ax.set_ylabel(r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]", fontsize=INFOSIZE) #('TOA upward flux diff [W/m$^2$]', fontsize=fsize)
            ylabel = r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$"


        # BG: created by me
        elif plot_type=='thermal_flx_diff':
            title='Thermal TOA flux difference'    
            if 'montecarlo' in Scene2.fn:
                cblabel = 'MYSTIC - BMA_FLX '
                label = 'MYSTIC - BMA_FLX, averaged'
            elif 'disort_pseudospherical' in Scene2.fn:
                cblabel = 'DISORT (ps) - BMA_FLX'
                label = 'DISORT (ps) - BMA_FLX, averaged'
            elif 'disort' in Scene2.fn:
                cblabel = 'DISORT - BMA_FLX'
                label = 'DISORT - BMA_FLX, averaged'
            else:
                print("BG: error, correct files?")
            title += '     [ ' + cblabel + ' ] '


            x1 = self.latitude
            x2 = Scene2.latitude
    
            data1 = self.thermal_combined_top_of_atmosphere_flux
            data2 = Scene2.thermal_eup

            # ----------- Quality-Status -----------------
            quality = self.quality_status[:]
            # boolean mask: True where quality is 0 or 1
            quality_mask = np.isin(quality, [0, 1]) 

            data1 = data1[quality_mask]
            x1 = x1[quality_mask]

            # ----------- Quality-Status -----------------
            quality = ACMCOM.quality_status[:]
            quality_mask = np.isin(quality, [0, 1])

            x2 = x2[quality_mask]
            data2 = data2[quality_mask]
            # ------------------------------------------------

            # Performe interpolation for comparison
            fill_value_data1 = 9.96921e+36
            fill_value_data2 = 0
            x, data1_interp, data2 = interpolate(x1, data1, fill_value_data1, x2, data2, fill_value_data2) 

            # Calculate ratio
            data = (data2 - data1_interp) #/ data1_interp

            if want_average_line:
                data_mean = (moving_average(data2, w=w_size) - moving_average(data1_interp, w=w_size)) / moving_average(data1_interp, w=w_size)
                p,=ax.plot(x, data_mean,
                        color='b',
                        linewidth=2, 
                        zorder=9,
                        alpha=.8,
                        label=label) 
                pl_list.append(p)


            # Add mean of the relative‑error to plot
            mean_err = np.nanmean(data)
            std  = np.nanstd(data)
            data_str = fr"⟨$\Delta F$⟩ = {mean_err:.3f} ± {std:.3f}" + r" W/m$^2$"
            print('-----------------------------------------------\n'
                  'thermal: ', data_str, '    (n = ', len(data), ')\n'
                  '-----------------------------------------------')
            ax.text(.98, .10, data_str,
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fsize*.8, color='k',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.8))

           
            baseline = 0
            padding = 0.1
            ax.axhline(0.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (y=0)')
            # Calculate the maximum absolute deviation from the baseline (y=1)
            max_deviation = np.max(np.abs(data - baseline)) 
        
            ymin = baseline - max_deviation - padding 
            ymax = baseline + max_deviation + padding

            # ax.set_ylabel(r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]", fontsize=INFOSIZE) #('TOA upward flux diff [W/m$^2$]', fontsize=fsize)
            ylabel = r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$"

        #BG: created by me
        elif plot_type=='solar_flx_diff_histogram':
            title='Solar TOA flux difference'    
            if 'montecarlo' in Scene2.fn:
                cblabel = 'MYSTIC - BMA_FLX '
                # label = 'MYSTIC - BMA_FLX, averaged'
            elif 'disort_pseudospherical' in Scene2.fn:
                cblabel = 'DISORT (ps) - BMA_FLX'
                # label = 'DISORT (ps) - BMA_FLX, averaged'
            elif 'disort' in Scene2.fn:
                cblabel = 'DISORT - BMA_FLX'
                # label = 'DISORT - BMA_FLX, averaged'
            else:
                print("BG: error, correct files?")
            title += '     [ ' + cblabel + ' ] '


            x1 = self.latitude
            x2 = Scene2.latitude
    
            data1 = self.solar_combined_top_of_atmosphere_flux
            data2 = Scene2.solar_eup

            # ----------- Quality-Status -----------------
            quality = self.quality_status[:]
            # boolean mask: True where quality is 0 or 2
            quality_mask = np.isin(quality, [0, 2]) 

            data1 = data1[quality_mask]
            x1 = x1[quality_mask]

            # ----------- Quality-Status -----------------
            quality = ACMCOM.quality_status[:]
            quality_mask = np.isin(quality, [0, 1])

            x2 = x2[quality_mask]
            data2 = data2[quality_mask]
            # ------------------------------------------------

            # Performe interpolation for comparison
            fill_value_data1 = 9.96921e+36
            fill_value_data2 = 0
            x, data1_interp, data2 = interpolate(x1, data1, fill_value_data1, x2, data2, fill_value_data2) 

            # Calculate ratio
            data = (data2 - data1_interp) #/ data1_interp

            # Add mean of the relative‑error to plot
            mean_err = np.nanmean(data)
            std  = np.nanstd(data)
            data_str = fr"⟨$\Delta F$⟩ = {mean_err:.3f} ± {std:.3f}" + r" W/m$^2$"
            print('solar: ', data_str, '    (n = ', len(data), ')')
            ax.text(.98, .10, data_str,
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fsize*.8, color='k',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.8))


            # vertical zero‐error line:
            ax.axvline(
                0.0,
                color='red',
                linestyle='--',
                linewidth=1.5,
                label='Zero error'
            )

            # labels, title, legend
            xlabel = r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$"; xlabel_specs = r" [W/m$^2$]"
            ylabel = 'Count'; ylabel_specs = " []"
          

        #BG: created by me
        elif plot_type=='thermal_flx_diff_histogram':
            title='Thermal TOA flux difference'    
            if 'montecarlo' in Scene2.fn:
                cblabel = 'MYSTIC - BMA_FLX '
                # label = 'MYSTIC - BMA_FLX, averaged'
            elif 'disort_pseudospherical' in Scene2.fn:
                cblabel = 'DISORT (ps) - BMA_FLX'
                # label = 'DISORT (ps) - BMA_FLX, averaged'
            elif 'disort' in Scene2.fn:
                cblabel = 'DISORT - BMA_FLX'
                # label = 'DISORT - BMA_FLX, averaged'
            else:
                print("BG: error, correct files?")
            title += '     [ ' + cblabel + ' ] '


            x1 = self.latitude
            x2 = Scene2.latitude
    
            data1 = self.thermal_combined_top_of_atmosphere_flux
            data2 = Scene2.thermal_eup

            # ----------- Quality-Status -----------------
            quality = self.quality_status[:]
            # boolean mask: True where quality is 0 or 1
            quality_mask = np.isin(quality, [0, 1]) 

            data1 = data1[quality_mask]
            x1 = x1[quality_mask]

            # ----------- Quality-Status -----------------
            quality = ACMCOM.quality_status[:]
            quality_mask = np.isin(quality, [0, 1])

            x2 = x2[quality_mask]
            data2 = data2[quality_mask]
            # ------------------------------------------------


            # Performe interpolation for comparison
            fill_value_data1 = 9.96921e+36
            fill_value_data2 = 0
            x, data1_interp, data2 = interpolate(x1, data1, fill_value_data1, x2, data2, fill_value_data2) 

            # Calculate ratio
            data = (data2 - data1_interp) #/ data1_interp
            
            # Add mean of the relative‑error to plot
            mean_err = np.nanmean(data)
            std  = np.nanstd(data)
            data_str = fr"⟨$\Delta F$⟩ = {mean_err:.3f} ± {std:.3f}" + r" W/m$^2$"
            print('thermal: ', data_str, '    (n = ', len(data), ')')
            ax.text(.98, .10, data_str,
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fsize*.8, color='k',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.8))

            # vertical zero‐error line:
            ax.axvline(
                0.0,
                color='red',
                linestyle='--',
                linewidth=1.5,
                label='Zero error'
            )
          

            # labels, title, legend
            xlabel = r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$"; xlabel_specs = r" [W/m$^2$]"
            ylabel = 'Count'; ylabel_specs = " []"
           
        
        elif plot_type == 'solar_flx_diff_correlation':
            xlabel_specs = r' [W/m$^2$]'

            xlabel = 'BMA-FLX'
            data1 = self.solar_combined_top_of_atmosphere_flux # BMA-FLX
            x1 = self.latitude 
            # ----------- Quality-Status -----------------
            quality = self.quality_status[:]
            # boolean mask: True where quality is 0 or 2
            quality_mask = np.isin(quality, [0, 2]) 

            data1 = data1[quality_mask]
            x1 = x1[quality_mask]
            # ------------------------------------------------

            if 'montecarlo' in Scene2.fn:   ylabel = 'MYSTIC'
            else:                           ylabel = 'DISORT' 
            data2 = Scene2.solar_eup # libRad
            x2 = Scene2.latitude
            # ----------- Quality-Status -----------------
            # BG: modification to only include calculated results with valid quality
            # Create a combined mask: non-zero and quality 0 or 1
            quality = ACMCOM.quality_status[:]
            mask = (data2 != 0) & np.isin(quality, [0, 1])

            data2 = data2[mask]
            x2 = x2[mask]
            # ------------------------------------------------¨
            
            title += ylabel
            title += ' - Solar TOA Flux'   
        
            # ax.set_aspect("equal", adjustable="box")

            # Performe interpolation for comparison
            fill_value_data1 = 9.96921e+36
            fill_value_data2 = 0
            not_used, data1_interp, data2 = interpolate(x1, data1, fill_value_data1, x2, data2, fill_value_data2) 

            x = data1_interp # model truth
            y = data2 
            # Stats
            r = np.corrcoef(x, y)[0, 1] # Correlation coefficient
            r2 = r**2 # The strength of linear association
            # Least squares fit y = a + b x
            b, a = np.polyfit(x, y, 1)
                # yhat = a + b * x
            # Errors
            diff = y - x
                # rmse = np.sqrt(np.mean(diff**2))
            mae = np.mean(np.abs(diff))
            bias = np.mean(diff)

            # Ranges for plotting
            xy_min = np.nanmin([x.min(), y.min()])
            xy_max = np.nanmax([x.max(), y.max()])
            pad = 0.03 * (xy_max - xy_min if xy_max > xy_min else 1.0)
            lo, hi = xy_min - pad, xy_max + pad

            # For setting ax.set_ylim/xlim
            ymin, xmin, ymax, xmax, pad = xy_min, xy_min, xy_max, xy_max, 0


            # y=x line
            p, = ax.plot([lo, hi], [lo, hi], lw=1.2, c='b', linestyle="--", label="1:1")
            pl_list.append(p)

            # Regression line over same span
            xx = np.linspace(lo, hi, 100)
            p, = ax.plot(xx, a + b * xx, lw=1.5, c='r', label=f"Linear Fit\n(least squares)") #: y = {a:.2f} + {b:.2f}x")
            pl_list.append(p)


            data_str = (
                # f"n = {x.size}\n"
                f"r² = {r2:.2f}\n"
                # f"RMSE = {rmse:.2f}\n"
                r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ (Bias) = " + f"{bias:.2f} " + r"W/m$^2$"
                # f"\nMAE = {mae:.2f}\n"
            )
            ax.text(.98, .02, data_str,
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fsize*.7, color='k',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.8))

        elif plot_type == 'thermal_flx_diff_correlation':
            return

        elif plot_type == 'solar_earthCARE_flx_rel_err':
            title='Solar TOA flux relative error     [  ACM_RT 1D/3D vs BMA_FLX  ] '    
            cblabel = 'ACM_RT 1D vs BMA_FLX'
            label = 'ACM_RT 3D vs BMA_FLX'
            itoa=0

            x1d_and_3d = Scene3.latitude_active
            flx1d = Scene3.flux_up_solar_1d_all_sky[:,itoa]
            flx3d = Scene3.flux_up_solar_3d_all_sky[:,itoa]

            xBMA = self.latitude 
            flxBMA = self.solar_combined_top_of_atmosphere_flux

            # ----------- Quality-Status -----------------
            quality = self.quality_status[:]
            # boolean mask: True where quality is 0 or 2
            quality_mask = np.isin(quality, [0, 2]) 

            flxBMA = flxBMA[quality_mask]
            xBMA = xBMA[quality_mask]

            # ----------- Quality-Status -----------------
            quality = Scene3.quality_status[:]
            quality_mask = np.isin(quality, [0, 1])

            x1d_and_3d = x1d_and_3d[quality_mask]
            flx1d = flx1d[quality_mask]
            flx3d = flx3d[quality_mask]
            # ------------------------------------------------

            # Performe interpolation for comparison
            fill_value_BMA = 9.96921e+36
            fill_value_1d  = 9.96921e+36
            fill_value_3d  = 9.96921e+36
            x, flxBMA_interp1d, flx1d = interpolate(xBMA, flxBMA, fill_value_BMA, x1d_and_3d, flx1d, fill_value_1d) 
            x2, flxBMA_interp3d, flx3d = interpolate(xBMA, flxBMA, fill_value_BMA, x1d_and_3d, flx3d, fill_value_3d) 
            if len(x2) == 0: label += ', No data'

            data1d = (flx1d - flxBMA_interp1d) #/ flxBMA_interp1d 
            data3d = (flx3d - flxBMA_interp3d) #/ flxBMA_interp3d 

            data = data1d
        

            p,=ax.plot(x2, data3d, 
                        color='b',
                        linewidth=2,
                        zorder=9,
                        label=label)
            pl_list.append(p)


            # Add mean of the relative‑error to plot
            mean_err1d = np.nanmean(data)
            mean_err3d = np.nanmean(data3d)
            std_err1d  = np.nanstd(data)
            std_err3d  = np.nanstd(data3d)
            data_str = fr"ACM_RT 1D vs BMA_FLX: ⟨$\Delta F$⟩ = {mean_err1d:.3f} ± {std_err1d:.3f}"+ r" W/m$^2$"
            if len(x2) != 0: data_str += fr"\nACM_RT 3D vs BMA_FLX: ⟨$\Delta F$⟩ = {mean_err3d:.3f} ± {std_err3d:.3f}"+ r" W/m$^2$"

            ax.text(.98, .10, data_str,
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fsize*.8, color='k',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.8))

           
            baseline = 0
            padding = 0.1
            ax.axhline(0.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (y=0)')
            # Calculate the maximum absolute deviation from the baseline (y=1)
            max_deviation = np.max(np.abs(data - baseline)) 
            # BG: adjust ymin/ymax for equal comparison
            max_deviation = 1.3
            
            ymin = baseline - max_deviation - padding 
            ymax = baseline + max_deviation + padding

            ax.set_ylabel(r"$F_{\mathrm{TOA}}^{\uparrow}$ - ratio []", fontsize=INFOSIZE) #('TOA upward flux ratio', fontsize=fsize)

            print(f'\n\n ------------------- ACM_RT 1D vs BMA_FLX (solar):   <rel-err> = {mean_err1d} ± {std_err1d},     n_points = {len(data1d)} ------------------')
            print(f'\n\n ------------------- ACM_RT 3D vs BMA_FLX (solar):   <rel-err> = {mean_err3d} ± {std_err3d},     n_points = {len(data3d)} ------------------')
            print(f'\n\n')



        elif plot_type == 'thermal_earthCARE_flx_rel_err':
            title='Thermal TOA flux relative error     [  ACM_RT 1D/3D vs BMA_FLX  ] '   
            cblabel = 'ACM_RT 1D vs BMA_FLX'
            label = 'ACM_RT 3D vs BMA_FLX'
            itoa=0

            x1d_and_3d = Scene3.latitude_active
            flx1d = Scene3.flux_up_thermal_1d_all_sky[:,itoa]
            flx3d = Scene3.flux_up_thermal_3d_reference_height_all_sky[:]

            xBMA = self.latitude 
            flxBMA = self.thermal_combined_top_of_atmosphere_flux

            # ----------- Quality-Status -----------------
            quality = self.quality_status[:]
            # boolean mask: True where quality is 0 or 1
            quality_mask = np.isin(quality, [0, 1]) 

            flxBMA = flxBMA[quality_mask]
            xBMA = xBMA[quality_mask]

            # ----------- Quality-Status -----------------
            quality = Scene3.quality_status[:]
            quality_mask = np.isin(quality, [0, 1])

            x1d_and_3d = x1d_and_3d[quality_mask]
            flx1d = flx1d[quality_mask]
            flx3d = flx3d[quality_mask]
            # ------------------------------------------------

            # Performe interpolation for comparison
            fill_value_BMA = 9.96921e+36
            fill_value_1d  = 9.96921e+36
            fill_value_3d  = 9.96921e+36
            x, flxBMA_interp1d, flx1d = interpolate(xBMA, flxBMA, fill_value_BMA, x1d_and_3d, flx1d, fill_value_1d) 
            x2, flxBMA_interp3d, flx3d = interpolate(xBMA, flxBMA, fill_value_BMA, x1d_and_3d, flx3d, fill_value_3d) 
            if len(x2) == 0: label += ', No data'

            data1d = (flx1d - flxBMA_interp1d) / flxBMA_interp1d 
            data3d = (flx3d - flxBMA_interp3d) / flxBMA_interp3d 

            data = data1d


            p,=ax.plot(x2, data3d, 
                        color='b',
                        linewidth=2,
                        zorder=9,
                        label=label)
            pl_list.append(p)


            # Add mean of the relative‑error to plot
            mean_err1d = np.nanmean(data)
            mean_err3d = np.nanmean(data3d)
            std_err1d  = np.nanstd(data)
            std_err3d  = np.nanstd(data3d)
            data_str = fr"ACM_RT 1D vs BMA_FLX: ⟨$\Delta F$⟩= {mean_err1d:.3f} ± {std_err1d:.3f}" + r" W/m$^2$"
            if len(x2) != 0: data_str += fr"\nACM_RT 3D vs BMA_FLX: ⟨$\Delta F$⟩ = {mean_err3d:.3f} ± {std_err3d:.3f}" + r" W/m$^2$"

            ax.text(.98, .10, data_str,
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fsize*.8, color='k',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.8))

            baseline = 0
            padding = 0.1
            ax.axhline(0.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (y=0)')
            # Calculate the maximum absolute deviation from the baseline (y=1)
            max_deviation = np.max(np.abs(data - baseline)) 
            # BG: adjust ymin/ymax for equal comparison
            max_deviation = 1.3
            
            ymin = baseline - max_deviation - padding 
            ymax = baseline + max_deviation + padding

            ax.set_ylabel(r"$F_{\mathrm{TOA}}^{\uparrow}$ - ratio []", fontsize=INFOSIZE) #('TOA upward flux ratio', fontsize=fsize)

            print(f'\n\n ------------------- ACM_RT 1D vs BMA_FLX (thermal):   <rel-err> = {mean_err1d} ± {std_err1d},     n_points = {len(data1d)} ------------------')
            print(f'\n\n ------------------- ACM_RT 3D vs BMA_FLX (thermal):   <rel-err> = {mean_err3d} ± {std_err3d},     n_points = {len(data3d)} ------------------')
            print(f'\n\n')



        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        elif plot_type=='librad_solar_flx':
            title='Solar and thermal TOA flux'    
            #            cblabel='albedo_direct_radiation_surface_visible'
            #            cblabel='twostr: solar_top_of_atmosphere_flux'
            cblabel='mystic: solar_top_of_atmosphere_flux'
            x = self.latitude
            data1 = self.solar_eup
            data=  moving_average(data1, w=21)  # Assessment_domain_along_size = 21
            data = self.solar_eup
 
            #           p,=ax.plot(x, data1, color='red', label='twostr: solar_top_of_atmosphere_flux, averaged')
            #           pl_list.append(p)
            if Scene2!=None:
                x2 = Scene2.latitude
                data2 = Scene2.solar_eup
                data2a=  moving_average(data2, w=21)  # Assessment_domain_along_size = 21
                p,=ax.plot(x2, data2, color='pink', label='disort: solar_top_of_atmosphere_flux')
                pl_list.append(p)
            #            p,=ax.plot(x2, data2a, color='black', label='disort: solar_top_of_atmosphere_flux, averaged')
            #            pl_list.append(p)

            x2 = self.latitude
            data2 = self.thermal_eup
            #                data2a=  moving_average(data2, w=21)  # Assessment_domain_along_size = 21
            p,=ax.plot(x2, data2, color='green', label='mystic: thermal_top_of_atmosphere_flux')
            pl_list.append(p)
            
            
            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =0
            if self.Name=='Baja':
                ymax = 750
            elif self.Name=='Halifax':
                ymax = 500
            elif self.Name=='Arctic_05378D':
                ymax = 1000
            else:
                ymax = 500
                
            ax.set_ylabel('TOA upward flux (W/m$^2$)', fontsize=fsize)
        elif plot_type=='thermal_combined_top_of_atmosphere_flux':
            data = self.thermal_combined_top_of_atmosphere_flux
            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =0
            ymax =500
            units=self.thermal_combined_top_of_atmosphere_flux_units
        elif plot_type=='thermal_combined_top_of_atmosphere_flux_quality_status':
            data = self.thermal_combined_top_of_atmosphere_flux_quality_status
            vmin=data.min()#0.0001 #
            vmax=data.max()#4 #
            ymin =0
            ymax =3
            
        else:
            print("PlotLine: unknown plot_type: "+plot_type+", exiting")
            exit()

        
        if 'histogram' not in plot_type:
            # Set y-limits
            ax.set_ylim(ymin,ymax)
            # Set x-limits
            if 'correlation' not in plot_type:
                if 'solar_both' in plot_type or 'thermal_both' in plot_type:
                    xmax, xmin = np.nanmax(x2), np.nanmin(x2)
                else:
                    xmax, xmin = np.nanmax(x), np.nanmin(x)
                pad = 0.05 * (xmax-xmin)
            ax.set_xlim(xmin - pad, xmax + pad)

    
                

        # Info-Prints:
        # print(x.shape, data.shape, self.latitude.shape)
        # print('xmin, xmax  datamin, datamax', plot_type, x.min(), x.max(), data.min(), data.max())
        if units != '':
            cblabel = cblabel + ' ('+units+')'

        if 'gabba' in plot_type:
            # ax.plot(x2, data2)
            data3=moving_average(data2, w=20)
            p,=ax.plot(x2, data3, color='black')
            
        if plot_type=='solar_zenith_angle' or plot_type=='solar_azimuth_angle':
            tmplist = []
            for ib in np.arange(self.bbr_directions):
                label=''.join(str(self.bbr_direction[ib,:]))
                label = label.replace('b',''); label = label.replace('\'',''); 
                l, = ax.plot(x, data[:, ib], label=label) # cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0)
                tmplist.append(l)
            ax.legend(handles=tmplist)
        elif plot_type=='plot_info':
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.set_ylabel('')
            # ax.legend(handles=pl_list)
        elif 'histogram' in plot_type:
            # Plot the histogram
            n_bins = 50
            ax.hist(
                data,
                bins=n_bins,
                density=False,     # set True if you’d rather show a normalized PDF
                alpha=0.7,
                edgecolor='black',
                label=cblabel
            )
            ax.legend(  loc='upper left', framealpha=0.7, 
                        borderaxespad=0.0,                  # space to axes
                        borderpad=0.25, labelspacing=0.25,   # compact box)
                        fontsize=INFOSIZE*.8)
        elif 'correlation' in plot_type: 
            # ax.scatter(x, y, s=8, alpha=.8, cmap=cmap)
            res = y - x
            sc = ax.scatter(x, y, s=8, alpha=.8, c=res, cmap=cmap)

            # Colorbar
            cb = fig.colorbar(
                sc, ax=ax, 
                shrink=.8, pad=0.0, 
                extend='both',
                extendrect=True)
            cb.set_label(rf"$\Delta F$ {ylabel} - {xlabel} [W/m$^2$]", size=INFOSIZE)
            cb.ax.tick_params(labelsize=INFOSIZE)

            # soften colorbar box + background
            cb.outline.set_visible(False)             # remove black frame
            cb.ax.set_facecolor('#f7f7f7')            # subtle bg behind ramp

            ax.legend(handles=pl_list,
                        loc='upper left', framealpha=0.7, 
                        borderaxespad=0.0,                   # space to axes
                        borderpad=0.25, labelspacing=0.25,   # compact box)
                        fontsize=INFOSIZE*.8)
        elif 'flx_diff' in plot_type:
            l, = ax.plot(x, data, 
                                label=cblabel, 
                                color='r',
                                marker=".", linestyle="None", markersize = 4
                                ) # cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0)
            ax.legend(handles=[l]+pl_list,
                        loc='upper left', framealpha=0.7, 
                        borderaxespad=0.0,                  # space to axes
                        borderpad=0.25, labelspacing=0.25,   # compact box)
                        fontsize=INFOSIZE*.8)
            ax.vlines(x, baseline, data, color='red', alpha=0.6, lw=0.2)  # lines to baseline
        else:
            #### ONLY FOR ATMOSPHERE - ANALYSIS ######
            ax.set_xlim(3.01, 4.99)
            ########################################

            l, = ax.plot(x, data, 
                                label=cblabel, 
                                color='r',
                                # marker=".", markersize = 4, linestyle="None"
                                ) # cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0)
            # ax.legend(handles=[l]+pl_list, fontsize=INFOSIZE*.7)
            ax.legend(handles=[l]+pl_list,
                        # loc='upper left', 
                        framealpha=0.7, 
                        borderaxespad=0.0,                  # space to axes
                        borderpad=0.25, labelspacing=0.25,   # compact box)
                        fontsize=INFOSIZE*.8)


        ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
        ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)

        

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
        ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
        fig.suptitle(title, fontsize=FONTSIZE, y=0.98, fontweight='bold')
        ax.set_title(f"{self.Name} - {date}  {time} (UTC)", fontsize=INFOSIZE)
        plt.figtext(0.001, 0.003, f"Baselines: AC = ({AC_baseline})  |  BA = ({BA_baseline})", fontsize=FONTSIZE*0.45)
        # Orbit nr: self.Name

        # BG: ----- plot-adjustments for nicer looking plots -----------
        fig.tight_layout()
        # Axes background (warm light grey)
        ax.set_facecolor('#f0f0f0')
        # Grid: major dashed, minor dotted
        ax.grid(which='major', linestyle='--', alpha=0.4)
        ax.grid(which='minor', linestyle=':',  alpha=0.2)
        ax.minorticks_on()

        

        # remove top/right border
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        # -------------------------------------------------------------

        if pngfile==None or pngfile=='':
            plt.show()
        else:
            if verbose:
                print("pngfile", pngfile)
            plt.savefig(pngfile)
            plt.close()

        return


    def InterpolateAngstrom(self, ACMCOM, angstrom_exponent=1):
        AMACD_lat = self.latitude
        AMACD_lon = self.longitude
        #
        # MSI provides angstrom_exponent for different wavelength
        # intervals. For the moment choose the longest wavelength one as
        # for the synthetic data the other one appears unrealistic.
        #
        AMACD_ang = self.aerosol_angstrom_exponent[:,:,angstrom_exponent] #lambda [670,865]
        ACMCOM_lat = np.transpose(ACMCOM.latitude)
        ACMCOM_lon = np.transpose(ACMCOM.longitude)

        # Geometry information is on tie grid which is similar to data grid
        # in along-track direction (and slightly not-aligned),
        # but with fewer points in the across-track direction.
        # Interpolate to full grid
        points = (AMACD_lon.flatten(), AMACD_lat.flatten())
        values = AMACD_ang.flatten()
        xi = (ACMCOM_lon.flatten(), ACMCOM_lat.flatten())

        AMACD_ang = griddata(points, values, xi, method='nearest')
        AMACD_ang = np.reshape(AMACD_ang, ACMCOM_lat.shape)
        self.aerosol_angstrom_exponent = AMACD_ang
        return

    def ReadEarthCAREh5(self, fn, ACM3D=None, verbose=False, Resolution='', RemoveMissingData=False):

        atmosphere=1 ################ based on ACM-CAP -> GOOD
        # atmosphere=0 ################ Composit profile -> BAD
     
        if verbose:
            print("Reading EarthCARE L2 file:", fn)

        file    = h5py.File(fn, 'r')
        self.fn= fn
        
        if 'libRad' in fn:
            SD=file
        elif Resolution != '':
            SD=file['ScienceData'][Resolution]
        else:
            SD=file['ScienceData']

        if verbose:
            for key in SD.keys():
                print(key)
            #            for item in SD.items():
            #                print(item)


        #if 'ACM_3D_' in fn:
        if 'ALL_3D_' in fn:
            #            print(SD['index_construction'].attrs.keys())
            #            print(SD['index_construction'].attrs['missing_value'])
            SDSpecificProductHeader = file['HeaderData']['VariableProductHeader']['SpecificProductHeader']
            #print('Tove2')
            self.nadir_pixel_index = SDSpecificProductHeader['nadir_pixel_index'][()]
            #print('Tove3')
            self.number_pixels_along_track_assessment_domain = SDSpecificProductHeader['number_pixels_along_track_assessment_domain'][()]
            self.number_pixels_across_track_assessment_domain = SDSpecificProductHeader['number_pixels_across_track_assessment_domain'][()]
            self.index_construction=SD['index_construction'][()]
            self.along_track_shape=self.index_construction.shape[1]
            self.across_track_shape=self.index_construction.shape[0]
            #            self.index_construction_quality_status=SD['index_construction_quality_status'][()]
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            self.missing_value = SD['index_construction'].attrs['missing_value']
            self.number_pixels_along_track_buffer_zone_back_view = SD['number_pixels_along_track_buffer_zone_back_view'][()]
            self.number_pixels_along_track_buffer_zone_fore_view = SD['number_pixels_along_track_buffer_zone_fore_view'][()]
            self.number_pixels_across_track_buffer_zone = SD['number_pixels_across_track_buffer_zone'][()]
            self.start_along_track_assessment_domain_day_3d=SD['start_along_track_assessment_domain_day_3d'][()]
            self.number_pixels_along_track_buffer_zone_back_view=SD['number_pixels_along_track_buffer_zone_back_view'][()]
            self.number_pixels_along_track_buffer_zone_fore_view=SD['number_pixels_along_track_buffer_zone_fore_view'][()]
            self.number_pixels_across_track_buffer_zone=SD['number_pixels_across_track_buffer_zone'][()]
            
            # Remove all indices equal 0. Does not appear to have any valid data
            if RemoveMissingData:
                self.indx = np.where(self.index_construction >0 )#> self.missing_value)
                self.along_track_shape =  int(len(self.indx[0])/self.across_track_shape)
                self.index_construction= self.index_construction[self.indx]
                self.index_construction = np.reshape(self.index_construction,(self.across_track_shape,self.along_track_shape))
                self.latitude = np.reshape(self.latitude[self.indx],(self.across_track_shape,self.along_track_shape))
                self.longitude = np.reshape(self.longitude[self.indx],(self.across_track_shape,self.along_track_shape))

        elif 'AM__ACD' in fn:
            SDSpecificProductHeader = file['HeaderData']['VariableProductHeader']['SpecificProductHeader']
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            self.aerosol_angstrom_exponent=SD['aerosol_angstrom_exponent'][()]
            self.aerosol_angstrom_exponent = self.aerosol_angstrom_exponent[:,:,:]

        elif 'ACM_RT' in fn:
            SDSpecificProductHeader = file['HeaderData']['VariableProductHeader']['SpecificProductHeader']
            #           self. = SDSpecificProductHeader[''][()]
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            self.latitude_active=SD['latitude_active'][()]
            self.longitude_active=SD['longitude_active'][()]
            self.height_layers=SD['height_layers'][()]
            self.height_levels=SD['height_levels'][()]
            self.flux_up_solar_1d_all_sky=SD['flux_up_solar_1d_all_sky'][()]
            self.flux_up_solar_3d_all_sky=SD['flux_up_solar_3d_all_sky'][()]
            
            self.flux_up_thermal_1d_all_sky=SD['flux_up_thermal_1d_all_sky'][()]
            self.flux_up_thermal_3d_reference_height_all_sky=SD['flux_up_thermal_3d_reference_height_all_sky'][()]

            #tms self.flux_up_solar_1d_all_sky=self.flux_up_solar_1d_all_sky[atmosphere,:,:]
            #tms self.flux_up_solar_3d_all_sky=self.flux_up_solar_3d_all_sky[atmosphere,:,:]
            self.flux_up_solar_1d_all_sky=self.flux_up_solar_1d_all_sky[0,:,:]
            self.flux_up_solar_3d_all_sky=self.flux_up_solar_3d_all_sky[0,:,:]

            # ESO
            self.flux_up_thermal_1d_all_sky=self.flux_up_thermal_1d_all_sky[0,:,:]
            self.flux_up_thermal_3d_reference_height_all_sky=self.flux_up_thermal_3d_reference_height_all_sky[0,:]   

            # BG: add quality-status
            self.quality_status = SD['quality_status'][()]         
            
        elif 'ACM_COM' in fn:
            # self.ice_water_content=SD['ice_water_content'][()]*1000 # Convert from kg/m**3 to g/m**3
            # self.ice_effective_radius=SD['ice_effective_radius'][()]*1e+6 # Convert from m to um
            # self.liquid_water_content=SD['liquid_water_content'][()]*1000 # Convert from kg/m**3 to g/m**3
            # self.liquid_effective_radius=SD['liquid_effective_radius'][()]*1e+6 # Convert from m to um
            # self.aerosol_extinction=SD['aerosol_extinction'][()]*1000 # Convert from /m to /km
            self.ice_water_content = np.asarray(SD['ice_water_content'][()], dtype=np.float64) * 1000.0
            self.ice_effective_radius = np.asarray(SD['ice_effective_radius'][()], dtype=np.float64) * 1e6
            self.liquid_water_content = np.asarray(SD['liquid_water_content'][()], dtype=np.float64) * 1000.0
            self.liquid_effective_radius = np.asarray(SD['liquid_effective_radius'][()], dtype=np.float64) * 1e6
            self.aerosol_extinction = np.asarray(SD['aerosol_extinction'][()], dtype=np.float64) * 1000.0

            self.ice_water_content_units = SD['ice_water_content'].attrs['units']
            self.ice_water_content_units = 'kg/m**3'
            self.ice_effective_radius_units = SD['ice_effective_radius'].attrs['units']
            self.ice_effective_radius_units = 'um'    
            self.liquid_water_content_units = SD['liquid_water_content'].attrs['units']
            self.liquid_water_content_units = 'kg/m**3'
            self.aerosol_extinction_units = SD['aerosol_extinction'].attrs['units']
            self.aerosol_extinction_units = '1/km'
            self.aerosol_classification=SD['aerosol_classification'][()]
                    # 0:Clear/not aerosol
                    # 10:Dust
                    # 11:Sea Salt
                    # 12:Continental Pollution
                    # 13:Smoke
                    # 14:Dusty smoke
                    # 15:Dusty mix
                    # 25:Stratospheric Ash
                    # 26:Stratospheric Sulfate
                    # 27:Stratospheric Smoke
            self.time = SD['time'][()]
            self.time_units = 'seconds since 2000-1-1 00:00:00.0 0:00'
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            self.latitude_active=SD['latitude_active'][()]
            self.longitude_active=SD['longitude_active'][()]
            self.height_layer=SD['height_layer'][()]
            self.height_level=SD['height_level'][()]
            self.pressure_level=SD['pressure_level'][()]
            self.pressure_layer_mean=SD['pressure_layer_mean'][()]
            self.temperature_level=SD['temperature_level'][()]
            self.temperature_layer_mean=SD['temperature_layer_mean'][()]
            self.volume_mixing_ratio_layer_mean_O3=SD['volume_mixing_ratio_layer_mean_O3'][()]
            self.volume_mixing_ratio_layer_mean_O2=SD['volume_mixing_ratio_layer_mean_O2'][()]
            self.specific_humidity_layer_mean=SD['specific_humidity_layer_mean'][()]
            self.specific_humidity_layer_mean_units = SD['specific_humidity_layer_mean'].attrs['units']
            self.volume_mixing_ratio_layer_mean_CO2=SD['volume_mixing_ratio_layer_mean_CO2'][()]
            self.volume_mixing_ratio_layer_mean_CH4=SD['volume_mixing_ratio_layer_mean_CH4'][()]
            self.volume_mixing_ratio_layer_mean_N2O=SD['volume_mixing_ratio_layer_mean_N2O'][()]

            self.liquid_water_content = self.liquid_water_content[atmosphere,:,:]
            self.liquid_effective_radius = self.liquid_effective_radius[atmosphere,:,:]
            self.ice_water_content = self.ice_water_content[atmosphere,:,:]
            self.ice_effective_radius = self.ice_effective_radius[atmosphere,:,:]
            self.aerosol_extinction = self.aerosol_extinction[atmosphere,:,:]
            self.aerosol_classification = self.aerosol_classification[atmosphere,:,:]

            self.surface_temperature = SD['surface_temperature'][()]
            self.albedo_direct_radiation_surface_visible = SD['albedo_direct_radiation_surface_visible'][()]
            self.albedo_direct_radiation_surface_near_infrared = SD['albedo_direct_radiation_surface_near_infrared'][()]
            self.albedo_diffuse_radiation_surface_visible = SD['albedo_diffuse_radiation_surface_visible'][()]
            self.albedo_diffuse_radiation_surface_near_infrared = SD['albedo_diffuse_radiation_surface_near_infrared'][()]

            # self.wavelengths_thermal_surface_emissivity = SD['wavelengths_thermal_surface_emissivity'][()]
            self.wavelengths_thermal_surface_emissivity = SD['wavenumbers_thermal_surface_emissivity'][()]
            self.types_surface_emissivity  = SD['types_surface_emissivity'][()]
            self.surface_emissivity_table  = SD['surface_emissivity_table'][()]
            self.surface_emissivity_type_index  = SD['surface_emissivity_type_index'][()]

            # BG: Variables needed for Ovean-Wave model 
            self.surface_albedo_classification = SD['surface_albedo_classification'][()]
            self.wind_speed_at_10_meters = SD['wind_speed_at_10_meters'][()]

            # BG: add quality-status
            self.quality_status = SD['quality_status'][()]
            self.quality_status = self.quality_status[atmosphere,:]

            
            #            print('self.liquid_water_content.shape', self.liquid_water_content.shape, self.liquid_water_content.max(),
            #                  self.aerosol_extinction.shape, self.aerosol_classification.shape)
            # Remove missing data. Not fully implemented
            if RemoveMissingData:
                self.surface_temperature = np.reshape(self.surface_temperature[ACM3D.indx],(ACM3D.across_track_shape,ACM3D.along_track_shape))
                self.latitude = np.reshape(self.latitude[ACM3D.indx],(ACM3D.across_track_shape,ACM3D.along_track_shape))
                self.longitude = np.reshape(self.longitude[ACM3D.indx],(ACM3D.across_track_shape,ACM3D.along_track_shape))
                
        elif 'BMA_FLX' in fn:
            SDGroup=file['ScienceData']
            self.bbr_direction = SDGroup['bbr_direction']
            if verbose:
                for key in SDGroup.keys():
                    print('key', key)
                    print('self.bbr_direction ', self.bbr_direction, self.bbr_direction.shape )
            self.bbr_directions = self.bbr_direction.shape[0]
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            # Is time in this product WRONG??? It differs from time in acm_com product
            #            self.time = SD['time'][()]  
            #            self.time_units = 'seconds since 2000-1-1 00:00:00.0 0:00'
            self.solar_zenith_angle = SD['solar_zenith_angle'][()]
            self.solar_zenith_angle_units = 'degrees'
            # Azimuth angle between the sun and the north. Measured clockwise
            self.solar_azimuth_angle = SD['solar_azimuth_angle'][()]
            self.solar_azimuth_angle_units = 'degrees'
            self.viewing_zenith_angle = SD['viewing_zenith_angle'][()]
            self.viewing_zenith_angle_units = 'degrees'
            self.viewing_azimuth_angle = SD['viewing_azimuth_angle'][()]
            self.viewing_azimuth_angle_units = 'degrees'

            self.solar_top_of_atmosphere_flux = SD['solar_top_of_atmosphere_flux'][()]
            self.solar_top_of_atmosphere_flux_units = 'W/m**2'
            self.solar_top_of_atmosphere_flux_error = SD['solar_top_of_atmosphere_flux_error'][()]
            self.solar_top_of_atmosphere_flux_error_units = 'W/m**2'
            self.solar_top_of_atmosphere_flux_quality_status = SD['solar_top_of_atmosphere_flux_quality_status'][()]

            self.solar_combined_top_of_atmosphere_flux = SD['solar_combined_top_of_atmosphere_flux'][()]
            self.solar_combined_top_of_atmosphere_flux_units = 'W/m**2'
            self.solar_combined_top_of_atmosphere_flux_error = SD['solar_combined_top_of_atmosphere_flux_error'][()]
            self.solar_combined_top_of_atmosphere_flux_error_units = 'W/m**2'
            self.solar_combined_top_of_atmosphere_flux_quality_status = SD['solar_combined_top_of_atmosphere_flux_quality_status'][()]

            self.thermal_combined_top_of_atmosphere_flux = SD['thermal_combined_top_of_atmosphere_flux'][()]
            self.thermal_combined_top_of_atmosphere_flux_units = 'W/m**2'
            self.thermal_combined_top_of_atmosphere_flux_error = SD['thermal_combined_top_of_atmosphere_flux_error'][()]
            self.thermal_combined_top_of_atmosphere_flux_error_units = 'W/m**2'
            self.thermal_combined_top_of_atmosphere_flux_quality_status = SD['thermal_combined_top_of_atmosphere_flux_quality_status'][()]

            # BG: add quality-status
            self.quality_status = SD['quality_status'][()]

            # Remove missing data. Not fully implemented
            RemoveMissingData=True
            if RemoveMissingData:
                #               print('latitude', self.latitude.shape, self.solar_zenith_angle.shape, self.solar_top_of_atmosphere_flux.shape); 
                self.indx = np.where(self.latitude <10000 )#> self.missing_value)
                self.along_track_shape=self.latitude.shape
                self.latitude = self.latitude[self.indx]
                self.longitude  = self.longitude[self.indx]
                self.solar_combined_top_of_atmosphere_flux=self.solar_combined_top_of_atmosphere_flux[self.indx]
                self.solar_combined_top_of_atmosphere_flux_quality_status = self.solar_combined_top_of_atmosphere_flux_quality_status[self.indx]
                self.solar_top_of_atmosphere_flux=self.solar_top_of_atmosphere_flux[self.indx[0],:]
                self.solar_top_of_atmosphere_flux_quality_status = self.solar_top_of_atmosphere_flux_quality_status[self.indx[0],:]
                self.thermal_combined_top_of_atmosphere_flux=self.thermal_combined_top_of_atmosphere_flux[self.indx]
                self.thermal_combined_top_of_atmosphere_flux_quality_status = self.thermal_combined_top_of_atmosphere_flux_quality_status[self.indx]
                #                print('latitude', self.latitude.shape, self.solar_zenith_angle.shape, self.solar_top_of_atmosphere_flux.shape); exit()
                tmp = np.zeros((len(self.indx[0]), self.bbr_directions))
                for ib in np.arange(self.bbr_directions):
                    tmp[:, ib]  = self.solar_zenith_angle[self.indx, ib]
                self.solar_zenith_angle  = tmp
                tmp = np.zeros((len(self.indx[0]), self.bbr_directions))
                for ib in np.arange(self.bbr_directions):
                    tmp[:, ib]  = self.solar_azimuth_angle[self.indx, ib]
                self.solar_azimuth_angle  = tmp
                #                print('latitude', self.latitude.shape, self.solar_zenith_angle.shape); #exit()

        elif 'libRad' in fn:
            SD=file
            if verbose:
                for key in SD.keys():
                    print('key', key)
 
            self.latitude=SD['latitude'][()]
            # self.longitude=SD['longitude'][()] 
            # self.solar_zenith_angle = SD['solar_zenith_angle'][()]
            # self.solar_zenith_angle_units = 'degrees'
            # # Azimuth angle between the sun and the north. Measured clockwise
            # self.solar_azimuth_angle = SD['solar_azimuth_angle'][()]
            # self.solar_azimuth_angle_units = 'degrees'
            # self.viewing_zenith_angle = SD['viewing_zenith_angle'][()]
            # self.viewing_zenith_angle_units = 'degrees'
            # self.viewing_azimuth_angle = SD['viewing_azimuth_angle'][()]
            # self.viewing_azimuth_angle_units = 'degrees'

            self.solar_eup = SD['solar_eup'][()]
            try:
                self.thermal_eup = SD['thermal_eup'][()]
            except:
                1;
            if 'mystic' in fn:
                self.solar_eup_std = SD['solar_eup_std'][()]
                self.thermal_eup_std = SD['thermal_eup_std'][()]
            self.solar_eup_units = 'W/m**2'
            self.thermal_eup_units = 'W/m**2'

        return

    def SetExtent(self):
        try:
            self.extent_left = self.longitude.min()
            self.extent_right = self.longitude.max()
            self.extent_bottom = self.latitude.min()
            self.extent_top = self.latitude.max()
        except:
            self.extent_bottom = self.latitude.min()
            self.extent_top = self.latitude.max()

    def WriteNetcdf(self, fn, verbose=True):
        if verbose:
            print("Writing libRadtran output to netcdf file: ", fn)

        ncf = Dataset(fn, 'w')

        along_tracks = self.latitude_active.shape[0]
        ncf.createDimension('along_track', along_tracks)

        latitude = ncf.createVariable('latitude',np.dtype('float').char,('along_track',))
        latitude.units = "degree_north" 
        latitude.long_name = "Latitude"
        latitude[:] = self.latitude_active

        solar_eup = ncf.createVariable('solar_eup',np.dtype('float').char,('along_track',))
        solar_eup.units = "W m-2" 
        solar_eup.long_name = "Solar upward flux at TOA"
        solar_eup[:] = self.solar_eup
        
        solar_eup_std = ncf.createVariable('solar_eup_std',np.dtype('float').char,('along_track',))
        solar_eup_std.units = "W m-2" 
        solar_eup_std.long_name = "Standard deviation of solar upward flux at TOA"
        solar_eup_std[:] = self.solar_eup_std

        thermal_eup = ncf.createVariable('thermal_eup',np.dtype('float').char,('along_track',))
        thermal_eup.units = "W m-2" 
        thermal_eup.long_name = "Thermal upward flux at TOA"
        thermal_eup[:] = self.thermal_eup
        
        thermal_eup_std = ncf.createVariable('thermal_eup_std',np.dtype('float').char,('along_track',))
        thermal_eup_std.units = "W m-2" 
        thermal_eup_std.long_name = "Standard deviation of thermal upward flux at TOA"
        thermal_eup_std[:] = self.thermal_eup_std
                
        ncf.close()
        




















###########################################################################################################################        
if __name__ == "__main__":
    # """
    # MAPS: INCLUDE UNITS
    #         surface\_emissivity\_type\_index & ACM-COM \\
    #         surface\_albedo\_classification & ACM-COM \\
    #         surface\_emissivity\_classification & ACM-COM \\
    #         surface\_emissivity\_table & ACM-COM \\
    #         wind\_speed\_at\_10\_meters & ACM-COM \\

    # CURTAINS
    #         liquid\_effective\_radius & ACM-COM \\
    #         aerosol\_classification & ACM-COM \\

    #         temperature\_level & ACM-COM \\
    #         temperature\_layer\_mean & ACM-COM \\
    #         pressure\_level & ACM-COM \\
    #         pressure\_layer\_mean & ACM-COM \\
    #         height\_level & ACM-COM \\
    #         height\_layer & ACM-COM \\
    #         volume\_mixing\_ratio\_layer\_mean\_O3 & ACM-COM \\
    #         volume\_mixing\_ratio\_layer\_mean\_CO2 & ACM-COM \\
    #         volume\_mixing\_ratio\_layer\_mean\_ch4 & ACM-COM \\
    #         volume\_mixing\_ratio\_layer\_mean\_N2O & ACM-COM \\
    #         volume\_mixing\_ratio\_layer\_mean\_CFC11 & ACM-COM \\
    #         volume\_mixing\_ratio\_layer\_mean\_CFC22 & ACM-COM \\
    #         volume\_mixing\_ratio\_layer\_mean\_CCL4 & ACM-COM \\
    #         volume\_mixing\_ratio\_layer\_mean\_O2 & ACM-COM \\
    # """
    # To reproduce any figure in the NEVAR WP4 report, set figname to the
    # corresponding figure \label in the NEVAR WP4 latex file. 

    verbose = False 

    # BG: ----- Things to remember ---------
    #   1. want_3D
    #   2. idx_scene
    #   3. fig_index
    # --------------------------------------

 
    want_3D = True      # BG: used in flux plot (DISORT or MYSTIC)
    want_ps = False     # BG: if want DISORT pseudospherical (want_3D overwrite want_ps)

    idx_scene = [7] 
    # idx_scene = [3, 4, 5, 8, 11]
    # idx_scene =[3,4,5,6,7,8]
    SceneNames = [['Orbit_05378D'],#0           # Marocco - Norway           # Previousy ['Arctic_05378D']
                  ['Orbit_05458F'],             # Chile
                  ['Orbit_05926C'],             # Old Greenland (13.06.2026)

                  ['Orbit_06888C'],#3           # Svalbard (14.08.2025)     
                  ['Orbit_07277C'],             # Svalbard (08.09.2025)

                  ['Orbit_06518D'],#5           # USA (21.07.2025)                
                  ['Orbit_06907D'],             # USA (15.08.2025)                        

                  ['Orbit_06497E'],#7           # Africa (20.07.2025)                            
                  ['Orbit_06886E'],             # Africa (14.08.2025)                             

                  ['Orbit_06600C'],#9           # Greenland (27.07.2025)          
                  ['Orbit_06662C'],             # Greenland (31.07.2025)   

                  ['Orbit_06331C'] #11          # Greenland (09.07.2025)        
                  ]

    SceneNames = [SceneNames[i][0] for i in idx_scene]

    # -------------------- Small hint for modification in .nc-file ---------
    additional_spesifications = '' 
    # additional_spesifications += '_TEST'
            # Aerosol-Versions
    # additional_spesifications += '_AOD(dynamic0.3)'
    # additional_spesifications += '_AOD(dynamic0.2)'
    # additional_spesifications += '_AOD(default)'
    # additional_spesifications += '_AOD(alpha0.8)'
    # additional_spesifications += '_AOD(alpha0.5)'
    # additional_spesifications += '_AOD(alpha0.3)'
    # additional_spesifications += '_AOD(alpha0.2)'
            # Ice-Habits
    # additional_spesifications += '_GHM'
    # additional_spesifications += '_SC'
    # additional_spesifications += '_RA'
            # mc_photons
    # additional_spesifications += '_GHM_mc1'
    # additional_spesifications += '_GHM_mc5'
    # additional_spesifications += '_GHM_mc25'
    # additional_spesifications += '_GHM_mc1e2'
    # additional_spesifications += '_GHM_mc1e3'
    # additional_spesifications += '_GHM_mc1e4'     
    # additional_spesifications += '_GHM_mc1e5'
    # additional_spesifications += '_GHM_mc1e6'
    # additional_spesifications += '_GHM_mc1e7'
           # All points 
    # additional_spesifications += '_All_FullBuffer'
    # additional_spesifications += '_All_SmallBuffer'
             # 3D-Cloud impact
    # additional_spesifications += '_wc'
    additional_spesifications += '_wc_test_atm_0'
    atmosphere = 0


    
    # ----------------------------------------------------------------------



    fig_index = 2
    figname = ['fig:flx_solar',#0                                                   
               'fig:flx_thermal',                                                   
               'fig:flx_both',                      # Solar + Thermal               

               'fig:librad_solar_flx_diff',#3       # MYSTIC / DISORT
               'fig:librad_thermal_flx_diff',       # MYSTIC / DISORT

               'fig:solar_flx_ratio',#5             # (MYSTIC/DISORT) / BMA_FLX         
               'fig:thermal_flx_ratio',             # (MYSTIC/DISORT) / BMA_FLX         
               'fig:both_flx_ratio',                # (MYSTIC/DISORT) / BMA_FLX         

               'fig:solar_flx_diff',#8              # (MYSTIC/DISORT) vs BMA_FLX        
               'fig:thermal_flx_diff',              # (MYSTIC/DISORT) vs BMA_FLX        
               'fig:both_flx_diff',                 # (MYSTIC/DISORT) vs BMA_FLX        

               'fig:solar_flx_diff_histogram',#11   # (MYSTIC/DISORT) vs BMA_FLX        
               'fig:thermal_flx_diff_histogram',    # (MYSTIC/DISORT) vs BMA_FLX        
               'fig:both_flx_diff_histogram',       # (MYSTIC/DISORT) vs BMA_FLX        

               'fig:correlation_diff',#14

               'fig:integrated_iwc',#15         # 2D plot
               'fig:integrated_lwc',            # 2D plot
               'fig:liquid_water_content',#17   # Curtain
               'fig:ice_water_content',         # Curtain
               'fig:earthCARE_flx_rel_err',         # ACM_RT 1D + 3D vs BMA_FLX
               'fig:plot_swat',#20
               'fig:plot_info'
               ][fig_index]


    
    # BG: fig:*both* when simulated in solar & thermal ('*solar,thermal*'-name in RESULT .nc files)

    pathL2TestProducts  = '/xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/'  
    ProductPathRTM      = './RESULTS/' # './netcdf/' 
    plotdir_base        = './figures/'   

      

    latitude_wanted     = 40.0
    librad_type         = 'SWIA'
    # librad_type         = 'SNIA'
    # librad_type         = 'SWNA'
    # librad_type         = 'SNNA'
    # librad_type         = 'NWIA'
    # librad_type         = 'SWIN'
    librad_version2     = ''
    Product2            = False
    version_identifier  = 'v01'

    

    # BG: Chose idx to select additional info on solar_both and thermal_both plot 
    # Chose from: 
    #   [None, 'elevation', 'lwp', 'iwp', 'tot_wp', 'albedo', 'aerosols', 'surface_temperature', 'CF']
    #           NOTE: do not put 'CF' at the end of quantity_list
    quantity_list = False
    # quantity_list = ['tot_wp']
    quantity_list = ['tot_wc']
    # quantity_list = ['CF']
    # quantity_list = ['CF', 'tot_wp']

    

    # BG: additional plot-settings
    want_average_line   = False                # Average line of librad-flux-values
    want_EarthCARE_info = True                 # Sets Scene3 = ACM3D (in PlotLien)
    want_product2       = True                 # Sets Scene4 = librad2 (in PlotLine)
    want_SZA            = True                 # Prints out SZA
    want_Cloud_Fraction = True                 # Prints out CF
    if want_Cloud_Fraction: want_2D = False    # CF of 2D swat, or 1D nadir column
    stacked = True                             # If add quanteties to plot, if should get own figure below
    



    

    # --------------------------------------------------------------------------------------------------------------------------------
    if figname != '':
       plot_types_map       = []
       plot_types_curtain   = []
       plot_types_line_solar= []
       plot_types_librad    = []
       plot_types_flx_geo   = []
       plot_types_flx_solar = []
       plot_types_flx_thermal = []

    # determine which source string goes into the filename
    if figname == 'fig:flx_solar':
        plot_types_flx_solar = ['solar_both']
        source_str = 'solar'
    elif figname == 'fig:flx_thermal':
        plot_types_flx_thermal = ['thermal_both']
        source_str = 'thermal'
    elif figname == 'fig:flx_both':
        plot_types_flx_solar   = ['solar_both']
        plot_types_flx_thermal = ['thermal_both']
        source_str = 'solar,thermal'
        if want_product2: Product2 = True
    elif figname == 'fig:librad_solar_flx_diff':
        want_ratio = False
        plot_types_librad = ['librad_solar_flx_diff']
        Product2 = True 
        source_str = 'solar,thermal'
    elif figname == 'fig:librad_thermal_flx_diff':
        want_ratio = False
        plot_types_librad = ['librad_thermal_flx_diff']
        Product2 = True 
        source_str = 'solar,thermal'
    elif figname == 'fig:solar_flx_ratio':
        plot_types_flx_solar = ['solar_flx_ratio']
        source_str = 'solar'
    elif figname == 'fig:thermal_flx_ratio':
        plot_types_flx_thermal = ['thermal_flx_ratio']
        source_str = 'thermal'
    elif figname == 'fig:both_flx_ratio':
        plot_types_flx_solar   = ['solar_flx_ratio']
        plot_types_flx_thermal = ['thermal_flx_ratio']
        source_str = 'solar,thermal'
    elif figname == 'fig:solar_flx_diff':
        plot_types_flx_solar = ['solar_flx_diff']
        source_str = 'solar'
    elif figname == 'fig:thermal_flx_diff':
        plot_types_flx_thermal = ['thermal_flx_diff']
        source_str = 'thermal'
    elif figname == 'fig:both_flx_diff':
        plot_types_flx_solar   = ['solar_flx_diff']
        plot_types_flx_thermal = ['thermal_flx_diff']
        source_str = 'solar,thermal'
    elif figname == 'fig:solar_flx_diff_histogram':
        plot_types_flx_solar   = ['solar_flx_diff_histogram']
        source_str = 'solar'
    elif figname == 'fig:thermal_flx_diff_histogram':
        plot_types_flx_thermal   = ['thermal_flx_diff_histogram']
        source_str = 'thermal'
    elif figname == 'fig:both_flx_diff_histogram':
        plot_types_flx_solar    = ['solar_flx_diff_histogram']
        plot_types_flx_thermal  = ['thermal_flx_diff_histogram']
        source_str = 'solar,thermal'
    elif figname == 'fig:correlation_diff':
        plot_types_flx_solar    = ['solar_flx_diff_correlation']
        plot_types_flx_thermal  = ['thermal_flx_diff_correlation']
        source_str = 'solar,thermal'
    elif figname == 'fig:plot_info':
        plot_types_flx_geo = ['plot_info']
        source_str = 'solar,thermal'
    else:
        source_str = ''  # for non-flux figures


   
    # Rest fignames without/independent of Products
    if   figname == 'fig:integrated_lwc':
        plot_types_map = ['integrated_lwc']
        source_str = 'solar,thermal'; Product2 = True
    elif figname == 'fig:integrated_iwc':
        plot_types_map = ['integrated_iwc']
        source_str = 'solar,thermal'; Product2 = True
    elif figname == 'fig:earthCARE_flx_rel_err':
        plot_types_flx_solar   = ['solar_earthCARE_flx_rel_err']
        plot_types_flx_thermal = ['thermal_earthCARE_flx_rel_err']
        Product = 'libRad_v01_montecarlo_3D_SWIA_solar_Arctic_05378D.nc' # Just put a earlier run .nc-file. Is not used!
    elif figname == 'fig:liquid_water_content':
        plot_types_curtain = ['liquid_water_content']
        source_str = 'solar,thermal'; Product2 = True
    elif figname == 'fig:ice_water_content':
        plot_types_curtain = ['ice_water_content']
        source_str = 'solar,thermal'; Product2 = True
    elif figname == 'fig:ice_effective_radius':
        plot_types_curtain = ['ice_effective_radius']
    elif figname == 'fig:aerosol_extinction':
        plot_types_curtain = ['aerosol_extinction']
    elif figname == 'fig:index_construction':
        plot_types_map = ['index_construction']
    elif figname == 'fig:plot_swat':
        plot_types_map = ['plot_swat']
    



    # pick the correct model version
    if want_3D:
        librad_version = 'montecarlo_3D'
        if Product2: 
            if want_ps:
                librad_version2 = 'disort_pseudospherical_1D' 
            else: 
                librad_version2 = 'disort_1D'
    else:
        if want_ps:
            librad_version = 'disort_pseudospherical_1D' # BG: this might be correct: 'disort '$'\n''pseudospherical'
        else:
            librad_version = 'disort_1D' # 'twostr_1D' 
        if Product2: librad_version2 = 'montecarlo_3D'



    librad_version2 =  'disort_1D'                  
            # Chose from: 'montecarlo_3D'  'disort_1D'   librad_version
    additional_spesifications2 =   additional_spesifications   
            #Chose from: '_wc'   '_All_SmallBuffer'    '_GHM'   additional_spesifications

                        # librad_type         = 'SWIN'; additional_spesifications2 = ''


    

    plot_types_flx = plot_types_flx_geo + plot_types_flx_solar + plot_types_flx_thermal

    for SceneName in SceneNames:
        # build Product once, if relevant -----------
        # SceneName = SceneName
        if source_str:
            Product = (
                'libRad_' + 
                version_identifier  + '_' +
                librad_version      + '_' +
                librad_type         + '_' +
                source_str          + '_' +
                SceneName           + 
                additional_spesifications
                                    + '.nc'
            )

            if Product2: 
                Product2 = (
                    'libRad_' + 
                    version_identifier  + '_' +
                    librad_version2     + '_' +
                    librad_type         + '_' +
                    source_str          + '_' +
                    SceneName           + 
                    additional_spesifications2
                                        + '.nc'
                )
        # --------------------------------------------

        # BG: create path to Orbit_number + type_solver ----------
        # Choose subfolder based on want_3D
        if want_3D:
            mode_folder = 'MYSTIC/'
        else:
            if want_ps:
                mode_folder = 'PSEUDOSPHERICAL/'
            else:
                mode_folder = 'DISORT/' #'TWOSTR/' 

        plotdir = os.path.join(plotdir_base, SceneName, mode_folder)
        png_spesifications = ''
        

        #### Use this to add to ./github_figures/ ####
        # plotdir_base        = ['./github_figures/3D_buffer/', './github_figures/ice_habit/MYSTIC/', './github_figures/ice_habit/DISORT/'][2] 
        # plotdir = plotdir_base 

        # # png_spesifications += '_surface_properties'
        # png_spesifications += '_RA_vs_GHM'
        # png_spesifications += '_SC_vs_GHM'
        # ###############################################
        
        # Make sure it exists
        os.makedirs(plotdir, exist_ok=True)
        print(f'Direcotry for figures: {plotdir}')
        # -------------------------------------------------------

        
        


        if len(plot_types_librad)>0 or len(plot_types_flx)>0:       
            ProductPath = ProductPathRTM
            ProductFile = os.path.join(ProductPath, Product)
            # print('ProductFile1', ProductFile)
            ProductFile = sorted(glob.glob(ProductFile))[0]    
            libRad = Scene(Name=SceneName, verbose=verbose)
            libRad.ReadEarthCAREh5(ProductFile, verbose=verbose)
            libRad.SetExtent()

            PrintStuff=False
            if PrintStuff:
                ia=0
                for lat, eup, std in zip(libRad.latitude, libRad.solar_eup, libRad.solar_eup_std ):
                    print(ia, lat, eup, std)
                    ia=ia+1
                exit()
            
            
            
        libRad2=None
        if Product2:      
            ProductPath = ProductPathRTM
            ProductFile = os.path.join(ProductPath, Product2) # BG: removed this [+'*'+SceneName+'*.nc')] and changed Product -> Product2
            # print('ProductFile2', ProductFile)
            ProductFile = sorted(glob.glob(ProductFile))[0]
            libRad2 = Scene(Name=SceneName, verbose=verbose)
            libRad2.ReadEarthCAREh5(ProductFile, verbose=verbose)
            libRad2.SetExtent()
            

        # BG: Find; Baseline, Data, Time
        BA_baseline = ''
        AC_baseline = ''
        
        
        Product ='ACM_RT_'#'ACM_COM' #
        #ProductPath = 'ECA_EXAB_'+Product+'*'  #'ECA_EXAA_'+Product+'*'
        #ProductPath = 'ECA_EXAB_'+Product+'*' #BG: marked out this for Orbit_05926C
        # ProductPath = 'ECA_EXAC_'+Product+'*'
        ProductPath = '*'+Product+'*'
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        ACMRT = Scene(Name=SceneName, verbose=verbose)
        ACMRT.ReadEarthCAREh5(ProductFile, verbose=verbose)
        ACMRT.SetExtent()
        # Extract Baseline ---------------------------   
        parts = ProductFile.split("ECA_EX", 1)
        out = parts[1][:2] if len(parts) > 1 else None
        if out == 'BA': BA_baseline += ' ACM_RT'
        elif out == 'AC': AC_baseline += ' ACM_RT'
        # print(out)
        #---------------------------------------------
        
    
        Product ='ALL_3D_'#'ACM_COM' #'ACM_3D_'#'ACM_COM' #
        # ProductPath = 'ECA_EXAB_'+Product+'*'  # 'ECA_EXAA_'+Product+'*'
        ProductPath = '*'+Product+'*' 
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        ACM3D = Scene(Name=SceneName, verbose=verbose)
        ACM3D.ReadEarthCAREh5(ProductFile, verbose=verbose)
        ACM3D.SetExtent()
        
        indlat = find_nearest_id(ACM3D.latitude,latitude_wanted)
        latitude_have_ACM3D = ACM3D.latitude.flatten()[indlat]
        val = latitude_wanted
        e=ACM3D.latitude
        nearest = np.unravel_index(np.argmin(np.abs(e - val), axis=None), e.shape)
        # Extract Baseline ---------------------------   
        parts = ProductFile.split("ECA_EX", 1)
        out = parts[1][:2] if len(parts) > 1 else None
        if out == 'BA': BA_baseline += ' ALL_3D'
        elif out == 'AC': AC_baseline += ' ALL_3D'
        # print(out)
        #---------------------------------------------
        
        Product ='BMA_FLX'
        # ProductPath = 'ECA_EXAB_'+Product+'*'  #'ECA_EXAA_'+Product+'*'
        ProductPath = '*'+Product+'*'  
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        BMAFLX = Scene(Name=SceneName, verbose=verbose)
        BMAFLX.ReadEarthCAREh5(ProductFile, Resolution='StandardResolution', verbose=verbose)
        BMAFLX.SetExtent()
        indlat = find_nearest_id(BMAFLX.latitude,latitude_wanted)
        latitude_have_BMAFLX = BMAFLX.latitude.flatten()[indlat]
        val = latitude_wanted
        e=BMAFLX.latitude
        nearest = np.unravel_index(np.argmin(np.abs(e - val), axis=None), e.shape)

        if len(plot_types_librad)>0 and len(plot_types_flx)>0:
            indlats=[]
            for latitude_wanted in libRad.latitude:
                nearest = np.unravel_index(np.argmin(np.abs(BMAFLX.latitude - latitude_wanted), axis=None), BMAFLX.latitude.shape)
                indlats.append(nearest[0])
            libRad.BMAFLXindlats = indlats
        # Extract Baseline ---------------------------   
        parts = ProductFile.split("ECA_EX", 1)
        out = parts[1][:2] if len(parts) > 1 else None
        if out == 'BA': BA_baseline += ' ' + Product
        elif out == 'AC': AC_baseline += ' ' + Product
        # print(out)
        #---------------------------------------------


        Product ='ACM_COM'
        # ProductPath = 'ECA_EXAC_'+Product+'*'  # 'ECA_EXAA_'+Product+'*'
        ProductPath = '*'+Product+'*'
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        ACMCOM = Scene(Name=SceneName, verbose=verbose)
        ACMCOM.ReadEarthCAREh5(ProductFile, verbose=verbose, ACM3D=ACM3D)
        ACMCOM.SetExtent()
        setE=False #True
        if setE:
            ACMCOM.extent_left = -56.4
            ACMCOM.extent_right = -56.1
            ACMCOM.extent_bottom = 60.95
            ACMCOM.extent_top = 61.35

        
        indlat = find_nearest_id(ACMCOM.latitude,latitude_wanted)
        latitude_have_ACMCOM = ACMCOM.latitude.flatten()[indlat]
        # Extract Baseline ---------------------------   
        parts = ProductFile.split("ECA_EX", 1)
        out = parts[1][:2] if len(parts) > 1 else None
        if out == 'BA': BA_baseline += ' ' + Product
        elif out == 'AC': AC_baseline += ' ' + Product
        # print(out)
        #---------------------------------------------
            

        Product = 'AM__ACD'
        ProductPath = '*'+Product+'*'  
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        # Extract Baseline ---------------------------
        parts = ProductFile.split("ECA_EX", 1)
        out = parts[1][:2] if len(parts) > 1 else None
        if out == 'BA': BA_baseline += ' ' + Product
        elif out == 'AC': AC_baseline += ' ' + Product
        # print(out)
        #---------------------------------------------


        # Fix Baseline
        AC_baseline = ", ".join(AC_baseline.split())
        BA_baseline = ", ".join(BA_baseline.split())

        # Extract Date
        date_num = '20' + ProductFile.split('20', 1)[1].split('T', 1)[0]
        date = date_num[6:8] + "." + date_num[4:6] + "." + date_num[0:4]

        # Extract Time
        first_time_num = ProductFile.split('T')[1].split('Z')[0]
        second_time_num = ProductFile.split('T')[2].split('Z')[0] 
        start_time = first_time_num[:2] + ":" + first_time_num[2:4]
        end_time = second_time_num[:2] + ":" + second_time_num[2:4]
        time = start_time + '-' + end_time

        # print(  f'\nAC_baseline = {AC_baseline}\nBA_baseline = {BA_baseline}\n' 
        #         f'{date} - {time}\n')
        




        for plot_type in plot_types_map:
            pngfile = plotdir+ SceneName+'_'+plot_type+'.png' #'' #
            if plot_type=='index_construction':
                ACM3D.Plot2D(plot_type=plot_type, pngfile=pngfile, verbose=verbose)
            elif plot_type=='integrated_iwc' or plot_type=='integrated_lwc':
                if Product2: ACMCOM.Plot2D(plot_type=plot_type, pngfile=pngfile, verbose=verbose, ACM3D=ACM3D, libRad=libRad2)
                else:        ACMCOM.Plot2D(plot_type=plot_type, pngfile=pngfile, verbose=verbose, ACM3D=ACM3D)
            else:
                ACMCOM.Plot2D(plot_type=plot_type, pngfile=pngfile, verbose=verbose)

        for plot_type in plot_types_line_solar:
            pngfile = plotdir+ SceneName+'_'+plot_type+'_Line.png' # '' #
            ACMCOM.PlotLine(plot_type=plot_type, pngfile=pngfile, verbose=verbose)

                
        for plot_type in plot_types_curtain:
            pngfile = plotdir+ SceneName+'_'+plot_type+'.png' #'' #
            if Product2: ACMCOM.PlotCurtain(plot_type=plot_type, pngfile=pngfile, verbose=verbose, libRad=libRad2)
            else:        ACMCOM.PlotCurtain(plot_type=plot_type, pngfile=pngfile, verbose=verbose)

        for plot_type in plot_types_flx:
            pngfile = plotdir+ SceneName+'_'+plot_type+'_'+librad_version+'_'+librad_type+'.png' #'' #
           
            # Info: Scene3 = None -> avoid potting ACM_RT 1D and 3D
            Scene3 = None 
            Scene4 = None
            if want_EarthCARE_info: 
                Scene3 = ACMRT
            if Product2: 
                Scene4 = libRad2
                # ----------- Add Scene4 to PlotLine!!!!! -----------------------

            BMAFLX.PlotLine(plot_type=plot_type, pngfile=pngfile, verbose=verbose, Scene2=libRad, Scene3=Scene3, Scene4=Scene4)
            

        for plot_type in plot_types_librad:
            pngfile = plotdir+ SceneName+'_'+plot_type+'_'+librad_version+'_'+librad_type+ png_spesifications + '.png' #'' #
            libRad.PlotLine(plot_type=plot_type, pngfile=pngfile, verbose=verbose, Scene2=libRad2)
            
        print(f"png-file: {pngfile}")


        # Print INFO: --------------------------------------------------------------------------
        print("------------------------------ INFO -----------------------------")
        # SZA
        if want_SZA:
            print(f"SZA (mean) = {get_SZA(BMAFLX, property='mean')}")  # can set property='mean'
        # CF:
        if want_Cloud_Fraction: 
            dim = '2D' if want_2D else '1D'
            cf  = calculate_cloud_fraction(ACMCOM, ACM3D, want_2D=want_2D)
            icf = calculate_cloud_fraction(ACMCOM, ACM3D, want_2D=want_2D, want_ice=True)
            print(f"Cloud Fraction ({dim}) = {cf}\nIce-Cloud Fraction ({dim}) = {icf}")
        print("------------------------------------------------------------------")
        # --------------------------------------------------------------------------------------
            







# BG---------------------------------- ERROR NOTES --------------------------------------------------
# Traceback (most recent call last):
#   File "/homevip/bgre/RTM/ReadEarthCAREL2_bg.py", line 1615, in <module>
#     ProductFile = sorted(glob.glob(ProductFile))[0]
#                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
# IndexError: list index out of range
#         -> Assume error in .nc file 
#         -> *solar*.nc?
#         -> *thermal*.nc?
#         -> *solar,thermal*.nc?
# ------------------------------------------------------------------------------------------------








































    ################################# OLD SYSTEM #############################################
    # want_3D = True # BG: Used in flux-plots to change between DISORT and MYSTIC runs

    # idx_scene = 0
    # SceneNames = [['Arctic_05378D'],            # Marocco - Norway
    #               ['Orbit_05458F']][idx_scene]  # Chile
   
    # fig_index = 6
    # figname = ['fig:flx_solar',                                                     # BG: remember "want_3D"
    #            'fig:flx_thermal',                                                   # BG: remember "want_3D"
    #            'fig:flx_both',                  # Solar + Thermal               

    #            'fig:librad_solar_flx_ratio',    # MYSTIC / DISORT
    #            'fig:librad_thermal_flx_ratio',  # MYSTIC / DISORT

    #            'fig:solar_flx_ratio',           # (MYSTIC/DISORT) / BMA_FLX         # BG: remember "want_3D"
    #            'fig:thermal_flx_ratio',         # (MYSTIC/DISORT) / BMA_FLX         # BG: remember "want_3D"

    #            'fig:integrated_iwc', 
    #            'fig:integrated_lwc'][fig_index]
  

    # pathL2TestProducts = '/xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/'  #TMS
    # ProductPathRTM = './RESULTS/' # './netcdf/' #ESO
    # plotdir = './figures/'   #TMS

    # # pathL2TestProducts = '//xnilu_wrk/users/eso/NEVAR/Products/'  #ESO
    # # ProductPathRTM = './ProductRTM/' #TMS
    


    # latitude_wanted = 40.0
    # librad_type ='SWIA' 
    # librad_version2 = ''
    # Product2 = False
    

    # # --------------------------------------------------------------------------------------------------------------------------------
    #             # 'fig:Halifax_librad_solar_flx_ratio_mysticSWIA' 
    #             # 'fig:Halifax_librad_solar_flx_ratio_v00SWIA'
    #             # 'fig:aerosol_extinction'
    #             #figname = 'fig:index_construction' # 'fig:Halifax_surface_albedo' # 'fig:Halifax_liquid_water' # 'fig:integrated_lwc' #
    #             # 'fig:Halifax_ice_water' # 'fig:integrated_iwc' # 'fig:Halifax_aerosol'
    #             # 'fig:Halifax_librad_solar_flx_ratio_vpseudosphericalSWIA'
    #             # 'fig:albedo_diffuse_radiation_surface_visible' # 'fig:albedo_diffuse_radiation_surface_near_infrared'
    #             # 'fig:liquid_water_content' # 'fig:ice_water_content' # 'fig:ice_effective_radius'
                
    #             # #pathL2TestProducts = '/home/aky/NILU/xnilu_wrk2/projects/NEVAR/data/EarthCARE_L2_Test_Products-v10.10/'
    #             # #pathL2TestProducts = '/xnilu_wrk2/projects/NEVAR/data/EarthCARE_L2_Test_Products-v10.10/'  #TMS
    #             # #ProductPathRTM = '/xnilu_wrk2/projects/NEVAR/data/libRadtran-SyntheticScenes/' #TMS
    #             # pathL2TestProducts = '/xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/'  #TMS
    #             # #pathL2TestProducts = '//xnilu_wrk/users/eso/NEVAR/Products/'  #ESO
    #             # #ProductPathRTM = './ProductRTM/' #TMS
    #             # ProductPathRTM = './RESULTS/' # './netcdf/' #ESO
                
    #             # #ProductPathRTM = './tmpRTIO/' #TMS
    #             # #ProductPathRTM = '/xnilu_wrk2/projects/NEVAR/data/libRadtran-SyntheticScenes/' #TMS

    #             # #SceneNames = ['Arctic_05384B','Baja','Halifax']# ['Arctic_05384B'] #  ['Halifax'] # 
    #             # SceneNames = ['Arctic_05378D']#['Arctic_05384B','Baja','Halifax']# ['Arctic_05384B'] #  ['Halifax'] # 
    #             # latitude_wanted = 40.0
    #             # librad_type ='SWIA' #'SWIN' #'SWNN' #'SNNN'# 
    #             # #librad_version =  'v02_montecarlo_3D_' #'vmystic033D' #'v00' #'vmystic' #
    #             # #librad_version = 'v01_disort_1D_' #'' #'vdisort' #'vmystic' #'vpseudospherical' # #
    #             # #librad_version = 'v08_disort_1D_' #'' #'vdisort' #'vmystic' #'vpseudospherical' # #
    #             # librad_version2 = ''

    #             # plotdir = './figures/'   #TMS
    #             #    plot_types = ['albedo_direct_radiation_surface_visible','surface_temperature']
    #             # plot_types_map = ['integrated_iwc'] #['integrated_lwc'] # ['index_construction'] #['albedo_direct_radiation_surface_near_infrared','albedo_diffuse_radiation_surface_visible', 'albedo_diffuse_radiation_surface_near_infrared']
    #             # plot_types_curtain = ['liquid_water_content','ice_water_content', 'aerosol_extinction','ice_effective_radius' ] #
    #             # #    plot_types_curtain = ['liquid_water_radius','ice_effective_radius' ] #
    #             # plot_types_curtain = [] #['specific_humidity_layer_mean'] #
    #             # plot_types_line_solar = [] #['aerosol_column'] # ['liquid_water_column','ice_water_column', 'albedo_direct_radiation_surface_visible']
                
    #             # plot_types_librad = ['librad_solar_flx', 'librad_thermal_flx'] #[] # ['librad_solar_flx_ratio']
                
    #             # plot_types_flx_geo =  [] #['solar_zenith_angle', 'solar_azimuth_angle']
    #             # plot_types_flx_solar = [] #['solar_both'] #'solar_diff', ] #['solar_combined_top_of_atmosphere_flux', 'solar_combined_top_of_atmosphere_flux_quality_status'] # 
    #             # plot_types_flx_thermal =  ['thermal_combined_top_of_atmosphere_flux'] #['thermal_combined_top_of_atmosphere_flux', 'thermal_combined_top_of_atmosphere_flux_quality_status'] #
    # # --------------------------------------------------------------------------------------------------------------------------------

    # if figname != '':
    #    plot_types_map = []
    #    plot_types_curtain = []
    #    plot_types_line_solar = []
    #    plot_types_librad = []
    #    plot_types_flx_geo =  []
    #    plot_types_flx_solar = []
    #    plot_types_flx_thermal =  [] 
    
    # if figname == 'fig:flx_solar':
    #     plot_types_flx_solar = ['solar_both']

    #     if want_3D:
    #         librad_version = 'v01_montecarlo_3D_'
    #         Product=['libRad_v01_montecarlo_3D_SWIA_solar_Arctic_05378D.nc',
    #                  'libRad_v01_montecarlo_3D_SWIA_solar_Orbit_05458F.nc'][idx_scene]
    #     else:
    #         librad_version = 'v01_disort_1D_'
    #         Product=['libRad_v01_disort_1D_SWIA_solar_Arctic_05378D.nc',
    #                  'libRad_v01_disort_1D_SWIA_solar_Orbit_05458F.nc'][idx_scene]

    
    # elif figname == 'fig:flx_thermal':
    #     plot_types_flx_thermal = ['thermal_both']

    #     if want_3D:
    #         librad_version = 'v01_montecarlo_3D_'
    #         Product= ['libRad_v01_montecarlo_3D_SWIA_thermal_Arctic_05378D.nc',
    #                   'libRad_v01_montecarlo_3D_SWIA_thermal_Orbit_05458F.nc'][idx_scene]
    #     else:
    #         librad_version = 'v01_disort_1D_'
    #         Product=['libRad_v01_disort_1D_SWIA_thermal_Arctic_05378D.nc',
    #                  'libRad_v01_disort_1D_SWIA_thermal_Orbit_05458F.nc'][idx_scene]
            

    # if figname == 'fig:flx_both':
    #     plot_types_flx_solar = ['solar_both']
    #     plot_types_flx_thermal = ['thermal_both']

    #     if want_3D:
    #         librad_version = 'v01_montecarlo_3D_'
    #         Product=['libRad_v01_montecarlo_3D_SWIA_solar,thermal_Arctic_05378D.nc',
    #                  'libRad_v01_montecarlo_3D_SWIA_solar,thermal_Orbit_05458F.nc'][idx_scene]
    #     else:
    #         librad_version = 'v01_disort_1D_'
    #         Product=['libRad_v01_disort_1D_SWIA_solar,thermal_Arctic_05378D.nc',
    #                  'libRad_v01_disort_1D_SWIA_solar,thermal_Orbit_05458F.nc'][idx_scene]
        
        
    # elif figname == 'fig:integrated_lwc':
    #     plot_types_map = ['integrated_lwc']

    # elif figname == 'fig:integrated_iwc':
    #     plot_types_map = ['integrated_iwc']

    # elif figname == 'fig:librad_solar_flx_ratio':
    #     librad_version = 'v01_montecarlo_disort_' 
    #     #librad_version2 =  'v01_disort_1D_' 
    #     plot_types_librad = ['librad_solar_flx_ratio']
    #     Product     = 'libRad_v01_montecarlo_3D_SWIA_solar_Arctic_05378D.nc'
    #     Product2    = 'libRad_v01_disort_1D_SWIA_solar_Arctic_05378D.nc'

    # elif figname == 'fig:librad_thermal_flx_ratio':
    #     librad_version = 'v01_montecarlo_disort_' 
    #     #librad_version2 =  'v01_disort_1D_' 
    #     plot_types_librad = ['librad_thermal_flx_ratio']
    #     Product     = 'libRad_v01_montecarlo_3D_SWIA_thermal_Arctic_05378D.nc'
    #     Product2    = 'libRad_v01_disort_1D_SWIA_thermal_Arctic_05378D.nc'

    # elif figname == 'fig:solar_flx_ratio':
    #     plot_types_flx_solar = ['solar_flx_ratio']
        
    #     if want_3D:
    #         librad_version = 'v01_montecarlo_3D_'
    #         Product=['libRad_v01_montecarlo_3D_SWIA_solar_Arctic_05378D.nc',
    #                  'libRad_v01_montecarlo_3D_SWIA_solar_Orbit_05458F.nc'][idx_scene]
    #     else:
    #         librad_version = 'v01_disort_1D_'
    #         Product=['libRad_v01_disort_1D_SWIA_solar_Arctic_05378D.nc',
    #                  'libRad_v01_disort_1D_SWIA_solar_Orbit_05458F.nc'][idx_scene]

    # elif figname == 'fig:thermal_flx_ratio':
    #     plot_types_flx_solar = ['thermal_flx_ratio']
        
    #     if want_3D:
    #         librad_version = 'v01_montecarlo_3D_'
    #         Product=['libRad_v01_montecarlo_3D_SWIA_thermal_Arctic_05378D.nc',
    #                  'libRad_v01_montecarlo_3D_SWIA_thermal_Orbit_05458F.nc'][idx_scene]
    #     else:
    #         librad_version = 'v01_disort_1D_'
    #         Product=['libRad_v01_disort_1D_SWIA_thermal_Arctic_05378D.nc',
    #                  'libRad_v01_disort_1D_SWIA_thermal_Orbit_05458F.nc'][idx_scene]
    


    # elif figname == 'fig:liquid_water_content':
    #     plot_types_curtain = ['liquid_water_content']
    #     SceneNames = ['Arctic_05378D']

    # elif figname == 'fig:ice_water_content':
    #     plot_types_curtain = ['ice_water_content']
    #     SceneNames = ['Arctic_05378D']

    # elif figname == 'fig:ice_effective_radius':
    #     plot_types_curtain = ['ice_effective_radius']
    #     SceneNames = ['Arctic_05378D']

    # elif figname == 'fig:aerosol_extinction':
    #     plot_types_curtain = ['aerosol_extinction']
    #     SceneNames = ['Arctic_05378D']

    # elif figname == 'fig:index_construction':
    #     plot_types_map = ['index_construction']
    #     SceneNames = ['Arctic_05378D']


    # BG: not used ---------------------------------------------------------------------------------------------
                        # elif figname == 'fig:Halifax_surface_albedo':
                        #     librad_type ='SNNN'
                        #     librad_version =  'v01_twostr_1D_' 
                        #     librad_version2 = 'v01_twostr_1D_' 
                        #     plot_types_line_solar = ['albedo_direct_radiation_surface_visible']
                        #     plot_types_flx_solar = ['solar_both']
                        #     SceneNames =['Arctic_05378D']
                        # elif figname == 'fig:Halifax_liquid_water':
                        #     librad_type ='SWNN'
                        #     librad_version =  'v01_twostr_1D_' 
                        #     librad_version2 = 'v01_twostr_1D_' 
                        #     plot_types_line_solar = ['liquid_water_column']
                        #     plot_types_flx_solar = ['solar_both']
                        #     SceneNames =  ['Halifax']
                        # elif figname == 'fig:Halifax_ice_water':
                        #     librad_type ='SWIN'
                        #     librad_version =  'v01_twostr_1D_' 
                        #     librad_version2 = 'v01_twostr_1D_' 
                        #     plot_types_line_solar = ['ice_water_column']
                        #     plot_types_flx_solar = ['solar_both']
                        #     SceneNames =  ['Halifax']
                        # elif figname == 'fig:Halifax_aerosol':
                        #     librad_type ='SWIA'
                        #     librad_version =  'v01_twostr_1D_' 
                        #     librad_version2 = 'v01_twostr_1D_' 
                        #     plot_types_line_solar = ['aerosol_column']
                        #     plot_types_flx_solar = ['solar_both']
                        #     SceneNames =  ['Halifax']
                        # elif figname == 'fig:Halifax_librad_solar_flx_ratio_v00SWIA':
                        #     librad_type ='SWIA'
                        #     librad_version = 'v01_twostr_1D_' 
                        #     librad_version2 =  'v01_disort_1D_' 
                        #     plot_types_librad = ['librad_solar_flx_ratio']
                        #     SceneNames =  ['Halifax']
                        # elif figname == 'fig:Halifax_librad_solar_flx_ratio_mysticSWIA':
                        #     librad_type ='SWIA'
                        #     librad_version = 'v01_montecarlo_1D_' 
                        #     librad_version2 =  'v01_disort_1D_' 
                        #     plot_types_librad = ['librad_solar_flx_ratio']
                        #     SceneNames =  ['Halifax']
                        # elif figname == 'fig:Halifax_librad_solar_flx_vmystic023DSWIA':
                        #     librad_type ='SWIA'
                        #     librad_version =  'v08_montecarlo_3D_' #'v05_montecarlo_3D_' # 'v04_montecarlo_3D_'
                        #     librad_version2 = 'v01_disort_1D_' #'v04_montecarlo_3D_'#
                        #     plot_types_librad = ['librad_solar_flx']
                        #     SceneNames =  ['Arctic_05378D','Baja','Halifax']    
                        #     SceneNames =  ['Baja', 'Arctic_05378D']    
                        #     SceneNames =  ['Halifax']    
                        # elif figname == 'fig:Halifax_librad_solar_flx_ratio_vpseudosphericalSWIA':
                        #     librad_type ='SWIA'
                        #     librad_version = 'v01_twostr_1D_' # 'v05_montecarlo_3D_' # 
                        #     librad_version2 = 'vpseudospherical_twostr_1D_' # 'v04_montecarlo_3D_'# 
                        #     plot_types_librad = ['librad_solar_flx_ratio']
                        #     SceneNames = ['Halifax'] # ['Baja'] #
                        # elif figname == 'fig:albedo_diffuse_radiation_surface_visible':
                        #     plot_types_map = ['albedo_diffuse_radiation_surface_visible']
                        #     SceneNames = ['Arctic_05378D','Baja','Halifax']
                        # elif figname == 'fig:albedo_diffuse_radiation_surface_near_infrared':
                        #     plot_types_map = ['albedo_diffuse_radiation_surface_near_infrared']
                        #     SceneNames = ['Arctic_05378D','Baja','Halifax']
    # ---------------------------------------------------------------------------------------------
    