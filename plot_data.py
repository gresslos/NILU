import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import glob
import ReadEarthCAREL2_bg as ReadEC

FONTSIZE=15
INFOSIZE=13
FIGSIZE=(10,5)
# default color cycle:
# colors = plt.cm.tab10.colors  
# colors = plt.cm.Set2.colors
# colors = plt.cm.Dark2.colors
colors = plt.cm.Paired.colors #  BEST

# If get more then pairs of orbits, use this: color = triplet_palette(orbits)[i] 
def triplet_palette(orbits):
    import matplotlib.pyplot as plt
    cmaps  = {'C': 'Blues', 'D': 'Greens', 'E': 'Oranges'}
    levels = [0.35, 0.60, 0.85]   # three pleasant shades
    idx = {'C':0,'D':0,'E':0,'other':0}
    colors = []
    for o in orbits:
        suf = (o or '')[-1]
        key = suf if suf in cmaps else 'other'
        cmap = plt.cm.get_cmap(cmaps.get(key, 'Greys'))
        colors.append(cmap(levels[idx[key] % 3]))
        idx[key] += 1
    return colors


########################################### IF WANT TO READ .TXT FILE ##############################################################
def plot_mean_data(filename, title, main_title=False, png_name="Data/figures/test.png"):
    versions, orbits, places, values = read_data_csv_mean(filename)

    # Generate png_name from filename
    png_name = "Data/figures/" + filename[5:-4] + ".png"

    plot(
        versions, orbits, places, values, np.full(len(values), np.nan), # Dummy stds, plot() expect 2nd array
        ylabel=r"$F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]",  #'TOA upward flux std [W/m$^2$]',
        title=title,
        show_stds=False, # Given data is mean only
        png_name=png_name
    )

def read_data_csv_mean(filename):
    """
    Reads data from a CSV-like text file where each row is:
    Orbit,Place,mc1e2,mc1e3,...

    Returns:
        versions: list of version names (from header)
        orbits: list of orbit names
        places: list of place names
        values: 2D list of values [orbit][version]
    """
    with open(filename) as file:
        lines = [l.strip() for l in file if l.strip()]

    # Header defines version names
    header = lines[0].split(",")
    versions = header[2:]  

    orbits = []
    places = []
    values = []

    for line in lines[1:]:
        parts = line.split(",")
        orbits.append(parts[0])
        places.append(parts[1])
        row_values = [float(x) for x in parts[2:]]
        values.append(row_values)

    return versions, orbits, places, values
####################################################################################################################################






def plot(versions, orbits, places, values, stds, ylabel, title, show_stds=True, png_name="Data/plot.png"):
    x = range(len(versions))
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # ------------------------------------------- Individual plot settings ----------------------------
    if '3D_buffer' in png_name:
        # Index: columns for 'solar' and 'thermal'
        isolar, itherm = 0, 1

        # Build arrays
        solar_mu = np.array([values[i][isolar] for i in range(len(orbits))], float)
        solar_sd = np.array([stds[i][isolar]  for i in range(len(orbits))], float)
        therm_mu = np.array([values[i][itherm] for i in range(len(orbits))], float)
        therm_sd = np.array([stds[i][itherm]  for i in range(len(orbits))], float)

        # Plot (matches your style)
        x  = np.arange(len(orbits))
        dx = 0.15 # 0.18

        ax.errorbar(x - dx, solar_mu, yerr=solar_sd, fmt='s', capsize=3, linewidth=2,
                    label='Solar ΔF', zorder=3, color='r')
        ax.errorbar(x + dx, therm_mu, yerr=therm_sd, fmt='s', capsize=3, linewidth=2,
                    label='Thermal ΔF', zorder=3, color='b')

        ax.axhline(0, lw=1, ls='--', alpha=0.6) # Baselien

        xticks_str = [f"{o} ({p})" for o, p in zip(orbits, places)]
        versions = xticks_str
        # ax.set_xlabel("Orbit", fontsize=INFOSIZE)

    else:
        for i, orbit in enumerate(orbits):
            label = f"{orbit} ({places[i]})"
            color = colors[i % len(colors)]
            mean = np.array(values[i])
            std = np.array(stds[i])

            ax.plot(x, mean, marker="o", color=color, label=label, linewidth=2)
            if show_stds:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    # ---------------------------------------------------------------------------------------------------

    # --------------------------------------- General plot settings --------------------------------------
    ax.set_xticks(x)
    if '3D_buffer' in what_to_plot: rotation = 20
    else:                           rotation = 0
    ax.set_xticklabels(versions, rotation=rotation, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel, fontsize=INFOSIZE)
    ax.set_title(title, fontsize=FONTSIZE, fontweight='bold')
    

    # --- Nice style adjustments ---
    ax.set_facecolor('#f0f0f0')               # background
    ax.grid(which='major', linestyle='--', alpha=0.4) # major grid lines
    ax.tick_params(axis='both', which='major', labelsize=INFOSIZE) # major ticks font size
    
    for spine in ['top', 'right']: # hide top and right spines
        ax.spines[spine].set_visible(False)
    # -------------------------------
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_name)
    print(f"Plot saved to {png_name}")

def plot_lollipop(versions, orbits, places, values, swia_values, ylabel, title, png_name="Data/plot.png"):
    values = np.asarray(values, float)
    x = np.arange(len(versions))
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # small horizontal jitter so orbits don't overlap within each component
    jitter = np.linspace(-0.35, 0.35, len(orbits)) if len(orbits) > 1 else [0.0]

    # symmetric y-limits around zero
    ymax = np.max(np.abs(values)) or 1.0
    ax.set_ylim(-1.1*ymax, 1.1*ymax)

    for i, orb in enumerate(orbits):
        xi = x + jitter[i]
        yi = values[i]
        color = colors[i % len(colors)]
        ax.vlines(xi, 0, yi, lw=2, color=color)
        lbl = f"{orb} (SWIA: {swia_values[i]:.2f}" + r" W/m$^{2}$)" if swia_values is not None else orb
        ax.plot(xi, yi, 'o', color=color, label=lbl)

    

    # --- Nice style adjustments ---
    ax.set_facecolor('#f0f0f0')               # background
    ax.grid(which='major', linestyle='--', alpha=0.4) # major grid lines
    ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*0.9) # major ticks font size
    
    for spine in ['top', 'right']: # hide top and right spines
        ax.spines[spine].set_visible(False)
    # -------------------------------

    ax.axhline(0, lw=1, ls='--', color='0.6')
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel, fontsize=INFOSIZE)
    ax.set_title(title, fontsize=FONTSIZE, fontweight='bold')
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(png_name)
    print(f"Plot saved to {png_name}")

    # deleta all this?
    # n = len(PNGs)
    
    # figsize = (10,5*n)

    # fig, axes = plt.subplots(n, 1, figsize=figsize, constrained_layout=True)
    # if n == 1:
    #     axes = [axes]

    # for ax, path in zip(axes, PNGs):
    #     ax.imshow(mpimg.imread(path))
    #     ax.axis('off')
    
    # out_path = 'Data/figures/testSubplot.png'
    # fig.savefig(out_path)
    # plt.close(fig)
    # print(f"Saved {out_path}")
    # return 

def plot_correlation(source, versions, orbits, places, values, ylabel, title_spec, show_stds=True, png_name="Data/plot.png"):
    fig = plt.figure(figsize=(7,6))
    fig.subplots_adjust(left=0.11, right=0.88,bottom=0.06)
    ax = fig.add_subplot(1,1,1)
    cmap = plt.get_cmap('viridis')  # 'viridis' 'inferno'     cmap = plt.get_cmap('jet')
    pl_list = []

    xlabel_specs = r' [W/m$^2$]'
    xlabel = 'BMA-FLX'
    ylabel = 'MYSTIC' if 'montecarlo' in librad_version else  'DISORT' 
    ylabel_specs = xlabel_specs
    source_str = 'Solar' if source == 'solar' else 'Thermal'
    title = ylabel
    title += f' - {source_str} TOA Flux'
    title += title_spec 

    x_part, y_part = [], []
    # for version, orbit, place, value in zip(versions, orbits, places, values):
    values = values[0]      # shape [[[Orbit1_libRad, Orbit1_BMAFLX], [Orbit2_libRad, Orbit2_BMAFLX], ...]] -> [[Orbit1_libRad, Orbit1_BMAFLX], [Orbit2_libRad, Orbit2_BMAFLX], ...]
    for value in values:    # value -> for a give orbit: [[Librad_pixel0, BMAFLX_pixel0][Librad_pixel1, BMAFLX_pixel1]...]
        value = np.asarray(value)
        print("Value.shape = ", value.shape) # (nr.pixels, 2)
        libRad, BMAFLX = value[:,0], value[:,1]

        x_part.append(BMAFLX)
        y_part.append(libRad)

    # Concatenate all groups to single vectors
    x = np.concatenate(x_part)
    y = np.concatenate(y_part)


    #### Calculate + Plot Stats #####
    r = np.corrcoef(x, y)[0, 1] # Correlation coefficient
    r2 = r**2 # The strength of linear association
    # Least squares fit y = a + b x
    b, a = np.polyfit(x, y, 1)
        # yhat = a + b * x
    # Errors
    diff = y - x
        # rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    mean = np.mean(diff); std = np.std(diff)

    # Ranges for plotting
    xy_min = np.nanmin([x.min(), y.min()])
    xy_max = np.nanmax([x.max(), y.max()])
    pad = 0.03 * (xy_max - xy_min if xy_max > xy_min else 1.0)
    lo, hi = xy_min - pad, xy_max + pad

    # For setting ax.set_ylim/xlim
    ymin, xmin, ymax, xmax = xy_min, xy_min, xy_max, xy_max

    # y=x line
    p, = ax.plot([lo, hi], [lo, hi], lw=1.2, c='r', linestyle="--", label="y = x")
    pl_list.append(p)

    # Regression line over same span
    xx = np.linspace(lo, hi, 100)
    p, = ax.plot(xx, a + b * xx, lw=1.5, c='r', label=f"y = {a:.2f} + {b:.2f}x") # WRITE IN FIG-TEXT: Linear Least Squares (fit)") 
    pl_list.append(p)

    data_str = (
        # f"n = {x.size}\n"
        f"R² = {r2:.2f}                     \n"
        # f"RMSE = {rmse:.2f}\n"
        r"$⟨\Delta F⟩$ (Bias) = " + f"{mean:.1f} ± {std:.1f}" + r" W/m$^2$"  # _{\mathrm{TOA}}^{\uparrow}
        # f"\nMAE = {mae:.2f}\n"
    )
    ax.text(.99, .01, data_str,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONTSIZE*.7, color='k',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='w', alpha=0.8))


    # PLOT CORRELATION PLOT 
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

    ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)

    # Set plotting limits
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin, xmax)

    ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
    ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98, fontweight='bold')
    # remove the "Orbit_" prefix and join nicely
    orbit_names = [o.replace("Orbit_", "") for o in orbits]
    ax.set_title(f"Orbits: {', '.join(orbit_names)}", fontsize=INFOSIZE*.8)

  
    ax.legend(handles=pl_list,
                loc='upper left', framealpha=0.7, 
                borderaxespad=0.0,                   # space to axes
                borderpad=0.25, labelspacing=0.25,   # compact box)
                fontsize=INFOSIZE*.8)

    fig.tight_layout()
    ax.set_facecolor('#f0f0f0')  # Axes background (warm light grey)
    # Grid: major dashed, minor dotted
    ax.grid(which='major', linestyle='--', alpha=0.4)
    ax.grid(which='minor', linestyle=':',  alpha=0.2)
    ax.minorticks_on()
    # remove top/right border
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)

    plt.savefig(png_name)
    plt.close()
    print(f"Plot saved to {png_name}")

    return



################################################################################################################
def get_instances(SceneName, Product, Product2=False, BMAFLX=False):
    # Get .nc file
    ProductPath = ProductPathRTM
    ProductFile = os.path.join(ProductPath, Product)
    print('ProductFile1', ProductFile)
    ProductFile = sorted(glob.glob(ProductFile))[0]  
    libRad = ReadEC.Scene(Name=SceneName)
    libRad.ReadEarthCAREh5(ProductFile)
    libRad.SetExtent()
    
    # BG: Find; Baseline, Data, Time
    BB_baseline, BA_baseline, AC_baseline = '', '', ''
        
    # Get ACM-COM 
    Product ='ACM_COM'
    ProductPath = '*'+Product+'*'
    ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
    ProductFile = sorted(glob.glob(ProductFile))[0]
    ACMCOM = ReadEC.Scene(Name=SceneName)
    ACMCOM.ReadEarthCAREh5(ProductFile) #, ACM3D=ACM3D)
    ACMCOM.SetExtent()
    # Extract Baseline ---------------------------   
    parts = ProductFile.split("ECA_EX", 1)
    out = parts[1][:2] if len(parts) > 1 else None
    if out == 'BA': BA_baseline += ' ' + Product
    elif out == 'AC': AC_baseline += ' ' + Product
    elif out == 'BB': BB_baseline += ' ' + Product
    # print(out)
    #---------------------------------------------

    if Product2:      
        ProductPath = ProductPathRTM
        ProductFile = os.path.join(ProductPath, Product2)
        # print('ProductFile2', ProductFile)
        ProductFile = sorted(glob.glob(ProductFile))[0]
        libRad2 = ReadEC.Scene(Name=SceneName)
        libRad2.ReadEarthCAREh5(ProductFile)
        libRad2.SetExtent()
    else: libRad2 = False

    # Get BMA-FLX 
    if BMAFLX:
        Product ='BMA_FLX'
        ProductPath = '*'+Product+'*'  
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        BMAFLX = ReadEC.Scene(Name=SceneName)
        BMAFLX.ReadEarthCAREh5(ProductFile, Resolution='StandardResolution')
        BMAFLX.SetExtent()
        # Extract Baseline ---------------------------   
        parts = ProductFile.split("ECA_EX", 1)
        out = parts[1][:2] if len(parts) > 1 else None
        if out == 'BA': BA_baseline += ' ' + Product
        elif out == 'AC': AC_baseline += ' ' + Product
        elif out == 'BB': BB_baseline += ' ' + Product
        # print(out)
        #---------------------------------------------
    
     # Fix Baseline
    AC_baseline = ", ".join(AC_baseline.split())
    BA_baseline = ", ".join(BA_baseline.split())
    baseline_str = "Baselines: "
    if AC_baseline != '':   baseline_str += f'AC = ({AC_baseline})  '
    if BA_baseline != '':   baseline_str += f'BA = ({BA_baseline})  '
    if BB_baseline != '':   baseline_str += f'BB = ({BB_baseline})  '
        


    # Extract Date
    date_num = '20' + ProductFile.split('20', 1)[1].split('T', 1)[0]
    date = date_num[6:8] + "." + date_num[4:6] + "." + date_num[0:4]

    # Extract Time
    first_time_num = ProductFile.split('T')[1].split('Z')[0]
    second_time_num = ProductFile.split('T')[2].split('Z')[0] 
    start_time = first_time_num[:2] + ":" + first_time_num[2:4]
    end_time = second_time_num[:2] + ":" + second_time_num[2:4]
    time = start_time + '-' + end_time

    info = (date, time, baseline_str)

    return libRad, BMAFLX, ACMCOM, libRad2, info







def get_data(source, spec, data, data_row, stds, stds_row, libRad, libRad2, ACMCOM, BMAFLX, metric='diff', statistics='mean'): 
    attr_by_source_libRad = {
        'solar':   'solar_eup',
        'thermal': 'thermal_eup',
    }
    attr_by_source_BMAFLX = {
        'solar':   'solar_combined_top_of_atmosphere_flux',
        'thermal': 'thermal_combined_top_of_atmosphere_flux',
    }
    attr_libRad = attr_by_source_libRad.get(source)
    attr_BMAFLX = attr_by_source_BMAFLX.get(source)

    x1 = libRad.latitude
    data1 = getattr(libRad, attr_libRad)  # dynamic attribute access -> libRad, Solar or Thermal
    # ----------- Quality-Status (ACMCOM) -----------------
    quality = ACMCOM.quality_status[:]
    quality_mask_ACMCOM = np.isin(quality, [0, 1])
    quality_mask_libRad = quality_mask_ACMCOM & (data1 != 0)
    
    # x1 = x1[quality_mask_ACMCOM]
    # data1 = data1[quality_mask_ACMCOM]
    x1 = x1[quality_mask_libRad]
    data1 = data1[quality_mask_libRad]


    # print(f'{spec:10} {source:10} flux: min={np.min(data1):.2f}, mean={np.mean(data1):.2f}, max={np.max(data1):.2f}\n')
    
    # ------------------------------------------------

    if BMAFLX:
        x2 = BMAFLX.latitude
        data2 = getattr(BMAFLX, attr_BMAFLX)  # dynamic attribute access

        # ----------- Quality-Status (BMAFLX) -----------------
        quality = BMAFLX.quality_status[:]
        # boolean mask: True where quality is
        #   0: solar and thermal OK
        #   1: Thermal OK
        #   2: Solar OK
        source_quality_status_idx = 1 if source == 'thermal' else 2
        quality_mask = np.isin(quality, [0, source_quality_status_idx]) 

        data2 = data2[quality_mask]
        x2 = x2[quality_mask]
        # ------------------------------------------------

        # Performe interpolation for comparison
        fill_value_data1 = 9.96921e+36
        fill_value_data1 = 0
        x, data2_interp, data1 = ReadEC.interpolate(x2, data2, fill_value_data1, x1, data1, fill_value_data1) 
        data2 = data2_interp

    elif libRad2:
        x2 = libRad2.latitude
        data2 = getattr(libRad2, attr_libRad)  # dynamic attribute access 

        # ----------- Quality-Status (ACMCOM) -----------------
        quality_mask_libRad2 = quality_mask_ACMCOM & (data2 != 0)
        x2 = x2[quality_mask_libRad2]
        data2 = data2[quality_mask_libRad2]
        
        # ------------------------------------------------

        x = x1
    else: print('Error: select BMAFLX or libRad2')



    if metric == 'diff': # Calculate differance  
        value = (data1 - data2) 
    elif metric == 'only_libRad': # Calculate only libRad
        value = data1


    
    if statistics == 'mean': # Calculate Mean
        data_value = np.nanmean(value)
    elif statistics == 'min': # Calculate Min
        data_value = np.nanmin(value)
    elif statistics == 'max': # Calculate Max
        data_value = np.nanmax(value)
    elif statistics == 'all_pixels_values':
        data_value = np.column_stack([value, data2]) # Returns [[libRad, BMA], ...]
        # data_value = np.vstack(data_value) # shape: (N, 2) -> [:,0]=libRad, [:,1]=BMA-FLX

    std  = np.nanstd(value)
    # print(f'{source:8}:  <diff> = {data_value:6.2f} pm {std:6.2f} W/m2  (n = {len(value)})')

    global old_spec
    if not globals().get("old_spec", False):
        old_spec = spec          # <-- set it on first hit
    if spec != old_spec:
        data.append(data_row.copy())
        stds.append(stds_row.copy())
        # IMPORTANT: start fresh row
        data_row.clear()       
        stds_row.clear()

    # data_row.append(float(data_value))  
    # stds_row.append(float(std))
    data_row.append(data_value)
    stds_row.append(std)
    
    old_spec = spec








def loop_through_data(source, Product2, SceneNames, librad_type='SWIA', statistics='mean'):
    if Product2 == True: BMAFLX = False
    else:                BMAFLX = True
    SceneNames = [SceneNames[i][0] for i in idx_scene]

    # --Create list of places:---
    special = {
        'Orbit_06888C': 'Svalbard',
        'Orbit_07277C': 'Svalbard',
        'Orbit_06331C': 'Greenland',
    }
    by_suffix = {'C': 'Svalbard', 'D': 'USA', 'E': 'Africa'}
    places = [ special.get(orbit, by_suffix.get(orbit[-1], 'Unknown')) for orbit in SceneNames ]
    # ----------------------------

    global old_spec
    old_spec = None
    data, data_row, stds, stds_row = [], [], [], []
    # SWIA-studies:
    if isinstance(librad_type, list) and isinstance(additional_spesifications, list): 
        for l_type, spec in zip(librad_type, additional_spesifications):
            for SceneName in SceneNames:
                Product = (
                        'libRad_' + 
                        version_identifier  + '_' +
                        librad_version      + '_' +
                        l_type              + '_' +
                        source_str          + '_' +
                        SceneName           + 
                        spec                + '.nc'
                    )

                libRad, BMAFLX, ACMCOM, libRad2, _ = get_instances(SceneName, Product, Product2, BMAFLX)
                get_data(source, l_type, data, data_row, stds, stds_row, libRad, libRad2, ACMCOM, BMAFLX, metric='only_libRad', statistics=statistics) 
                # Note, changed spec -> l_type to let know how to construct data-array

    # Other studies:
    elif isinstance(additional_spesifications, list): # Usually the case
        for spec in additional_spesifications:
            for SceneName in SceneNames:
                Product = (
                        'libRad_' + 
                        version_identifier  + '_' +
                        librad_version      + '_' +
                        librad_type         + '_' +
                        source_str          + '_' +
                        SceneName           + 
                        spec                + '.nc'
                    )
                if Product2: 
                        librad_version2 = 'montecarlo_3D'
                        # additional_spesifications2 = additional_spesifications2
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

                # print(f'ScneName: {SceneName}:')
                libRad, BMAFLX, ACMCOM, libRad2, _ = get_instances(SceneName, Product, Product2, BMAFLX)
                if what_to_plot == 'correlation':   metric = 'only_libRad'
                else:                               metric = 'diff'
                get_data(source, spec, data, data_row, stds, stds_row, libRad, libRad2, ACMCOM, BMAFLX, metric=metric, statistics=statistics)

   
    
    # --- flush the last row ---
    if data_row and stds_row:
        data.append(data_row.copy()); data_row.clear()
        stds.append(stds_row.copy()); stds_row.clear()   

    Scenenames, places = np.asarray(SceneNames), np.asarray(places)
    if 'correlation' not in what_to_plot:
        data, stds = np.asarray(data), np.asarray(stds)
        data, stds = np.transpose(data), np.transpose(stds)

    return SceneNames, places, data, stds





def plot_traj_on_globe(SceneNames, vmin=-4, vmax=1.0, cmap_name="coolwarm", title="Title", cblabel="Value"):
    fig = plt.figure(figsize=FIGSIZE)
    cmap = plt.get_cmap(cmap_name)
    projection=ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    # Map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-')
        # ax.add_feature(cfeature.LAKES, alpha=0.5)
        # ax.add_feature(cfeature.RIVERS)
    gl=ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels=False

    Product2, BMAFLX = False, False
    glob_min_lat, glob_max_lat, glob_min_lon, glob_max_lon = np.nan, np.nan, np.nan, np.nan
  
    label_groups = []  # store representative positions of each plotted group
    LABEL_Y_OFFSET = 10.0  # degrees to move labels downward for stacked labels
    OVERLAP_DX = 3.0       # longitude threshold for "too close"
    OVERLAP_DY = 3.0       # latitude threshold for "too close"
    for spec in additional_spesifications:
        for SceneName in SceneNames:
            Product = (
                    'libRad_' + 
                    version_identifier  + '_' +
                    librad_version      + '_' +
                    librad_type         + '_' +
                    source_str          + '_' +
                    SceneName           + 
                    spec                + '.nc'
                )

            _, _, ACMCOM, _ = get_instances(SceneName, Product, Product2, BMAFLX)

            x = ACMCOM.longitude; y = ACMCOM.latitude; data = ACMCOM.surface_temperature * 0
            # update local extents
            min_lon, max_lon = np.nanmin(x), np.nanmax(x)
            min_lat, max_lat = np.nanmin(y), np.nanmax(y)
            # update global extents
            if np.isnan(glob_min_lat) or min_lat < glob_min_lat: glob_min_lat = min_lat
            if np.isnan(glob_max_lat) or max_lat > glob_max_lat: glob_max_lat = max_lat
            if np.isnan(glob_min_lon) or min_lon < glob_min_lon: glob_min_lon = min_lon
            if np.isnan(glob_max_lon) or max_lon > glob_max_lon: glob_max_lon = max_lon
            

            # Trajectory plot specs
            SceneName = SceneName.replace("Orbit_", "")

            center_lon = 0.5 * (min_lon + max_lon)
            center_lat = max_lat

            # Check if close to an existing group
            group_idx = None
            for i, g in enumerate(label_groups):
                if (abs(center_lon - g["lon"]) < OVERLAP_DX and abs(center_lat - g["lat"]) < OVERLAP_DY):
                    group_idx = i
                    break

            # Decide if to plot full trajectory
            if group_idx is None:
                # pcolormesh for each swath
                cs = ax.pcolormesh(
                    x, y, data,
                    transform=ccrs.PlateCarree(),
                    shading='auto',
                    cmap=cmap,
                    vmin=vmin, vmax=vmax
                )
                idx_nadir = 151
                ax.plot(x[idx_nadir, :], y[idx_nadir, :],
                        color="b", linewidth=0.6, alpha=0.9)
                ax.text(center_lon, center_lat, SceneName,
                    transform=ccrs.PlateCarree(),
                    fontsize=INFOSIZE,
                    fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15',facecolor='white',edgecolor='none', alpha=0.8)
                    )
                # Update/add to the group for not future plotting of similar swaths
                label_groups.append({"lon": center_lon, "lat": center_lat, "n_labels": 1,})
            else: 
                # Too close: do NOT plot swath or line
                # Only add a label stacked underneath the first one.
                g = label_groups[group_idx]
                g["n_labels"] += 1

                label_lat = g["lat"] - LABEL_Y_OFFSET * (g["n_labels"] - 1)

                ax.text(
                    g["lon"], label_lat, SceneName,
                    transform=ccrs.PlateCarree(),
                    fontsize=INFOSIZE,
                    fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',edgecolor='none', alpha=0.8)
                )

    

            # # Colorbar
            # cb = fig.colorbar(
            #     cs, ax=ax,
            #     shrink=1.0, pad=0.02,
            #     extend='both',
            #     extendrect=True
            # )
            # cb.set_label(cblabel, size=INFOSIZE)
            # cb.ax.tick_params(labelsize=INFOSIZE * 0.8)
            # cb.outline.set_visible(False)
            # cb.ax.set_facecolor('#f7f7f7')

    # Under/bad colors for cleaner look
    cs.cmap.set_under('#f0f0f0')
    cs.cmap.set_bad('#f0f0f0')

    # Axis labels + limits = "2D globe"
    pad = 5 # Latitude/Longitude
    ax.set_xlim(glob_min_lon - pad, glob_max_lon + pad)
    ax.set_ylim(glob_min_lat - pad, glob_max_lat + pad)
    # ax.set_global()

   
    ax.set_xlabel(r'Longitude [N$^\circ$]', fontsize=INFOSIZE)
    ax.set_ylabel(r'Latitude [N$^\circ$]', fontsize=INFOSIZE)
        # ax.xaxis.set_major_locator(MultipleLocator(2.0))  # one tick every 2°
        # ax.xaxis.set_minor_locator(MultipleLocator(2.0))
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
    ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98, fontweight='bold')  
   
    # Background, grid, spines
    ax.set_facecolor('#f0f0f0')
    ax.grid(which='major', linestyle='--', alpha=0.4)
    ax.grid(which='minor', linestyle=':',  alpha=0.2)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # ax.legend()
    ax.set_aspect('auto')  # let x/y stretch freely
    # [left, bottom, width, height] in figure fraction coordinates
    # ax.set_position([0.02, 0.05, 0.96, 0.88])
    # fig.tight_layout()
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.92)   # then put the suptitle inside that
    # fig.suptitle(title, fontsize=FONTSIZE, fontweight='bold')
    # fig.savefig(png_name)
    fig.savefig(png_name, bbox_inches="tight")
    print(f"Plot saved to {png_name}")




def plot_multi_libRad(libRad_list, additional_spesifications, BMAFLX, ACMCOM, plot_type, pngfile, info, quantity_list):
    x = BMAFLX.latitude
    cmap = plt.get_cmap('viridis')  # 'viridis' 'inferno'  
    colors = plt.cm.Set2.colors
    # colors = plt.cm.Dark2.colors    
    # colors = plt.cm.tab10.colors  
    xmin = BMAFLX.latitude.min() #-90 #
    xmax = BMAFLX.latitude.max() #90 #
    pl_list = []
    xlabel = "Latitude"; xlabel_specs = r" [N$^\circ$]"
    ylabel= r"$F_{\mathrm{TOA}}^{\uparrow}$"; ylabel_specs = r" [W/m$^2$]"
    date, time, baseline_str = info
    stacked = True


    
    
    if 'flx_diff' in plot_type: 
        fig = plt.figure(figsize=(10,4))
        gs = fig.add_gridspec(1, 2, width_ratios=[3,1]) 
        ax = fig.add_subplot(gs[0, 0])
    elif 'correlation' in plot_type: 
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
                # gs = fig.add_gridspec(1, 2, width_ratios=[3,1]) 
                # ax = fig.add_subplot(gs[0, 0])
    else:                       
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1,1,1)

    ################################ Differnt Plotting-Codes ###############################
    if plot_type == 'solar_both': 
        title='Solar TOA flux' #+ f' - Atmosphere {atmosphere} (ACM-COM)'
        cblabel='BMA_FLX' #: solar_combined_top_of_atmosphere_flux'
                
        # ---------- BG: Get additional data to plot (twin or stacked-axis) ----------------
        if quantity_list:
            add_profile_list = [] # List for ax2-legends
            for quantity in quantity_list:
                ax, ax2 = ReadEC.add_profile_to_plot(fig, ax, ACMCOM, fsize=FONTSIZE, legend_list=add_profile_list, quantity=quantity, quantity_list=quantity_list, stacked=stacked)
            if add_profile_list: ax2.legend(handles=add_profile_list, 
                                            loc='upper right', framealpha=0.7, 
                                            borderaxespad=0.0,                  # space to axes
                                            borderpad=0.25, labelspacing=0.25,   # compact box)
                                            fontsize=INFOSIZE*.8)
        # ------------------------------------------------------------------------

        x = BMAFLX.latitude 
        data = BMAFLX.solar_combined_top_of_atmosphere_flux
        # ----------- Quality-Status -----------------
        if want_quality_status: 
            quality = BMAFLX.quality_status[:]
            # boolean mask: True where quality is 0 or 2
            quality_mask = np.isin(quality, [0, 2]) 

            # Apply the mask to data
            x[~quality_mask], data[~quality_mask] = np.nan, np.nan
        else: 
            mask = (data < 1e30) & (data > 0); x[~mask] = np.nan; data[~mask] = np.nan
        # ------------------------------------------------
       
        for i, (libRad, spec) in enumerate(zip(libRad_list, additional_spesifications)):
            color = colors[i % len(colors)]

            x2 = libRad.latitude
            data2 = libRad.solar_eup
            # ----------- Quality-Status -----------------
            if want_quality_status: 
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

            RTM_str = 'MYSTIC (3D)' if '3D' in librad_version else 'DISORT (1D)'
            if         'MCIPA' in spec: RTM_str += ' - MCIPA'
            elif 'SmallBuffer' in spec: RTM_str += ' - SmallBuffer'
            elif 'FullBuffer'  in spec: RTM_str += ' - LargeBuffer'
            elif 'MegaBuffer'  in spec: RTM_str += ' - MegaBuffer'
            elif 'GigaBuffer'  in spec: RTM_str += ' - GigaBuffer'
            p,=ax.plot(x2, data2, color=color, label=RTM_str,
                        # marker=".", markersize = 4, linestyle="None", 
                        alpha=.9,
                        linewidth=1,
                        zorder=i)
            pl_list.append(p)

                                    # if Scene3 != None: 
                                    #     x3 = Scene3.latitude_active
                                    #     itoa=0
                                    #     data3 = Scene3.flux_up_solar_1d_all_sky[:,itoa]
                                    #     data4 = Scene3.flux_up_solar_3d_all_sky[:,itoa]

                                    #     # ----------- Quality-Status -----------------
                                    #     if want_quality_status: 
                                    #         quality = Scene3.quality_status[0,:]
                                    #         # boolean mask: True where quality is 0 or 1
                                    #         quality_mask = np.isin(quality, [0, 1]) 

                                    #         data3 = data3[quality_mask]
                                    #         data4 = data4[quality_mask]
                                    #         x3 = x3[quality_mask]
                                    #     # ------------------------------------------------

                                    #     p,=ax.plot(x3, data3, color='gray', label='ACM_RT 1D', linewidth=2) #, alpha=.8) "flux_up_solar_1d_all_sky, TOA"
                                    #     pl_list.append(p)

                                    #     p,=ax.plot(x3, data4, 
                                    #             color = 'black', marker = 'o', markersize = 3, linestyle = 'None', 
                                    #             # color='black', marker = 'x', markersize=8, linestyle="None", 
                                    #             zorder=8, 
                                    #             label='ACM_RT 3D') # "flux_up_solar_3d_all_sky, TOA"
                                    #     pl_list.append(p) 
        

        vmin=data.min()
        vmax=data.max()
        ymin =0
        ymax = 1.3*np.nanmax(data) 

        ax.set_xlim(np.nanmin(x2),np.nanmax(x2)) # x2 -> libRad-data
        l, = ax.plot(x, data, 
                            label=cblabel, 
                            color='r',
                            zorder=10
                            # marker=".", markersize = 4, linestyle="None"
                            ) # cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0)
        leg = ax.legend(handles=[l]+pl_list,
                    # loc='upper left', 
                    framealpha=0.7, 
                    borderaxespad=0.0,                  # space to axes
                    borderpad=0.25, labelspacing=0.25,   # compact box)
                    fontsize=INFOSIZE*.8)
        leg.set_zorder(99) # put in front
      
    #########################################################################################################################################
    if plot_type == 'solar_flx_diff':
        baseline = 0
        padding = 0.1
        title = 'Solar TOA flux difference'  
        if 'Buffer' in libRad_list[0].fn: title +=  ' - 3D Buffer Size Sensitivity'
        title += ' - [MYSTIC - BMAFLX]'
        
        # ---------- BG: Get additional data to plot (twin or stacked-axis) ----------------
        if quantity_list:
            add_profile_list = [] if stacked else pl_list # List for ax2-legends
            for quantity in quantity_list:
                ax, ax2 = ReadEC.add_profile_to_plot(fig, ax, ACMCOM, fsize=FONTSIZE, legend_list=add_profile_list, quantity=quantity, quantity_list=quantity_list, stacked=stacked)
            if stacked and add_profile_list: ax2.legend(handles=add_profile_list, 
                                                        loc='upper right', framealpha=0.7, 
                                                        borderaxespad=0.0,                  # space to axes
                                                        borderpad=0.25, labelspacing=0.25,   # compact box)
                                                        fontsize=INFOSIZE*.8)
        # ------------------------------------------------------------------------

        x1 = BMAFLX.latitude
        data1 = BMAFLX.solar_combined_top_of_atmosphere_flux
        # ----------- Quality-Status -----------------
        if want_quality_status: 
            quality = BMAFLX.quality_status[:]
            # boolean mask: True where quality is 0 or 2
            quality_mask = np.isin(quality, [0, 2]) 

            data1 = data1[quality_mask]
            x1 = x1[quality_mask]
        else: 
            mask = (data1 < 1e30) & (data1 > 0); x1[~mask] = np.nan; data1[~mask] = np.nan
        # -------------------------------------------
       
       
        for i, (libRad, spec) in enumerate(zip(libRad_list, additional_spesifications)):
            color = colors[i % len(colors)]
            x2 = libRad.latitude
            data2 = libRad.solar_eup
            # ----------- Quality-Status -----------------
            if want_quality_status: 
                quality = ACMCOM.quality_status[:]
                quality_mask = np.isin(quality, [0, 1]) & (data2 != 0)
            
                x2 = x2[quality_mask]
                data2 = data2[quality_mask]
            # ------------------------------------------------


            # IF only selected area
            if modify_xlim: 
                lat_min, lat_max = lat_ranges   
                mask1 = (x1 >= lat_min) & (x1 <= lat_max) # mask for this latitude band
                mask2 = (x2 >= lat_min) & (x2 <= lat_max) # mask for this latitude band
                x1, data1, x2, data2 = x1[mask1], data1[mask1], x2[mask2], data2[mask2]
                
                

            # Performe interpolation for comparison
            fill_value_data1 = 9.96921e+36
            fill_value_data2 = 0
            x, data1_interp, data2 = ReadEC.interpolate(x1, data1, fill_value_data1, x2, data2, fill_value_data2) 

            # Calculate ratio
            data = (data2 - data1_interp) #/ data1_interp

            RTM_str = 'MYSTIC (3D)' if '3D' in librad_version else 'DISORT (1D)'
            if         'MCIPA' in spec: RTM_str += ' - MCIPA'
            elif 'SmallBuffer' in spec: RTM_str += ' - SmallBuffer'
            elif 'FullBuffer'  in spec: RTM_str += ' - LargeBuffer'
            elif 'MegaBuffer'  in spec: RTM_str += ' - MegaBuffer'
            elif 'GigaBuffer'  in spec: RTM_str += ' - GigaBuffer'

            # Add mean of the relative‑error to plot
            mean_err = np.nanmean(data)
            std  = np.nanstd(data)
            data_str = fr"{RTM_str+'\n'} ⟨$\Delta F$⟩ = {mean_err:.1f} ± {std:.1f}" + r" W/m$^2$"
            print('-----------------------------------------------\n'
                    'solar: ', data_str, '    (n = ', len(data), ')\n'
                    '-----------------------------------------------')

            if quantity_list is None: 
                y_pos = .7-(.8/len(libRad_list))*i
                fsize = FONTSIZE*.8
            else: 
                y_pos = .83-.07*i
                fsize = FONTSIZE*.7

            fig.text(.99,y_pos, data_str,
                    ha='right', va='bottom',
                    fontsize=fsize, color='k',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6))
                
            p, = ax.plot(x, data, label=RTM_str, color=color, linewidth=1.4, alpha=0.8, zorder=i
                        # marker=".", linestyle="None", markersize = 4
                        ) 
            pl_list.append(p)
            # ax.vlines(x, baseline, data, color=color, alpha=0.6, lw=0.2)  # lines to baseline
        ax.axhline(0.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (y=0)')
        # Calculate the maximum absolute deviation from the baseline (y=1)
        max_deviation = np.max(np.abs(data - baseline)) 
        ymin = baseline - max_deviation - padding 
        ymax = baseline + max_deviation + padding
        ylabel = r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$"

        ax.legend(handles=pl_list,
                    # loc='upper left', 
                    framealpha=0.7, 
                    borderaxespad=0.0,                  # space to axes
                    borderpad=0.25, labelspacing=0.25,   # compact box)
                    fontsize=INFOSIZE*.8)
    #########################################################################################################################################
    if plot_type == 'correlation_plot':
        title = 'Solar TOA flux difference'  
        if 'Buffer' in libRad_list[0].fn: title +=  ' - 3D Buffer Size Sensitivity'
        
        xlabel_specs = r' [W/m$^2$]'
        xlabel = 'BMA-FLX'
        data1 = BMAFLX.solar_combined_top_of_atmosphere_flux # BMA-FLX
        x1 = BMAFLX.latitude 
        # ----------- Quality-Status -----------------
        if want_quality_status: 
            quality = BMAFLX.quality_status[:]
            # boolean mask: True where quality is 0 or 2
            quality_mask = np.isin(quality, [0, 2]) 

            data1 = data1[quality_mask]
            x1 = x1[quality_mask]
        else: 
            mask = (data1 < 1e30) & (data1 > 0); x1[~mask] = np.nan; data1[~mask] = np.nan
        # ------------------------------------------------

        for i, (libRad, spec) in enumerate(zip(libRad_list, additional_spesifications)):
            color = colors[i % len(colors)]
            ylabel = 'libRadtran' if '3D' not in libRad.fn else 'MYSTIC'
           
            data2 = libRad.solar_eup 
            x2 = libRad.latitude
            # ----------- Quality-Status -----------------
            if want_quality_status: 
                # Create a combined mask: non-zero and quality 0 or 1
                quality = ACMCOM.quality_status[:]
                mask = (data2 != 0) & np.isin(quality, [0, 1])

                data2 = data2[mask]
                x2 = x2[mask]
            # ------------------------------------------------

            # IF only selected area
            if modify_xlim: 
                lat_min, lat_max = lat_ranges   
                mask1 = (x1 >= lat_min) & (x1 <= lat_max) # mask for this latitude band
                mask2 = (x2 >= lat_min) & (x2 <= lat_max) # mask for this latitude band
                x1, data1, x2, data2 = x1[mask1], data1[mask1], x2[mask2], data2[mask2]

            RTM_str = 'MYSTIC (3D)' if '3D' in libRad.fn else 'DISORT (1D)'
            if         'MCIPA' in spec: RTM_str += ' - MCIPA'
            elif 'SmallBuffer' in spec: RTM_str += ' - SmallBuffer'
            elif 'FullBuffer'  in spec: RTM_str += ' - LargeBuffer'
            elif 'MegaBuffer'  in spec: RTM_str += ' - MegaBuffer'
            elif 'GigaBuffer'  in spec: RTM_str += ' - GigaBuffer'

            # Performe interpolation for comparison
            fill_value_data1 = 9.96921e+36
            fill_value_data2 = 0
            not_used, data1_interp, data2 = ReadEC.interpolate(x1, data1, fill_value_data1, x2, data2, fill_value_data2) 

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
            mean = np.mean(diff); std = np.std(diff)

            # Ranges for plotting
            xy_min = np.nanmin([x.min(), y.min()])
            xy_max = np.nanmax([x.max(), y.max()])
            pad = 0.03 * (xy_max - xy_min if xy_max > xy_min else 1.0)
            lo, hi = xy_min - pad, xy_max + pad

            sc = ax.scatter(x, y, s=4, alpha=.6, color=color, label=RTM_str)
            # pl_list.append(sc)

            # Regression line over same span
            xx = np.linspace(lo, hi, 100)
            # p, = ax.plot(xx, a + b * xx, lw=1.5, c=color, label=f"{RTM_str}\n   y = {a:.2f} + {b:.2f}x") # WRITE IN FIG-TEXT: Linear Least Squares (fit)") 
            p, = ax.plot(xx, a + b * xx, lw=1.5, c=color, label=f"y = {a:.2f} + {b:.2f}x") # WRITE IN FIG-TEXT: Linear Least Squares (fit)") 
            pl_list.append(p)

            data_str = (
                RTM_str + '      \n'
                # f"n = {x.size}\n"
                f"R² = {r2:.2f}                       \n"
                # f"RMSE = {rmse:.2f}\n"
                r"$⟨\Delta F⟩$ (Bias) = " + f"{mean:.1f} ± {std:.1f} " + r" W/m$^2$"  # _{\mathrm{TOA}}^{\uparrow}
                # f"\nMAE = {mae:.2f}\n"
            )
            fsize = FONTSIZE*.6 if len(libRad_list) > 3 else FONTSIZE*.7
            y_diff = 0.09*i if len(libRad_list) > 3 else 0.1*i
            fig.text(.99, .07*len(libRad_list)-y_diff, data_str,
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fsize, color='k',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6))

       
           

        

        # y=x line
        p, = ax.plot([lo, hi], [lo, hi], lw=1.2, c='r', linestyle="--", label="y = x")
        pl_list.append(p)
        
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.legend(handles=pl_list,
                    loc='upper left', framealpha=0.7, 
                    borderaxespad=0.0,                   # space to axes
                    borderpad=0.25, labelspacing=0.25,   # compact box)
                    fontsize=INFOSIZE*.8)


  
        




    ########################################################################################
    ################## MODIFICATION FOR LATITUDE-RANGE #################
    if modify_xlim and plot_type != 'correlation_plot': 
        lat_min, lat_max = lat_ranges
        
        mask = (x >= lat_min) & (x <= lat_max) # mask for this latitude band
        pad = 10
        data_max = np.nanmax(data[mask]) + pad
        data_min = np.nanmin(data[mask]) - pad
        if plot_type == 'solar_both' or plot_type == 'thermal_both':
            data_min = 0; data_max += 40 # To account for 1D RTM large variation

        ax.set_xlim(lat_ranges)
        ax.set_ylim(data_min, data_max+10)
    ####################################################################

    ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
    ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98, fontweight='bold')

    ax.set_title(f"{libRad_list[0].Name} - {date}  {time} (UTC)", fontsize=INFOSIZE)
    plt.figtext(0.001, 0.004, baseline_str, fontsize=FONTSIZE*0.45)
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
    plt.savefig(pngfile)
    plt.close()
            
    return




###################################################################################################################################
if __name__ == "__main__":
    # ------ Settings ------
    # show_stds = True 
    show_stds = False
    # ----------------------
    

    ################################### NOT CHANGE #################################################################
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

                  ['Orbit_06331C'],#11          # Greenland (09.07.2025)       

                  ['Orbit_07883D'] #12          # Norway (FILEFJELL) 
                  ]

    ProductPathRTM              = './RESULTS/'
    pathL2TestProducts          = '/xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/'  
    version_identifier          = 'v01'
    librad_type                 = 'SWIA'
    source_str                  = 'solar,thermal'
    statistics                  = 'mean'                                            # chose from: {'mean', 'min', 'max'}



    ######################### SELECT WHAT TO PLOT ######################################################################




    ##########################
    #### Select verrsion: ####
    plot_version_idx = 5
    plot_list = [
    'Ice_habit',    #0
    'thermal_AOD',  #1
    'mc_photons',   #2
    'SWIA',         #3
    '3D_buffer',    #4
    'correlation',  #5
    'plot_traj',    #6
    'multi_libRad'  #7
    ]
    what_to_plot = plot_list[plot_version_idx]



    #######################################################################################################################
    if what_to_plot == 'Ice_habit':
        Product2  = False
        show_stds = False
        idx_scene = [3,4,5,6,7,8]
        versions  = ['GHM', 'SC', 'RA']
        sources   = ['solar', 'thermal']
        librad_version = 'disort_1D'

        additional_spesifications  = ['_GHM', '_SC', '_RA']

        titles      =  ["DISORT - Ice-Habit Sensitivity\nSolar TOA Flux Difference to BMA-FLX (mean values)",
                        "DISORT - Ice-Habit Sensitivity\nThermal TOA Flux Difference to BMA-FLX (mean values)"]
        # [r"Ice habit - Solar $(F_{\mathrm{TOA}}^{\mathrm{DISORT, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value",
        #  r"Ice habit - Thermal $(F_{\mathrm{TOA}}^{\mathrm{DISORT, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value"]
        png_names   =  ["Data/figures/Ice_Habit_all_points_solar.png",
                        "Data/figures/Ice_Habit_all_points_thermal.png"]


    #######################################################################################################################
    elif what_to_plot == 'thermal_AOD':
        Product2  = False
        show_stds = False
        idx_scene = [3,4,5,6,7,8]
        versions  = ["aerosol_default",r"$\times$ 0.13",r"$\times$ 0.28",r"$\times$ 0.47",r"$\times$ 0.60",r"$\times$ 0.60"+"\nabsorbing",r"$\times$ 0.47"+"\nabsorbing"]
        # ["aerosol_default","AOD(0.13)","AOD(0.28)","AOD(0.47)","AOD(0.60)","AOD(absorbing-0.60)","AOD(absorbing-0.47)"]
        sources   = ['thermal']
        librad_version = 'montecarlo_3D'

        additional_spesifications = ['_AOD(default)', '_AOD(alpha0.8)', '_AOD(alpha0.5)', '_AOD(alpha0.3)', '_AOD(alpha0.2)', '_AOD(dynamic0.2)', '_AOD(dynamic0.3)']

        titles      =  ["MYSTIC - Thermal Aerosol Model Sensitivity\nThermal TOA Flux Difference to BMA-FLX (mean values)"]
                        #[r"Aerosol version - Thermal $(F_{\mathrm{TOA}}^{\mathrm{MYSTIC, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value"]
        png_names   =  ["Data/figures/aerosol_versions.png"]


    #######################################################################################################################
    elif what_to_plot == 'mc_photons':
        Product2  = False
        show_stds = False
        idx_scene = [3,4,5,6,7,8]
        versions  = ["mc1e2","mc1e3","mc1e4","mc1e5","mc1e6","mc1e7"]
        sources   = ['solar', 'thermal']
        librad_version = 'montecarlo_3D'

        additional_spesifications = ["_GHM_mc1e2","_GHM_mc1e3",'_GHM_mc1e4', '_GHM_mc1e5', '_GHM_mc1e6', '_GHM_mc1e7'] # ,"_mc1e8"]

        titles      =  ["MYSTIC - MC Photons Sensitivity\nSolar TOA Flux Difference to BMA-FLX (mean values)",
                        "MYSTIC - MC Photons Sensitivity\nThermal TOA Flux Difference to BMA-FLX (mean values)"]
                        # [r"Monte Carlo photons - Solar $(F_{\mathrm{TOA}}^{\mathrm{MYSTIC, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value",
                        # r"Monte Carlo photons - Thermal $(F_{\mathrm{TOA}}^{\mathrm{MYSTIC, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value"]
        png_names   =  ["Data/figures/mc_photons_solar.png",
                        "Data/figures/mc_photons_thermal.png"]

        ###### MC-std (MYSTIC-only statistics) ###########################
        # Note: want to see std-plot -> look at mc_photons_"source"_std.txt 
        # INFO: These values is for all computed pixels, not only for pixel with good quality status
        plot_mean_data("Data/mc_photons_solar_std.txt", title="MYSTIC - MC Photons Sensitivity on STD\nSolar TOA Flux-STD (mean values)") #r"MC photons std - Solar MYSTIC $\sigma(F_{\mathrm{TOA}}^{version})$: mean value")
        plot_mean_data("Data/mc_photons_thermal_std.txt", title="MYSTIC - MC Photons Sensitivity on STD\nSolar TOA Flux-STD (mean values)") #r"MC photons std - Thermal MYSTIC $\sigma(F_{\mathrm{TOA}}^{version})$: mean value")
        ###################################################################


    #######################################################################################################################
    elif what_to_plot == 'SWIA':
        Product2  = False
        show_stds = False   
        idx_scene = [3,4,5,6,7,8,11]
        versions  = ['SWIA','SNIA','SWNA','SNNA','NWIA','SWIN']
        sources   = ['solar', 'thermal']
        librad_type  = versions
        librad_version = 'montecarlo_3D'
        additional_spesifications = ['_GHM_mc1e4','','','','','','',] 
        statistics = 'mean'  # chose from: {'mean', 'min', 'max'}

        titles      =  ["MYSTIC - Atmospheric Components\nSolar TOA Flux Difference: SWIA - Version (" + statistics + " values)",
                        "MYSTIC - Atmospheric Components\nThermal TOA Flux Difference: SWIA - Version (" + statistics + " values)"]
                        # [r"Atmospheric components - Solar MYSTIC $(F_{\mathrm{TOA}}^{\mathrm{SWIA}}-F_{\mathrm{TOA}}^{\mathrm{variant}})$: " + statistics + " value",
                        # r"Atmospheric components - Thermal MYSTIC $(F_{\mathrm{TOA}}^{\mathrm{SWIA}}-F_{\mathrm{TOA}}^{\mathrm{variant}})$: " + statistics + " value"]
        png_names   =  ["Data/figures/SWIA_solar_" + statistics + ".png",    
                        "Data/figures/SWIA_thermal_" + statistics + ".png"]
    #######################################################################################################################

    elif what_to_plot == '3D_buffer':
        Product2  = True
        show_stds = False
        idx_scene = [3,4,5,8,11]
        sources   = ['solar', 'thermal']
        versions  = sources
        librad_version = 'montecarlo_3D'
        # Create list to store data for both sources
        formatted_data = [] # [solar_array, thermal_array]
        formatted_stds = [] # [solar_array, thermal_array]

        additional_spesifications  = ['_All_FullBuffer']
        additional_spesifications2 = '_All_SmallBuffer'
        titles      =  ["", "MYSTIC - 3D Buffer Size Sensitivity\nSolar & Thermal TOA Flux Difference: Large vs Small Buffer (mean values)"] # r"3D-Buffer - Solar & Thermal $(F_{\mathrm{TOA}}^{\mathrm{LargeBuffer}}-F_{\mathrm{TOA}}^{\mathrm{SmallBuffer}})$: mean value"]
        png_names   =  ["", "Data/figures/3D_buffer.png"]
    #######################################################################################################################

    elif what_to_plot == 'correlation':
        Product2 = False
        show_stds = False
        statistics = 'all_pixels_values'
        idx_scene =  [5,8,12] #[3,4,5,8,11] #[3,4,5,8,11] # [3,4,5,8,11] # [3,4] #[3,4,5,6,7,8]
        sources   = ['solar'] # Chose either solar or thermal, not both
        versions = sources
        librad_version = 'montecarlo_3D' # 'montecarlo_3D' 'disort_1D'

        idx = 1
        additional_spesifications = [['_All'],['_All_MCIPA'],['_All_SmallBuffer'],['_All_FullBuffer'],['_All_MegaBuffer'], ['_All_GigaBuffer'],['_GHM'],['_SC'],['_RA']][idx]
        titles = [[' - DISORT'],[' - MCIPA'],[' - Small Buffer'],[' - Large Buffer'],[' - Mega Buffer'],[' - Giga Buffer'],[' - GHM'],[' - SC'],[' - RA']][idx]
        # titles = ['']
        png_names = ['Data/figures/correlation' + additional_spesifications[0][0] + '.png']
    
    elif what_to_plot == 'plot_traj':
        idx_scene =  [3,4,5,8,11] # [3,4] #[3,4,5,6,7,8]
        SceneNames = [SceneNames[i][0] for i in idx_scene]
        sources   = ['']
        librad_version = 'montecarlo_3D'
        additional_spesifications = [['_All_FullBuffer'],['_All_SmallBuffer']][0]

        titles = ['Trajectories']
        png_names = ['Data/figures/trajectories.png']

    elif what_to_plot == 'multi_libRad':
        Product2 = False; BMAFLX = True

        want_quality_status = True
        modify_xlim = False
        if modify_xlim: 
            idx = 2
            lat_ranges = [
                (42, 52),       # 6158D 
                (43, 51),       # 7883D
                (10.1, 13.9),   # 6886E
                (74.27, 74.41), # 6888C
                (80.63, 80.74), # 6888C
                (51.40, 52.00), # 6518D
                (11.50, 12.00), # 6886E
                (18.40, 21.40), # 6886E
                ####### OLD ##########
                (74.0, 74.5),   # 6888C
                (80.5, 81.0),   # 6888C
                (75.0, 76.0),   # 6331C
                (41.0, 52.5),   # 6518D
                (11.5, 15.0),   # 6886E
                (46.0, 52.0)    # 6518D
            ][idx]
        
        ### Choose between 5, 8, 12 ###
        idx_scene = 5
        SceneName = SceneNames[idx_scene][0] 
        sources   = ['solar']
        librad_version_list = ['montecarlo_3D','montecarlo_3D','montecarlo_3D','montecarlo_3D','montecarlo_3D'][1:5]
        additional_spesifications = ['_All_MCIPA', '_All_SmallBuffer', '_All_FullBuffer', '_All_MegaBuffer', '_All_GigaBuffer'][1:5]
        plot_type_list =  [
            ['solar_both', 'solar_flx_diff', 'correlation_plot'],
            ['solar_both'],
            ['correlation_plot'],
            ['solar_flx_diff']
            ][0]
        quantity_list = None 
        # quantity_list = ['tot_wp','tot_wc']

        titles = ['']; png_names = [''] 
        plotdir_base = './figures/'; mode_folder = 'MYSTIC/' if '3D' in librad_version_list[-1] else 'DISORT/' 
        plotdir = os.path.join(plotdir_base, SceneName, mode_folder)










    ################################# NOT CHANGE  #########################################
    for source, title, png_name in zip(sources, titles, png_names):

        if what_to_plot == 'plot_traj':
            plot_traj_on_globe(SceneNames, title=title)
            continue 

         
        elif "multi_libRad" in what_to_plot: 
            for plot_type in plot_type_list: 
                libRad_list = []
                for librad_version, spec in zip(librad_version_list, additional_spesifications):
                    Product = (
                            'libRad_' + 
                            version_identifier  + '_' +
                            librad_version      + '_' +
                            librad_type         + '_' +
                            source_str          + '_' +
                            SceneName           + 
                            spec                + '.nc'
                        )

                    libRad, BMAFLX, ACMCOM, libRad2, info = get_instances(SceneName, Product, Product2, BMAFLX)
                    libRad_list.append(libRad)

                pngfile = plotdir+ SceneName+'_'+plot_type+'_'+librad_version+'_'+librad_type+'.png'
                plot_multi_libRad(libRad_list, additional_spesifications, BMAFLX, ACMCOM, plot_type, pngfile, info, quantity_list)
                print(f"png-file: {pngfile}")
            continue




        orbits, places, data, stds = loop_through_data(source, Product2, SceneNames, librad_type, statistics)

        # --- SWIA-studies ---
        if "SWIA" in what_to_plot: 
            # 1. Caluclate difference to SWIA data 
            # 2. Remove SWIA
            # 3. Return: Flux(SWIA - XXXX)
            data = np.array(data)
            SWIA = data[:,[0]]
            data = SWIA - data

            # now remove SWIA column
            data = data[:, 1:]
            versions_without_SWIA = versions[1:]

            plot_lollipop(
                versions_without_SWIA, orbits, places, data, swia_values=np.squeeze(SWIA), # (n,) for legend labels
                ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]", 
                title=title,
                png_name=png_name
            )
            continue # do not call plot() 
        
        # --- 3D-buffer-studies ---
        elif "3D_buffer" in what_to_plot: 
            formatted_data.append(data.squeeze())  # data.squeeze() -> shape (5,)
            formatted_stds.append(stds.squeeze())
            continue # do not call plot() 
       
        
        elif "correlation" in what_to_plot:
            plot_correlation(
                source, versions, orbits, places, data,
                ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]",   #'TOA upward flux <diff> [W/m$^2$]',
                title_spec=title,
                show_stds=show_stds,
                png_name=png_name
            )
            continue # do not call plot() 
    
        # --- Ice_habit, Thermal_AOD, mc_photons ---
        plot(
            versions, orbits, places, data, stds,
            ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]",   #'TOA upward flux <diff> [W/m$^2$]',
            title=title,
            show_stds=show_stds,
            png_name=png_name
        )


    if "3D_buffer" in what_to_plot:
        formatted_data = np.transpose(np.asarray(formatted_data)) 
        formatted_stds = np.transpose(np.asarray(formatted_stds))
        # shape (n_orbits, 2): [:,0]=solar, [:,1]=thermal
        plot(
            versions, orbits, places, formatted_data, formatted_stds,
            ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]",  
            title=title,
            show_stds=show_stds,
            png_name=png_name
        )

    #########################################################################################

        















































# ############################################# OLD .txt VERSION ####################################




# def plot(versions, orbits, places, values, stds, ylabel, title, show_stds=True, png_name="Data/plot.png"):
#     x = range(len(versions))
#     fig, ax = plt.subplots(figsize=FIGSIZE)


#     # ------------------------------------------- Individual plot settings ----------------------------
#     if '3D_buffer' in png_name:
#         # Index: columns for 'solar' and 'thermal'
#         isolar, itherm = 0, 1

#         # Build arrays
#         solar_mu = np.array([values[i][isolar] for i in range(len(orbits))], float)
#         solar_sd = np.array([stds[i][isolar]  for i in range(len(orbits))], float)
#         therm_mu = np.array([values[i][itherm] for i in range(len(orbits))], float)
#         therm_sd = np.array([stds[i][itherm]  for i in range(len(orbits))], float)

#         # Plot (matches your style)
#         x  = np.arange(len(orbits))
#         dx = 0.15 # 0.18

#         ax.errorbar(x - dx, solar_mu, yerr=solar_sd, fmt='s', capsize=3, linewidth=2,
#                     label='Solar ΔF', zorder=3, color='r')
#         ax.errorbar(x + dx, therm_mu, yerr=therm_sd, fmt='s', capsize=3, linewidth=2,
#                     label='Thermal ΔF', zorder=3, color='b')

#         ax.axhline(0, lw=1, ls='--', alpha=0.6) # Baselien

#         xticks_str = [f"{o} ({p})" for o, p in zip(orbits, places)]
#         versions = xticks_str
#         # ax.set_xlabel("Orbit", fontsize=INFOSIZE)

#     else:
#         for i, orbit in enumerate(orbits):
#             label = f"{orbit} ({places[i]})"
#             color = colors[i % len(colors)]
#             mean = np.array(values[i])
#             std = np.array(stds[i])

#             ax.plot(x, mean, marker="o", color=color, label=label, linewidth=2)
#             if show_stds:
#                 ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
#     # ---------------------------------------------------------------------------------------------------

#     # --------------------------------------- General plot settings --------------------------------------
#     ax.set_xticks(x)
#     ax.set_xticklabels(versions, rotation=20, fontsize=INFOSIZE)
#     ax.set_ylabel(ylabel, fontsize=INFOSIZE)
#     ax.set_title(title, fontsize=FONTSIZE)

#     # --- Nice style adjustments ---
#     ax.set_facecolor('#f0f0f0')               # background
#     ax.grid(which='major', linestyle='--', alpha=0.4) # major grid lines
#     ax.tick_params(axis='both', which='major', labelsize=INFOSIZE) # major ticks font size
    
#     for spine in ['top', 'right']: # hide top and right spines
#         ax.spines[spine].set_visible(False)
#     # -------------------------------
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(png_name)
#     print(f"Plot saved to {png_name}")

# def plot_lollipop(versions, orbits, places, values, swia_values, ylabel, title, png_name="Data/plot.png"):
#     values = np.asarray(values, float)
#     x = np.arange(len(versions))
#     fig, ax = plt.subplots(figsize=FIGSIZE)

#     # small horizontal jitter so orbits don't overlap within each component
#     jitter = np.linspace(-0.35, 0.35, len(orbits)) if len(orbits) > 1 else [0.0]

#     # symmetric y-limits around zero
#     ymax = np.max(np.abs(values)) or 1.0
#     ax.set_ylim(-1.1*ymax, 1.1*ymax)

#     for i, orb in enumerate(orbits):
#         xi = x + jitter[i]
#         yi = values[i]
#         color = colors[i % len(colors)]
#         ax.vlines(xi, 0, yi, lw=2, color=color)
#         lbl = f"{orb} (SWIA: {swia_values[i]:.2f}" + r" W/m$^{2}$)" if swia_values is not None else orb
#         ax.plot(xi, yi, 'o', color=color, label=lbl)

    

#     # --- Nice style adjustments ---
#     ax.set_facecolor('#f0f0f0')               # background
#     ax.grid(which='major', linestyle='--', alpha=0.4) # major grid lines
#     ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*0.9) # major ticks font size
    
#     for spine in ['top', 'right']: # hide top and right spines
#         ax.spines[spine].set_visible(False)
#     # -------------------------------

#     ax.axhline(0, lw=1, ls='--', color='0.6')
#     ax.set_xticks(x)
#     ax.set_xticklabels(versions, fontsize=INFOSIZE)
#     ax.set_ylabel(ylabel, fontsize=INFOSIZE)
#     ax.set_title(title, fontsize=FONTSIZE)
#     ax.legend(ncol=2, frameon=False)
#     fig.tight_layout(); fig.savefig(png_name)
#     print(f"Plot saved to {png_name}")





# def plot_mean_std_data(filename, title, main_title=False, png_name="Data/figures/test.png"):
#     # Mulitple files:
#     if isinstance(filename, (list, tuple)):
#         N = len(filename)
#         versions = []
#         orbits   = []
#         places   = []
#         values   = []
#         stds     = []
        
#         for i, file in enumerate(filename):
#             v, o, p, val, s = read_data_cvs_mean_std(file)
#             versions.append(v); orbits.append(o); places.append(p); values.append(val); stds.append(s)


#         plot_subplot(
#             versions, orbits, places, values, stds,
#             ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]",  #'TOA upward flux std [W/m$^2$]',
#             titles=title,
#             main_title=main_title,
#             show_stds=False, # Given data is mean only
#             png_name=png_name
#         )
#     # Single File: 
#     else: 
#         versions, orbits, places, means, stds = read_data_cvs_mean_std(filename)

#         # Generate png_name from filename
#         png_name = "Data/figures/" + filename[5:-4] + ".png"

#         plot(
#             versions, orbits, places, means, stds,
#             ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]",   #'TOA upward flux <diff> [W/m$^2$]',
#             title=title,
#             show_stds=show_stds,
#             png_name=png_name
#         )

# def plot_mean_data(filename, title, main_title=False, png_name="Data/figures/test.png"):
#     # Mulitple files:
#     if isinstance(filename, (list, tuple)):
#         N = len(filename)
#         versions = []
#         orbits   = []
#         places   = []
#         values   = []
        
#         for i, file in enumerate(filename):
#             v, o, p, val = read_data_csv_mean(file)
#             versions.append(v); orbits.append(o); places.append(p); values.append(val)


#         plot_subplot(
#             versions, orbits, places, values, np.full_like(np.asarray(values, float), np.nan), # Dummy stds, plot() expect 2nd array
#             ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]",  #'TOA upward flux std [W/m$^2$]',
#             titles=title,
#             main_title=main_title,
#             show_stds=False, # Given data is mean only
#             png_name=png_name
#         )
#     # Single File: 
#     else: 
#         versions, orbits, places, values = read_data_csv_mean(filename)

#         # Generate png_name from filename
#         png_name = "Data/figures/" + filename[5:-4] + ".png"

#         # Plot spesification  -----------
#         if "SWIA" in title:
#             # print(values)
#             # 1. Caluclate difference to SWIA values 
#             # 2. Remove SWIA
#             # 3. Return: Flux(SWIA - XXXX)
#             values = np.array(values)
#             SWIA = values[:,[0]]
#             values = SWIA - values

#             # now remove SWIA column
#             values = values[:, 1:]
#             versions = versions[1:]

#             plot_lollipop(
#                 versions, orbits, places, values, swia_values=np.squeeze(SWIA), # (n,) for legend labels
#                 ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]", 
#                 title=title,
#                 png_name=png_name
#             )
#             return
#         # --------------------------------

#         plot(
#             versions, orbits, places, values, np.full(len(values), np.nan), # Dummy stds, plot() expect 2nd array
#             ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]",  #'TOA upward flux std [W/m$^2$]',
#             title=title,
#             show_stds=False, # Given data is mean only
#             png_name=png_name
#         )

    

#     # # Plot spesification  -----------
#     # if "SWIA" in title:
#     #     # print(values)
#     #     # 1. Caluclate difference to SWIA values 
#     #     # 2. Remove SWIA
#     #     # 3. Return: Flux(SWIA - XXXX)
#     #     values = np.array(values)
#     #     SWIA = values[:,[0]]
#     #     values = SWIA - values

#     #     # now remove SWIA column
#     #     values = values[:, 1:]
#     #     versions = versions[1:]

#     #     plot_lollipop(
#     #         versions, orbits, places, values, swia_values=np.squeeze(SWIA), # (n,) for legend labels
#     #         ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]", 
#     #         title=title,
#     #         png_name=png_name
#     #     )
#     #     return
#     # # --------------------------------

#     # plot(
#     #     versions, orbits, places, values, np.full(len(values), np.nan), # Dummy stds, plot() expect 2nd array
#     #     ylabel=r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]",  #'TOA upward flux std [W/m$^2$]',
#     #     title=title,
#     #     show_stds=False, # Given data is mean only
#     #     png_name=png_name
#     # )



# def read_data_cvs_mean_std(filename):
#     """
#     Reads data from a CSV-like text file where each row is:
#     Orbit,Place,type1,type2,...

#     Returns:
#         versions: list of version names (from header)
#         orbits: list of orbit names
#         places: list of place names
#         means: 2D list of mean values [orbit][version]
#         stds: 2D list of std values [orbit][version]
#     """
#     with open(filename) as file:
#         lines = [l.strip() for l in file if l.strip()]

#     header = lines[0].split(",")
#     versions = header[2:]  # aerosol version names

#     orbits = []
#     places = []
#     means = []
#     stds = []

#     for line in lines[1:]:
#         parts = line.split(",")
#         orbits.append(parts[0])
#         places.append(parts[1])
#         row_means = []
#         row_stds = []
#         for cell in parts[2:]:
#             mean_str, std_str = cell.split("±")
#             row_means.append(float(mean_str))
#             row_stds.append(float(std_str))
#         means.append(row_means)
#         stds.append(row_stds)

#     return versions, orbits, places, means, stds

# def read_data_csv_mean(filename):
#     """
#     Reads data from a CSV-like text file where each row is:
#     Orbit,Place,mc1e2,mc1e3,...

#     Returns:
#         versions: list of version names (from header)
#         orbits: list of orbit names
#         places: list of place names
#         values: 2D list of values [orbit][version]
#     """
#     with open(filename) as file:
#         lines = [l.strip() for l in file if l.strip()]

#     # Header defines version names
#     header = lines[0].split(",")
#     versions = header[2:]  

#     orbits = []
#     places = []
#     values = []

#     for line in lines[1:]:
#         parts = line.split(",")
#         orbits.append(parts[0])
#         places.append(parts[1])
#         row_values = [float(x) for x in parts[2:]]
#         values.append(row_values)

#     return versions, orbits, places, values


# def plot_subplot(versions_list, orbits_list, places_list, values_list, stds_list, ylabel, titles, main_title, show_stds=True, png_name="Data/plot.png"):
#     N = len(values_list)
#     x = range(len(versions_list[0]))
#     fig, axes = plt.subplots(N, 1, figsize=(FIGSIZE[0], FIGSIZE[1]*N)) #, constrained_layout=True)

#     for i in range(N):
#         versions = versions_list[i]
#         orbits   = orbits_list[i]
#         places   = places_list[i]
#         values   = np.array(values_list[i])
#         stds     = np.array(stds_list[i])
#         title    = titles[i]

#         ax = axes[i]

#         for i, orbit in enumerate(orbits):
#             label = f"{orbit} ({places[i]})"
#             color = colors[i % len(colors)]
#             mean = np.array(values[i])
#             std = np.array(stds[i])

#             ax.plot(x, mean, marker="o", color=color, label=label, linewidth=2)
#             if show_stds:
#                 ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    
#         ax.set_xticks(x)
#         ax.set_xticklabels(versions, rotation=0, fontsize=INFOSIZE)
#         ax.set_ylabel(ylabel, fontsize=INFOSIZE)
#         ax.set_title(title, fontsize=FONTSIZE-1)

#         # --- Nice style adjustments ---
#         ax.set_facecolor('#f0f0f0')               # background
#         ax.grid(which='major', linestyle='--', alpha=0.4) # major grid lines
#         ax.tick_params(axis='both', which='major', labelsize=INFOSIZE) # major ticks font size
        
#         for spine in ['top', 'right']: # hide top and right spines
#             ax.spines[spine].set_visible(False)
#         # -------------------------------
#         ax.legend()

    
#     fig.suptitle(main_title, fontsize=FONTSIZE)
#     # fig.tight_layout(rect=(0, 0, 1, 0.99))  # keep top 5% free for suptitle
#     fig.tight_layout()
#     fig.savefig(png_name)
#     print(f"Plot saved to {png_name}")



    #############################################################################


    # # AEROSOL VERSION
    # plot_mean_std_data("Data/aerosol_versions.txt", title=r"Aerosol version — Thermal $(F_{\mathrm{TOA}}^{\mathrm{MYSTIC, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value")


    # # ICE HABIT
    # filenames   = ["Data/Ice_Habit_all_points_solar.txt", "Data/Ice_Habit_all_points_thermal.txt",]      
    # titles      = ["Solar (SW)", "Thermal (LW)"]
    # main_title  = r"Ice habit — $(F_{\mathrm{TOA}}^{\mathrm{MYSTIC, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value"
    # png_name    = "Data/figures/Ice_Habit_all_points.png"
    # plot_mean_std_data(filename=filenames, title=titles, main_title=main_title, png_name=png_name)
    #     # plot_mean_std_data("Data/Ice_Habit_all_points_solar.txt", title=r"Ice habit — Solar $(F_{\mathrm{TOA}}^{\mathrm{MYSTIC, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value")
    #     # plot_mean_std_data("Data/Ice_Habit_all_points_thermal.txt", title=r"Ice habit — Thermal $(F_{\mathrm{TOA}}^{\mathrm{MYSTIC, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value")


    # # SWIA  
    # plot_mean_data("Data/SWIA_solar_mean.txt", title=r"Atmospheric components — Solar MYSTIC $(F_{\mathrm{TOA}}^{\mathrm{SWIA}}-F_{\mathrm{TOA}}^{\mathrm{variant}})$: mean value")
    # plot_mean_data("Data/SWIA_thermal_mean.txt", title=r"Atmospheric components — Thermal MYSTIC $(F_{\mathrm{TOA}}^{\mathrm{SWIA}}-F_{\mathrm{TOA}}^{\mathrm{variant}})$: mean value")
    # plot_mean_data("Data/SWIA_solar_min.txt", title=r"Atmospheric components — Solar MYSTIC $(F_{\mathrm{TOA}}^{\mathrm{SWIA}}-F_{\mathrm{TOA}}^{\mathrm{variant}})$: min value")
    # plot_mean_data("Data/SWIA_thermal_min.txt", title=r"Atmospheric components — Thermal MYSTIC $(F_{\mathrm{TOA}}^{\mathrm{SWIA}}-F_{\mathrm{TOA}}^{\mathrm{variant}})$: min value")
    # plot_mean_data("Data/SWIA_solar_max.txt", title=r"Atmospheric components — Solar MYSTIC $(F_{\mathrm{TOA}}^{\mathrm{SWIA}}-F_{\mathrm{TOA}}^{\mathrm{variant}})$: max value")
    # plot_mean_data("Data/SWIA_thermal_max.txt", title=r"Atmospheric components — Thermal MYSTIC $(F_{\mathrm{TOA}}^{\mathrm{SWIA}}-F_{\mathrm{TOA}}^{\mathrm{variant}})$: max value")


    # # MC PHOTONS -------- Not use i think -----------------
    # # plot_mean_std_data("Data/mc_photons_solar.txt", title=r"MC photons — Solar $(F_{\mathrm{TOA}}^{\mathrm{MYSTIC, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value")
    # # plot_mean_std_data("Data/mc_photons_thermal.txt", title=r"MC photons — Thermal $(F_{\mathrm{TOA}}^{\mathrm{MYSTIC, version}}-F_{\mathrm{TOA}}^{\mathrm{BMA-FLX}})$: mean value")


    # # MC-std (MYSTIC-only statistics) 
    # # -----INFO: These values is for all computed pixels, not only for pixel with good quality status-----------------------
    # filenames   = ["Data/mc_photons_solar_std.txt", "Data/mc_photons_thermal_std.txt"]      
    # titles      = ["Solar (SW)", "Thermal (LW)"]
    # main_title  = r"MC photons std — MYSTIC $\sigma(F_{\mathrm{TOA}}^{version})$: mean value"
    # png_name    = "Data/figures/mc_photons_std.png"
    # plot_mean_data(filename=filenames, title=titles, main_title=main_title, png_name=png_name)
    #     # plot_mean_data("Data/mc_photons_solar_std.txt", title=r"MC photons std — Solar MYSTIC $\sigma(F_{\mathrm{TOA}}^{version})$: mean value")
    #     # plot_mean_data("Data/mc_photons_thermal_std.txt", title=r"MC photons std — Thermal MYSTIC $\sigma(F_{\mathrm{TOA}}^{version})$: mean value")

    
    # filenames   = ["Data/mc_photons_solar_std_show_arve.txt", "Data/mc_photons_thermal_std_show_arve.txt"]            
    # titles      = ["Solar (SW)", "Thermal (LW)"]
    # main_title  = r"MC photons std — MYSTIC $\sigma(F_{\mathrm{TOA}}^{version})$: mean value"     
    # png_name    = "Data/figures/mc_photons_std_show_arve.png"
    # plot_mean_data(filename=filenames, title=titles, main_title=main_title, png_name=png_name)
    #     # plot_mean_data("Data/mc_photons_solar_std_show_arve.txt", title=r"MC photons std — Solar MYSTIC $\sigma(F_{\mathrm{TOA}}^{version})$: mean value")
    #     # plot_mean_data("Data/mc_photons_thermal_std_show_arve.txt", title=r"MC photons std — Thermal MYSTIC $\sigma(F_{\mathrm{TOA}}^{version})$: mean value")
    # # ----------------------------------------------------------------------------------------------------------------------


    # # 3D BUFFER
    # plot_mean_std_data("Data/3D_buffer.txt", title=r"3D-Buffer — Solar & Thermal $(F_{\mathrm{TOA}}^{\mathrm{LargeBuffer}}-F_{\mathrm{TOA}}^{\mathrm{SmallBuffer}})$: mean value")




    

    
    




