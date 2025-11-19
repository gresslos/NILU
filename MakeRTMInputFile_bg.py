import numpy as np
import pickle
import os
import sys
import multiprocessing
import glob
import scipy
import scipy.constants
from scipy.interpolate import griddata
import dted
import pdb # For debugging
from mpi4py import MPI # Use MPI for multiprocessing
from datetime import datetime, timedelta, timezone


#import ReadEarthCAREL2 as ReadEC
import ReadEarthCAREL2_bg as ReadEC
import UVspec as UVspec



# BG: overflow encouner - Warnings from reading ACM_COM files
# BG: Hiding the 'overflow encountered in multiply' warnings:
np.seterr(over='ignore')


"""
BG NOTE 
-------------------- CHANGES MADE ---------------------
    1. line 851 (if surface):       irec -> ACMCOM.surface_emissivity_type_index    
    2. line 676-677 (if iccloud):   lwc -> iwc = iwc[::-1] og reff -> icreff = icreff[::-1]   
    3. line 397:                    changed T = last_T (not use semi-realistic T = 300 as Arve proposed) (see code for explenation)
    4. line 1321:                   Inserted "try" - "except" block in case .flx.src files are not produced (happend at 'Orbit_05458F' ia = 4030)
                                            - did not manage to find out why it happend
    5. line 596 (if wccloud) 
    and  line 746 (if iccloud):     iz += 1 for each defined altitude (h > 0) -> not defined for wc, ic > 0 (see code for explenation)

    Can find my changes in code by locating comments: "# BG:"
---------------------------------------------------------
"""






# tms cron, aurai: bash; conda activate NEVARenv; python3 TestRTMInputFile.py &> tmpA&
# ESO: module load gcc openmpi netcdf-c gsl
#      source env/bin/activate
#      srun --mem=10Gb --time=0-12:0:0 python ./MakeRTMInputFile_v2.py > run.out 2>&1 &


# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

# initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
np_mpi = comm.Get_size()


def pdeg2km(p):
    latfact = 111.32
    p0 = latfact*p[0]/1000
    lonfact = 40075 * np.cos(np.deg2rad( p[1] )) / 360
    p1 = latfact*p[1]/1000
    return (p0,p1)


def Check3DBufferSize(ACM3D, ia):

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ACM3D, ia)
    if Nx>0 and Ny>0:
        BufferOK=True
    else:
        BufferOK=False
        
    return BufferOK


def Calc3DBufferSize(ACM3D, ia):
    
    # See Fig. 3, Qu et al, 2023, AMT.
    n_buffer_fore = ACM3D.number_pixels_along_track_buffer_zone_fore_view[ia]
    n_buffer_back = ACM3D.number_pixels_along_track_buffer_zone_back_view[ia]
    n_buffer = np.max([n_buffer_fore, n_buffer_back])
    n_assess= int(ACM3D.number_pixels_along_track_assessment_domain/2)+1 # Add 1 for safety as int always round down
    
    m_buffer = ACM3D.number_pixels_across_track_buffer_zone[ia]                  
    m_assess = ACM3D.number_pixels_across_track_assessment_domain
    # INFO:
    # fixed values:     m_assess and n_assess 
    # dynamic values:   m_buffer and n_buffer (depend on ia) 

    # BG: Decide to use my buffer, not dynamic buffer!
    m_assess = 5 # REMERBER THIS!!!!!!!!!!!!!!
    m_buffer = 5 # REMERBER THIS!!!!!!!!!!!!!!
    n_assess = 6 # REMERBER THIS!!!!!!!!!!!!!!
    n_buffer = 6 # REMERBER THIS!!!!!!!!!!!!!!         
    
    # BG: Reduces buffer size for faster execution
    if want_small_buffer:
        m_assess = 3 # REMERBER THIS!!!!!!!!!!!!!!
        m_buffer = 3 # REMERBER THIS!!!!!!!!!!!!!!
        n_assess = 3 # REMERBER THIS!!!!!!!!!!!!!!
        n_buffer = 3 # REMERBER THIS!!!!!!!!!!!!!!

    ialongs = np.arange(ia-n_assess-n_buffer, ia+n_assess+n_buffer+1)
    iacrosses = np.arange(ACM3D.nadir_pixel_index-m_buffer-m_assess, ACM3D.nadir_pixel_index+m_buffer+m_assess+1)
    
    Nx = iacrosses.shape[0]
    Ny = ialongs.shape[0]

    if verbose:
        print(  f"---------- Inside Calc3DBufferSize -------\n"
                f"m_buffer = {m_buffer}, m_assess = {m_assess}\n"
                f"n_buffer = {n_buffer}, n_assess = {n_assess}\n"
                f"Nx = {Nx} "        #, iacrosses = {iacrosses}\n"
                f"Ny = {Ny}\n\n")   #, ialongs = {ialongs}\n\n")   

    return Nx, Ny, iacrosses, ialongs


#def GetElevation(lat_wanted, lon_wanted, DEMfolder = '../data/DEM/'):
#def GetElevation(lat_wanted, lon_wanted, DEMfolder = '/xnilu_wrk2/projects/NEVAR/data/DEM/'):
def GetElevation(lat_wanted, lon_wanted, ia, DEMfolder = '/xnilu_wrk2/projects/NEVAR/data/DEM/'):
  
    lat_below = int(lat_wanted)
    lon_below = int(lon_wanted)

    # ESO: I think there was some rounding error in the edge cases determining correct lat/lon
    # tile (and corresponding DEM file name). 

    # ESO:
    # Determine the lower-left corner of the tile containing the point
    lat_tile = int(np.floor(lat_wanted))
    lon_tile = int(np.floor(lon_wanted))

    DEM_files = DEMfolder+"DEM_files.txt"
    
    fp = open(DEM_files)
    lines = fp.readlines()
    fp.close()
    lines = [item.strip() for item in lines]
    
    lines=np.array(lines)
    
    # Find latitudes and longitudes of DEM files      
    LATs = []
    LONs = []
    for line in lines:
        fn = line.split('.')[0]
        items = fn.split('_')
        LatStartStr = items[3]
        LonStartStr = items[5]
        #LatStart = int(LatStartStr[1:])
        # ESO: trying a bugfix
        if 'N' in LatStartStr:
            LatStart = int(LatStartStr[1:])
        elif 'S' in LatStartStr:
            LatStart = -int(LatStartStr[1:])
        # Longitude
        if 'E' in LonStartStr:
            LonStart = int(LonStartStr[1:])
        elif 'W' in LonStartStr:
            LonStart = -int(LonStartStr[1:])        
        
        LATs.append(LatStart)
        LONs.append(LonStart)
     
    LATs = np.array(LATs)
    LONs = np.array(LONs)
    # y = np.where(LATs==lat_below)

    # Find the DEM file that matches the tile
    inds = np.where((LATs == lat_tile) & (LONs == lon_tile))[0]
 
    # if no DEM file found for this latitude we are over ocean and elevation is 0.0
    #if len(inds[0])==0:
    if len(inds)==0:    
        elevation = 0.0
        return elevation
    
    #Find closest longitude file
    minind=9999
    mindist=99999
    for ind,lon in enumerate(LONs[inds]):
        dist = np.abs(lon - lon_below)
        if dist< mindist:
            mindist = dist
            minind=ind

    dirn = lines[inds[0]].replace('.tar', '')
    fn = DEMfolder + dirn + '/' + 'DEM/*.dt1'
    dteds = glob.glob(fn)
    if not dteds:
        elevation=0.0
        return elevation  # No .dt1 file found

    #print('Tove:', lat_wanted, lon_wanted, lat_below, lon_below, mindist, minind, dist)

    # if no DEM file found for this longitude we are over ocean and elevation is 0.0
    # if mindist>0:
    #     elevation = 0.0
    #     return elevation

    # dirn = lines[inds[0][minind]].replace('.tar','')
    # fn = DEMfolder+dirn+'/'+'DEM/*.dt1'

    # dted_file = glob.glob(fn)[0]
    dted_file = dteds[0]

    tile = dted.Tile(dted_file)
    # print(tile.dsi.south_west_corner.latitude, tile.dsi.south_west_corner.longitude,
    #       tile.dsi.south_east_corner.latitude, tile.dsi.south_east_corner.longitude,
    #       tile.dsi.north_east_corner.latitude, tile.dsi.north_east_corner.longitude,
    #       tile.dsi.north_west_corner.latitude, tile.dsi.north_west_corner.longitude)

    try:
        elevation = tile.get_elevation(dted.LatLon(latitude=lat_wanted, longitude=lon_wanted ))
    except Exception as e:
        elevation=0.0
        print("Error for ia: {} \n {}".format(ia, e))
        print("dirn, fn, dist", dirn, fn, dist)
        print("lat_wanted, lon_wanted", lat_wanted, lon_wanted, " Setting elevation to 0.0")
        print("mindist"), mindist
        print("type mindist"), type(mindist)
        #pdb.set_trace()
        #tms MPI.Finalize()
        #tms sys.exit()


    return elevation


def extrapolate_aod(tau_ref, lambda_ref, lambda_target, alpha):
    """
    Extrapolate Aerosol Optical Depth (AOD) using the Ångström power law.

    Parameters
    ----------
    tau_ref         : Reference AOD at wavelength lambda_ref (unitless).
    lambda_ref      : Reference wavelength in nanometers (e.g., 355).
    lambda_target   : Target wavelength in nanometers (e.g., 4500 for 4.5 µm).
    alpha           : Ångström exponent, typically derived from measured AODs at two wavelengths.
        - Larger alpha (~1–2)   : fine-mode dominated aerosols.
        - Smaller alpha (~0–0.5): coarse-mode dominated aerosols.

    Notes
    -----
    - The Ångström law assumes a simple power-law dependence of AOD on wavelength:
        τ(λ) = τ(λ0) * (λ / λ0)^(-α)
    - To be noted: Extrapolation far into the infrared (e.g., >2000 nm) is highly uncertain
    """

    # Example
    # -------
    # >>> tau355 = 0.20         # AOD at 355 nm
    # >>> alpha = 1.3           # Angström exponent
    # >>> tau4500 = extrapolate_aod(tau355, 355.0, 4500.0, alpha)
    # >>> print(tau4500)
    # 0.00736  # ~AOD at 4.5 µm


    return tau_ref * (lambda_target / lambda_ref) ** (-alpha)

def aerosol_thermal_impact_bool(ia, ACMCOM):
    aerosol_class = ACMCOM.aerosol_classification[:, ia]
    # List above consist of: 
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

    # List of indecies having LW radiative impact (absorption)
    # Only DUSTY-THINGDS + starpsoheric Ash
    idx_list = [10, 14, 15, 25]
            # 10:Dust
            # 15:Dusty mix
            # 25:Stratospheric Ash
        

    # -> IF:    these numbers found in aerosol_class    -> use extrapolate_aod -> get AOD(lmb=4500nm)
    # -> ELSE:  assume small/negligeble aerosol impact  -> use aerosol_default

    # mask = np.isin(aerosol_class, idx_list)   # boolean array
    # if mask.any():
    #     # At least one LW-important aerosol present
    #     return True
    # else:
    #     return False

    # Return True if idx_list is in aerosol_class, else False
    return bool(np.isin(aerosol_class, idx_list).any())

           

# BG: This function is not used anywhere 
def ReduceLayers(ACMCOM, Name='', verbose=False):

    maxlevels=10
    maxlayers=maxlevels-1
    zsize =  ACMCOM.height_level.shape[0]
    lsize =  ACMCOM.height_level.shape[1]
    print(maxlevels, zsize, zsize-maxlevels, ACMCOM.height_level[:, ia])

    ACMCOMnew = ReadEC.Scene(Name=SceneName, verbose=verbose)
    ACMCOMnew.latitude_active = ACMCOM.latitude_active
    ACMCOMnew.longitude_active = ACMCOM.longitude_active
    ACMCOMnew.albedo_diffuse_radiation_surface_visible = ACMCOM.albedo_diffuse_radiation_surface_visible
    ACMCOMnew.albedo_diffuse_radiation_surface_near_infrared = ACMCOM.albedo_diffuse_radiation_surface_near_infrared
    ACMCOMnew.wavelengths_thermal_surface_emissivity = ACMCOM.wavelengths_thermal_surface_emissivity
    ACMCOMnew.types_surface_emissivity = ACMCOM.types_surface_emissivity
    ACMCOMnew.surface_emissivity_table = ACMCOM.surface_emissivity_table
    ACMCOMnew.surface_emissivity_type_index = ACMCOM.surface_emissivity_type_index
    ACMCOMnew.height_level = np.zeros((maxlevels, lsize ))
    newstart = zsize-maxlevels
    ACMCOMnew.height_level[:, :] = ACMCOM.height_level[newstart:zsize, :]
    ACMCOMnew.pressure_level = np.zeros((maxlevels, lsize ))
    ACMCOMnew.pressure_level[:, :] = ACMCOM.pressure_level[newstart:zsize, :]
    ACMCOMnew.temperature_level = np.zeros((maxlevels, lsize ))
    ACMCOMnew.temperature_level[:, :] = ACMCOM.temperature_level[newstart:zsize, :]
    ACMCOMnew.volume_mixing_ratio_layer_mean_O3 = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.volume_mixing_ratio_layer_mean_O3[:, :] = ACMCOM.volume_mixing_ratio_layer_mean_O3[newstart:zsize, :]
    ACMCOMnew.volume_mixing_ratio_layer_mean_O2 = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.volume_mixing_ratio_layer_mean_O2[:, :] = ACMCOM.volume_mixing_ratio_layer_mean_O2[newstart:zsize, :]
    ACMCOMnew.specific_humidity_layer_mean = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.specific_humidity_layer_mean[:, :] = ACMCOM.specific_humidity_layer_mean[newstart:zsize, :]
    ACMCOMnew.volume_mixing_ratio_layer_mean_CO2 = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.volume_mixing_ratio_layer_mean_CO2[:, :] = ACMCOM.volume_mixing_ratio_layer_mean_CO2[newstart:zsize, :]
    
    ACMCOMnew.liquid_water_content = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.liquid_water_content[:, :] = ACMCOM.liquid_water_content[newstart:zsize, :]
    ACMCOMnew.liquid_effective_radius = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.liquid_effective_radius[:, :] = ACMCOM.liquid_effective_radius[newstart:zsize, :]

    ACMCOMnew.ice_water_content = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.ice_water_content[:, :] = ACMCOM.ice_water_content[newstart:zsize, :]
    ACMCOMnew.ice_effective_radius = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.ice_effective_radius[:, :] = ACMCOM.ice_effective_radius[newstart:zsize, :]

    ACMCOMnew.aerosol_classification = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.aerosol_classification[:, :] = ACMCOM.aerosol_classification[newstart:zsize, :]
    ACMCOMnew.aerosol_extinction = np.zeros((maxlevels-1, lsize ))
    ACMCOMnew.aerosol_extinction[:, :] = ACMCOM.aerosol_extinction[newstart:zsize, :]

    ACM3Dnew = ReadEC.Scene(Name=SceneName, verbose=verbose)

    return  ACMCOMnew

def SetRTM(UVS, ia, iacr, ACM3D=None, AMACD=None, ACMCOM=None, ACMRT=None, BMAFLX=None, input_dir='', iccloud=True,
           wccloud=True, surface=True, aerosol=True, mc_basename_path='', RTdimension = '1D',
           mol_abs_param='kato2', data_dir='', source='solar', SceneName=None):

    elevation=False
    sza = BMAFLX.solar_zenith_angle[BMAFLX.indlatBMAFLX,1]  # Nadir view is in element one
    phi0 = BMAFLX.solar_azimuth_angle[BMAFLX.indlatBMAFLX,1]  # Nadir view is in element one
    # Convert from EarthCare convention to uvspec convention
    phi0 = phi0[0]+180
    if phi0>=360: phi0=phi0-360
    
    UVS.status = 'OK'
    # First some general input
    UVS.inp['data_files_path']=data_dir

    # BG: number of photon-packets injected
    # mc_photons = '100'        # 1e2
    # mc_photons = '1000'       # 1e3
    mc_photons = '10000'      # 1e4
    # mc_photons = '100000'     # 1e5
    # mc_photons = '1000000'    # 1e6
    # mc_photons = '10000000'   # 1e7
    # mc_photons = '25'
    # mc_photons = '100'
    # mc_photons = '400'
    # mc_photons = '1'
        
    if RTdimension == '3D':       
        if UVS.inp['rte_solver'] != 'montecarlo':
            print('3D geometry only available with MYSTIC. You specified:', UVS.inp['rte_solver'])
            exit()
                
    if rte_solver=='montecarlo':
        UVS.inp['mc_std']=''
        UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source
        UVS.inp['mc_basename']=UVS.mc_basename
        # ESO: set randomseed explicitly (for error checking and debugging)
        UVS.inp['mc_randomseed'] = (ia+1)*72341//11
        if RTdimension == '3D':  
            UVS.inp['mc_photons'] = mc_photons
            # UVS.inp['mc_photons']='5'    
            # UVS.inp['mc_photons']='25'     
            # UVS.inp['mc_photons']='100'
            # UVS.inp['mc_photons']='400'
            # UVS.inp['mc_photons']='1600'
            Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ACM3D, ia)  #calculates along and cross track buffer size 
            
            # Check if latitude increase or decrease
            is_increasing = np.all(np.diff(ACM3D.latitude[:, ia]) > 0)
            #            print(ACM3D.latitude[:, ia])
            #            print(is_increasing); exit()
            # In wc_file x propagates east and y propagates north, hence if latitude
            # is increasing (ascending node) keep ialongs, but reverse iacrosses.
            # If latitude is decreasing (descending node), reverse ialongs and
            # keep iacrosses.

            if is_increasing:
                iacrosses=iacrosses[::-1]  # reverse: ialong=[1,2,3,4] --> ialong=[4,3,2,1]
                ialongs=ialongs
            else:
                iacrosses=iacrosses
                ialongs=ialongs[::-1]

            # 3D wc and ic files are aligned parallel to the meridians, the EartCARE swath is not.
            # To correct for this perform a solar azimuth angle shift
            p0 = (ACM3D.longitude[iacrosses[0],  ialongs[0]],   ACM3D.latitude[iacrosses[0],  ialongs[0]])
            p1 = (ACM3D.longitude[iacrosses[0],  ialongs[-1]],  ACM3D.latitude[iacrosses[0],  ialongs[-1]])
            px = (p0[0], p1[1])

            p0 = pdeg2km(p0)
            p1 = pdeg2km(p1)
            px = pdeg2km(px)
                
            p0p1 = np.sqrt(np.power(p1[0]-p0[0],2) + np.power(p1[1]-p0[1],2))
            p0px = np.sqrt(np.power(px[0]-p0[0],2) + np.power(px[1]-p0[1],2))
                
            phi0_shift = np.rad2deg(np.arccos(p0px/p0p1)) 
            #print('phi0, phi0_shift', phi0, phi0_shift)
            
            if phi0 > 0 and phi0< 180:
                phi0 = phi0-phi0_shift
            else:
                phi0 = phi0+phi0_shift
                    
            #print('phi0', phi0, phi0_shift)
        else:
            UVS.inp['mc_photons'] = '1000'

        
    #    UVS.inp['pseudospherical']=''
    UVS.inp['mol_abs_param']=mol_abs_param
    UVS.inp['source']=source
    if source=='thermal':
        UVS.inp['wavelength']='4500 100000'
        UVS.inp['mc_backward']=''
        UVS.inp['mc_backward_output']='eup'
    else:
        UVS.inp['wavelength']='300 4000'    

    UVS.inp['output_process']='sum'
    UVS.inp['zout']='TOA'
    UVS.inp['output_user']='sza zout edir eup'
    UVS.inp['quiet']=''
    # ---- Calculate Date -> day_of_year ---- (changes made 14.11.2025)
    ProductFile = os.path.basename(ACMCOM.fn)
    date_num = '20' + ProductFile.split('20', 1)[1].split('T', 1)[0] # Extract Date from ProductFile
    year = date_num[0:4]; month = date_num[4:6]; day = date_num[6:8]
    date = day + "." + month+ "." + year
    day_of_year = datetime(int(year), int(month), int(day)).timetuple().tm_yday # Get day_of_year    
    UVS.inp['day_of_year']=day_of_year
    # print(date, day_of_year)   
    # -----------------------------------------
    
    # Gaseous atmosphere
    atmosphere_file=input_dir+'tmp'+'{:00004d}'.format(ia)+'atm.dat'
    
    
    UVS.inp['atmosphere_file']=atmosphere_file  
    f = open(atmosphere_file,'w')
    # h,p,T are on levels which we need. The rest are on layers. Instead of interpolating effectiely move the layers to levels
    # and skip top level
    iatm=0
    last_h=9999


   

    for h,p,T,o3ppv,o2ppv,h2oppv,co2ppv in zip(ACMCOM.height_level[1:,ia],ACMCOM.pressure_level[1:,ia],ACMCOM.temperature_level[1:,ia],\
                                               ACMCOM.volume_mixing_ratio_layer_mean_O3[:,ia], ACMCOM.volume_mixing_ratio_layer_mean_O2[:,ia],\
                                               ACMCOM.specific_humidity_layer_mean[:,ia], ACMCOM.volume_mixing_ratio_layer_mean_CO2[:,ia]):
        if h > -0.0001 and p < 1e+10:
            air = 1e-06*scipy.constants.N_A*p/(scipy.constants.R*T)  # in cm-3
            o3 = air*o3ppv
            o2 = air*o2ppv*1e+6
            h2o = air*h2oppv
            co2 = air*co2ppv
            last_o3ppv=o3ppv
            last_o2ppv=o2ppv
            last_h2oppv=h2oppv
            last_co2ppv=co2ppv
            last_h = h #- surface_h +100
            # /1000 converts from m to km
            # /100 converts from Pa to hPa

            f.write('{:8.3f} {:10.5f} {:8.3f} {:12.6e} {:12.6e} {:12.6e} {:12.6e} {:12.6e}\n'.format(np.abs(h/1000.),p/100.,T,air,o3,o2,h2o,co2)) 
            iatm=iatm+1

            # BG: used for setting T at h = 0 (for 3D run) 
            last_T = T; last_p = p

                    # BG: How to find surface_temperature and surface_altitude
                    # surface_h = min(v for v,p in zip(ACMCOM.height_level[1:,ia], ACMCOM.pressure_level[1:,ia]) if v > 0 and p < 1e10)
                    # print("     surface h = ", surface_h)

                    # print("     surface temperature: ", ACMCOM.surface_temperature[iacr,ia]) 
                    # print("last_T = ", last_T)
                    # print("last_p = ", last_p)

    # Add surface at 0 km altitude if not included in profile
    if np.abs(last_h) > 0.0001 and RTdimension == '3D':
        h= 0
        p= 101300 #BG: Assume correct given normal atm-pressure at sea-surface  
        T= last_T 
            # BG: NOTE
            # Different T at h = 0 give different flx-results
            # T = last_T -> best
            # T = ACMCOM.surface_temperature[iacr,ia] -> next best
            # T = 300 (semi-realistic) -> worst result
                # - This was Arve's recomendation given he said:
                # "Just add some semi-realistc numbers, these will not be included anyways because of elevation file."
            
            # BG Explenation why T = last_T is best
            # - if h_DEM > h_surface_EarthCARE -> emitted flux dependent on T at h=0 -> it is included -> oposit of what Arve said

        air = 1e-06*scipy.constants.N_A*p/(scipy.constants.R*T)  # in cm-3
        o3 = air*last_o3ppv
        o2 = air*last_o2ppv*1e+6
        h2o = air*last_h2oppv
        co2 = air*last_co2ppv
        f.write('{:8.3f} {:10.5f} {:8.3f} {:12.6e} {:12.6e} {:12.6e} {:12.6e} {:12.6e}\n'.format(np.abs(h/1000.),p/100.,T,air,o3,o2,h2o,co2))
        surface_altitude=last_h
        elevation=True
        iatm=iatm+1
    else:
        surface_altitude=0.0001     
        elevation=True



    f.close()
    if iatm==0: UVS.status = 'No atm data for latitude: {:d} {:f}'.format(ia, ACMCOM.latitude_active[ia])

    if 'reptran' in UVS.inp['mol_abs_param']:
        atmosphere_ch4_file=input_dir+'tmp'+'{:00004d}'.format(ia)+'ch4_atm.dat'
        UVS.inp['mol_file CH4']=atmosphere_ch4_file
        fch4 = open(atmosphere_ch4_file,'w')
        atmosphere_n2o_file=input_dir+'tmp'+'{:00004d}'.format(ia)+'n2o_atm.dat'
        UVS.inp['mol_file N2O']=atmosphere_n2o_file
        fn2o = open(atmosphere_n2o_file,'w')

        iatm=0
        last_h=9999
        for h,p,T,ch4ppv,n2oppv in zip(ACMCOM.height_level[1:,ia],ACMCOM.pressure_level[1:,ia],ACMCOM.temperature_level[1:,ia],\
                                                   ACMCOM.volume_mixing_ratio_layer_mean_CH4[:,ia], ACMCOM.volume_mixing_ratio_layer_mean_N2O[:,ia]):
            if h > -0.0001 and p < 1e+10:
                air = 1e-06*scipy.constants.N_A*p/(scipy.constants.R*T)  # in cm-3
                ch4 = air*ch4ppv
                n2o = air*n2oppv
                last_ch4ppv=ch4ppv
                last_no2ppv=n2oppv
                last_h=h
                # /1000 converts from m to km
                # /100 converts from Pa to hPa
                fch4.write('{:8.3f} {:12.6e}\n'.format(np.abs(h/1000.),ch4))
                fn2o.write('{:8.3f} {:12.6e}\n'.format(np.abs(h/1000.),n2o))
                iatm=iatm+1

        # Add surface at 0 km altitude if not included in profile
        if np.abs(last_h) > 0.0001 and RTdimension == '3D':
            # Just add some semi-realistc numbers, these will not be included anyways because of elevation file.
            h=0
            p=101300
            T=300
            air = 1e-06*scipy.constants.N_A*p/(scipy.constants.R*T)  # in cm-3
            ch4 = air*last_ch4ppv
            n2o = air*last_n2oppv
            fch4.write('{:8.3f} {:12.6e}\n'.format(np.abs(h/1000.),ch4))
            fn2o.write('{:8.3f} {:12.6e}\n'.format(np.abs(h/1000.),n2o))
            iatm=iatm+1

        fch4.close()
        fn2o.close()
    






    # Water cloud
    if wccloud:
        if 'reptran' in UVS.inp['mol_abs_param']:
            UVS.inp['wc_properties']='mie interpolate'
        else:
            UVS.inp['wc_properties']='mie'

        if RTdimension == '1D':       
            wc_file_1D=input_dir+'tmp'+'{:00004d}'.format(ia)+'wc1D.dat'
            UVS.inp['wc_file 1D']=wc_file_1D
            f = open(wc_file_1D,'w')
            iwc=0
            for h,p,wc,wcreff in zip(ACMCOM.height_level[1:,ia],ACMCOM.pressure_level[1:,ia],\
                                     ACMCOM.liquid_water_content[:,ia], ACMCOM.liquid_effective_radius[:,ia]):
                if h > -0.0000001 and p < 1e+10:
                    if wcreff > 25: wcreff=25.0
                    elif wcreff < 1.0: wcreff=1.0
                    f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(np.abs(h/1000.), wc, wcreff))
                    iwc=iwc+1

            f.close()
            if iwc==0: UVS.status = 'No wc data for latitude: {:d} {:f}'.format(ia, ACMCOM.latitude_active[ia])

        elif RTdimension == '3D':       

            ialong=0
            iacross=0
                    
            # Write mystic wc_file, see libRadtran documentation for file format
            Nxwc3D=Nx
            Nywc3D=Ny
            iz=0
            last_h=99999
            for h,p in zip(ACMCOM.height_level[1:,ia],ACMCOM.pressure_level[1:,ia]):
                if h > -1e-4 and p < 1e+10:
                    last_h=h
                    last_p=p
                    iz=iz+1

            if np.abs(last_h) > 1e-4 and last_p < 1e+10:
                iz=iz+1
                
            Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
            flag = 3 # mystic 3D format
            dx = 1.0 # 1 km horizontal resolution
            dy = 1.0
            wc_file_3D=input_dir+'tmp'+'{:00004d}'.format(ia)+'wc3D.dat'
            f = open(wc_file_3D,'w')
            f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
            f.write('{:f} {:f} '.format(dx, dy))
            iz=0
            hl = ACMCOM.height_level[1:,ia]; hl = hl[::-1]  #reverse order
            pl = ACMCOM.pressure_level[1:,ia]; pl = pl[::-1]
            indx = np.where( (hl>-1e-04) & (pl<1e+10) )
            first_h = 999999
            if len(indx[0]>=1):
                first_h=hl[indx[0][0]]
                if first_h > 0.0: #1e-06:
                    h=0
                    f.write('{:8.3f} '.format(np.abs(h/1000)) )
                    iz=iz+1

            for h,p in zip(hl[indx[0]], pl[indx[0]]):
                if h > -1e-4 and p < 1e+10:
                    f.write('{:8.3f} '.format(np.abs(h/1000)) )
                    iz=iz+1
                
            f.write('\n')


            ix=1
            wclines=0
            for iac in iacrosses:
                iy=1
                for ial in ialongs:
                    irec = ACM3D.index_construction[iac, ial]
                    # BG: modification for reduced swat-length
                    if "Orbit_05926C" in SceneName:   irec -= 2700 
                    elif "Orbit_06888C" in SceneName: irec -= 2527 
                    elif "Orbit_07277C" in SceneName: irec -= 2527
                    elif "Orbit_06331C" in SceneName: irec -= 2636

                    iz = 1

                    hl = ACMCOM.height_level[2:,irec]; hl = hl[::-1]
                    pl = ACMCOM.pressure_level[2:,irec]; pl = pl[::-1]
                    lwc = ACMCOM.liquid_water_content[3:,irec]; lwc = lwc[::-1]   #ignore the top levels (0=67km,1=62,...3=52km)
                    reff = ACMCOM.liquid_effective_radius[3:,irec]; reff = reff[::-1]
                    
                    
                    # BG: made changes
                    # - iz += 1 for each realistic h > 0 measurement by EarthCARE
                    # - previosly iz += 1 only for realistic h > 0 and wc > 0 (this was the error)
                    for h,p,wc,wcreff in zip(hl, pl, lwc, reff):
                        if h > -1e-4 and p < 1e+10:
                            if wc>0.0:
                                if wcreff > 25: wcreff=25.0
                                elif wcreff < 1.0: wcreff=1.0
                                f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, wc, wcreff))
                                    # BG: Explenation:
                                    # -This gets wrtitten if lwc > 0 at altitude h > 0
                                    # if lwv = 0 -> default line at 605 is written to file

                                wclines=wclines+1
                                   
                            iz=iz+1 # BG: correct placement
                    iy=iy+1
                ix=ix+1

            # Add dummy line to make mystic run
            if wclines==0:
                f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                    
            f.close()
            UVS.inp['wc_file 3D']=wc_file_3D
            UVS.inp['mc_std']=''
            # UVS.inp['mc_photons'] = mc_photons
            # UVS.inp['mc_photons']='100' # BG: mark out given defined above
            # UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)
            # ESO bug here?
            UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source
            #            UVS.inp['']=
            
    # Ice cloud
    if iccloud:
                    # UVS.inp['ic_properties']='yang' #'fu'
                    # UVS.inp['ic_habit_yang2013']='solid_column severe'
        UVS.inp['ic_properties']='baum_v36 interpolate' #'fu'
        UVS.inp['ic_habit']='ghm'
        # UVS.inp['ic_habit']='solid-column' 
        # UVS.inp['ic_habit']='rough-aggregate'                
    
        # Options are:
        #   'ghm'               - General Habit Mixture (GHM) involving 9 habits
        #   'solid-column'      - Severely roughened solid columns   
        #   'rough-aggregate'   - Severly roughened aggregates
        

        #        UVS.inp['ic_fu reff_def']='on'

        if RTdimension == '1D':                   
            ic_file_1D=input_dir+'tmp'+'{:00004d}'.format(ia)+'ic1D.dat'
            UVS.inp['ic_file 1D']=ic_file_1D
            f = open(ic_file_1D,'w')
            iic=0
            last_h=99999
            for h,p,ic,icreff in zip(ACMCOM.height_level[1:,ia],ACMCOM.pressure_level[1:,ia],\
                                     ACMCOM.ice_water_content[:,ia], ACMCOM.ice_effective_radius[:,ia]):
                if h > -0.0001 and p < 1e+10:
                    if icreff > 60.: icreff=60.0
                    elif icreff < 5.: icreff=5.0 #9.316

                    f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(np.abs(h/1000.), ic, icreff))
                    last_h = h/1000.
                    iic=iic+1

            f.close()
            if iic==0: UVS.status = 'No ic data for latitude: {:d} {:f}'.format(ia, ACMCOM.latitude_active[ia])

        elif RTdimension == '3D':       
            ialong=0
            iacross=0

            # Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ACM3D, ia)
            
            # Write mystic wc_file, see libRadtran documentation for file format
            Nxwc3D=Nx
            Nywc3D=Ny
            iz=0
            last_h=99999
            for h,p in zip(ACMCOM.height_level[1:,ia],ACMCOM.pressure_level[1:,ia]):
                if h > -1e-4 and p < 1e+10:
                    last_h=h
                    last_p=p
                    iz=iz+1

            if np.abs(last_h) > 1e-4 and last_p < 1e+10:
                iz=iz+1
                
            Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
            flag = 3
            dx = 1.0 # 1 km horizontal resolution
            dy = 1.0
            ic_file_3D=input_dir+'tmp'+'{:00004d}'.format(ia)+'ic3D.dat'
            f = open(ic_file_3D,'w')
            f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
            f.write('{:f} {:f} '.format(dx, dy))
            iz=0
            hl = ACMCOM.height_level[1:,ia]; hl = hl[::-1]
            pl = ACMCOM.pressure_level[1:,ia]; pl =pl[::-1]
            indx = np.where( (hl>-1e-04) & (pl<1e+10) )
            first_h = 999999
            if len(indx[0]>=1):
                first_h=hl[indx[0][0]]
                if first_h > 0.0: #1e-06:
                    h=0
                    f.write('{:8.3f} '.format(np.abs(h/1000)) )
                    iz=iz+1

            for h,p in zip(hl[indx[0]], pl[indx[0]]):
                if h > -1e-4 and p < 1e+10:
                    f.write('{:8.3f} '.format(np.abs(h/1000)) )
                    iz=iz+1                
            f.write('\n')
                
            ix=1
            iclines=0
            
            for iac in iacrosses:
                iy=1
                for ial in ialongs:
                    irec = ACM3D.index_construction[iac, ial]
                    # BG: modification for reduced swat-length
                    if "Orbit_05926C" in SceneName:   irec -= 2700 
                    elif "Orbit_06888C" in SceneName: irec -= 2527 
                    elif "Orbit_07277C" in SceneName: irec -= 2527
                    elif "Orbit_06331C" in SceneName: irec -= 2636

                    iz = 1

                    hl = ACMCOM.height_level[2:,irec]; hl = hl[::-1]
                    pl = ACMCOM.pressure_level[2:,irec]; pl = pl[::-1]
                    iwc = ACMCOM.ice_water_content[3:,irec]; iwc = iwc[::-1]           
                    reff=ACMCOM.ice_effective_radius[3:,irec]; reff = reff[::-1]      
                
                
                # BG: made changes (same as for if wcloud)
                    for h,p,ic,icreff in zip(hl, pl, iwc, reff):
                        if h > -1e-4 and p < 1e+10:
                            if ic>0.0 and iz < Nz and not np.isinf(ic):
                                # See libRadtran documentation and ic_properties option 
                                # baum_v36 for these numbers                            
                                if icreff > 60: icreff=60.0
                                elif icreff < 5.0: icreff=5.0
                                f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, ic, icreff))
                                iclines=iclines+1

                            iz=iz+1 # BG: my placement
                    iy=iy+1
                ix=ix+1

            # Add dummy line to make mystic run
            if iclines==0:
                f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                    
            f.close()
            UVS.inp['ic_file 3D']=ic_file_3D
            



    # BG: fix for clear sky scenario (SNNA)---------------------------------------------------
    # BG: creating a empty (no clouds) tmp*wc3D.dat files 
    # BG: to my understanding -> treat atmosphere as 3D structure -> maching DEM -> no errors
    if wccloud == False and iccloud == False:
        UVS.inp['wc_properties']='mie'

        iz=0
        last_h=99999
        for h,p in zip(ACMCOM.height_level[1:,ia],ACMCOM.pressure_level[1:,ia]):
            if h > -1e-4 and p < 1e+10:
                last_h=h
                last_p=p
                iz=iz+1

        if np.abs(last_h) > 1e-4 and last_p < 1e+10:
            iz=iz+1
            
        Nz = iz-1
        flag = 3 # mystic 3D format
        dx = 1.0 # 1 km horizontal resolution
        dy = 1.0
        wc_file_3D=input_dir+'tmp'+'{:00004d}'.format(ia)+'wc3D.dat'
        f = open(wc_file_3D,'w')
        f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
        f.write('{:f} {:f} '.format(dx, dy))

        # Write height axis
        iz=0
        hl = ACMCOM.height_level[1:,ia]; hl = hl[::-1]  #reverse order
        pl = ACMCOM.pressure_level[1:,ia]; pl = pl[::-1]
        indx = np.where( (hl>-1e-04) & (pl<1e+10) )
        first_h = 999999
        if len(indx[0])>=1:
            first_h=hl[indx[0][0]]
            if first_h > 0.0: #1e-06:
                h=0
                f.write('{:8.3f} '.format(np.abs(h/1000)) )
                iz=iz+1

        for h,p in zip(hl[indx[0]], pl[indx[0]]):
            if h > -1e-4 and p < 1e+10:
                f.write('{:8.3f} '.format(np.abs(h/1000)) )
                iz=iz+1      
        f.write('\n')

        # Add dummy line to make mystic run
        f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
        f.close()

        UVS.inp['wc_file 3D']=wc_file_3D
        UVS.inp['mc_std']=''
        # UVS.inp['mc_photons'] = mc_photons
        # UVS.inp['mc_photons']='100' # BG: mark out given defined above
       
        UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source
    # -------------------------------------------------------------------------------------------
            
    # Aerosol
    #
    # NOTE: This way of specifying aerosol does not use the aerosol classification information from
    #       EarthCARE. The way to include that is to use the "aerosol_file explicit" option and
    #       the EarthCARE aerosol optical information available for the EarthCARE aerosol classification
    #       scheme.
    #
    ########## BG: testing to include AEROSOLS in source=='thermal' #################
    ########## changes made (27.08.2025) ############################################
    if aerosol: # and source=='solar':  
        aero_tau_file_1D=input_dir+'tmp'+'{:00004d}'.format(ia)+'aero1D.dat'
        f = open(aero_tau_file_1D,'w')
        nheights = ACMCOM.aerosol_extinction.shape[0]-1
        ih = 0
        # Calculate aerosol optical depth following this one:
        # http://stcorp.github.io/harp/doc/html/algorithms/derivations/aerosol_optical_depth.html
        aero_tau_tot=0
        last_h=99999

        # Get Angstrøm Exponent ---------------
        if not np.isfinite(aero_tau_tot):
            aero_tau_tot=0.0
        # alpha = AMACD.aerosol_angstrom_exponent[ia,75] # BG: change to nadir-pixel-index
        alpha_solar = AMACD.aerosol_angstrom_exponent[ia,iacr] # Use lmb=670-865nm range (not 355-670nm)
        if alpha_solar > 10:
            alpha_solar = 1.1

        ###### BG (04.09.2025): Testing smaller alpha -> AOD fall off slower towards LW -> higher AOD(LW)
        # alpha_thermal = 1.1   # x0.0
        # alpha_thermal = 0.8   # x0.13
        # alpha_thermal = 0.5   # x0.32         # Reason try this: some scenes in having too high thermal flux (see OneNote Aerosol)
        # alpha_thermal = 0.3   # x0.47
        alpha_thermal = 0.2     # x0.60
            

            # -------- Explenation: -----------
            # - Fine mode (combustion, urban, smoke)  ~ 1.3 - 2.0
            # - Continetioal aerosol (Background)     ~ 0.8 - 1.3
            # - Coarser mode (dust, sea-salt)         ~ 0.0 - 0.8
        # ------------------------------------

        fill_value_threshold = 1e30
        while ih<nheights:
            h=ACMCOM.height_level[ih+1,ia]
            dz = (ACMCOM.height_level[ih,ia]-ACMCOM.height_level[ih+1,ia])/1000.  # From m to km
            if ACMCOM.aerosol_classification[ih, ia]>=0:
                if np.isinf(ACMCOM.aerosol_extinction[ih, ia]) or ACMCOM.aerosol_extinction[ih, ia] > fill_value_threshold: # extiction coeff at 355nm
                    tau_aero=0.0
                else:
                    tau_aero = ACMCOM.aerosol_extinction[ih, ia]*dz
                    #### BG (03.09.2025): Try extrapolate AOD(lmb=355nm) to 4500nm (first lmb in 'thermal') #####
                    if source=='thermal':
                        initial_lmb = 355 # [nm]
                        target_lmb = 4500 # [nm]
                        tau_aero = extrapolate_aod(tau_aero, initial_lmb, target_lmb, alpha_thermal)
                    #############################################################################################
                f.write('{:8.3f} {:10.5f}\n'.format(np.abs(h/1000.), tau_aero))
                last_h = h/1000.
                aero_tau_tot=aero_tau_tot+tau_aero
            ih=ih+1

        # Add surface at 0 km altitude if not included in profile
        # print('aerosol last_h', last_h)
        # if last_h <= -0.0001 and RTdimension == '3D':
        #     h=0
        #     tau_aero=0
        #     f.write('{:8.3f} {:10.5f}\n'.format(np.abs(h/1000.), tau_aero))
        f.close()
            
        

        beta=aero_tau_tot
        #tms (ori) if aero_tau_tot > 0.0:
        #print("ESO: aero_tau_tot", aero_tau_tot)

        if aero_tau_tot > 0.0001:
            UVS.inp['aerosol_default']=''  # v00                
            #               UVS.inp['aerosol_haze']='5'  # v01
            #               UVS.inp['aerosol_haze']='1'  # v02

            ########## BG (02.09.2025): testing to remove aerosol file (defined for lmb=355nm) for source=='thermal' #############################
            if source=='solar': 
                UVS.inp['aerosol_file tau']=aero_tau_file_1D           
                UVS.inp['aerosol_angstrom']='{:f} {:f}'.format(alpha_solar, beta)
                UVS.inp['aerosol_set_tau_at_wvl 355']=aero_tau_tot
                # print('AOD(solar) = ', aero_tau_tot)
            ######################################################################################################################################

            ######## BG (03.09.2025): Try extrapolate AOD(lmb=355nm) to 4500nm (first lmb in 'thermal') #####
            if source=='thermal':
                # print('Thermal absorbing aerosol present?   ->  ', aerosol_thermal_impact_bool(ia, ACMCOM)) 
                if aerosol_thermal_impact_bool(ia, ACMCOM): # if detected aerosol which absorbe in 'thermal'-spectrum
                    UVS.inp['aerosol_file tau']=aero_tau_file_1D    
                    UVS.inp['aerosol_set_tau_at_wvl 4500']=aero_tau_tot  
                # print('AOD (thermal) =', aero_tau_tot)
            ##################################################################################################



            ######## ------ BG: NOTES --------- ########
            # AOD(lmb=355nm), but Angstrøm Exponent alpha(lmb=670-855nm). OK?
            #         Attemt for explenation:
            #             - Arve said alpha for the synthetic data the other one appears unrealistic.
            #             - Looks like is correct for real data (looked into Orbit_06497E -> alpha either nan or 1.3)
            ######## -------------------------- ########
            



    # Surface
    if surface:
        
        # BG: Ocean BRDF properties by Cox and Munk (1954) method
        # BG: Add if surface = Ocean (idx = 6)
        if ACMCOM.surface_albedo_classification[iacr, ia] == 6: # Shape [across_track, along_track] = open water
            UVS.inp['brdf_cam']=f'u10 {ACMCOM.wind_speed_at_10_meters[iacr, ia]}'
        #     print(f'                                                                         Use Cox and Munk Model for ia = {ia}')
        # else:
        #     print(f'                                                                          NOT USE ia = {ia}')


        if RTdimension == '1D':                   
            albedo_file=input_dir+'tmp'+'{:00004d}'.format(ia)+'albedo'+source+'.dat'
            UVS.inp['albedo_file']=albedo_file
            f = open(albedo_file,'w')
            # According to Qu et al., AMT, 2023, page 2320, section 2, the L2 plane is at j=0.
            # Not easy to find documented anywhere else.....
            #        UVvisalb = ACMCOM.albedo_direct_radiation_surface_visible[0,ia]
            #        NIRalb =   ACMCOM.albedo_direct_radiation_surface_near_infrared[0,ia]
            if source=='solar':
                UVvisalb = ACMCOM.albedo_diffuse_radiation_surface_visible[iacr,ia]
                NIRalb =   ACMCOM.albedo_diffuse_radiation_surface_near_infrared[iacr,ia]

                ########## BG: TESTING REDUCING ALBEDO ########################
                # print(f'UV_vis_albedo = {UVvisalb}\nNIR_albedo = {NIRalb}')
                # r_factor = 0.75
                # UVvisalb    = UVvisalb * r_factor
                # NIRalb      = NIRalb * r_factor
                ###############################################################

                if UVvisalb>1.0 or NIRalb>1.0:
                    UVvisalb=NIRalb=0.0
                f.write('{:8.3f} {:8.5f}\n'.format(200.0, UVvisalb))
                f.write('{:8.3f} {:8.5f}\n'.format(700.0, UVvisalb))
                f.write('{:8.3f} {:8.5f}\n'.format(700.1, NIRalb))
                f.write('{:8.3f} {:8.5f}\n'.format(4500.1, NIRalb))
            elif source=='thermal':
                nwvl = ACMCOM.wavelengths_thermal_surface_emissivity.shape[0]
                cmtonm=1e-07
                for iwvl in np.arange(nwvl-1,-1,-1):
                    albedo = 1-ACMCOM.surface_emissivity_table[ACMCOM.surface_emissivity_type_index[iacr,ia]-1,iwvl]
                    wvl = 1./(ACMCOM.wavelengths_thermal_surface_emissivity[iwvl] * cmtonm)
                    f.write('{:10.3f} {:8.5f}\n'.format(wvl, albedo))

                # Add one more longer wavelength to comply with Fu wavelength grid. Assume albedo is the same.
                f.write('{:10.3f} {:8.5f}\n'.format(110000, albedo))

            f.close()
                
        elif RTdimension == '3D':       
            # mc_albedo_file=input_dir+'tmp'+'{:00004d}'.format(ia)+'mc_albedo_file.dat'
            # UVS.inp['mc_albedo_file']=mc_albedo_file
            # dx = 1.0 # 1 km horizontal resolution
            # dy = 1.0
            # f = open(mc_albedo_file,'w')
            # f.write('{:d} {:d} {:f} {:f}\n'.format(Nx, Ny, dx, dy))
            # ix=1           
            # iclines=0
            # for iac in iacrosses:
            #     iy=1
            #     for ial in ialongs:
            #         irec = ACM3D.index_construction[iac, ial]
            #         UVvisalb = ACMCOM.albedo_diffuse_radiation_surface_visible[0,irec]
            #         f.write('{:d} {:d} {:f}\n'.format(ix, iy, UVvisalb ))
            #         iclines=iclines+1
            #         iy=iy+1
            #     ix=ix+1
            # f.close()

            mc_albedo_spectral_file=input_dir+'tmp'+'{:00004d}'.format(ia)+'mc_albedo_spectral_file'+source+'.dat'
            mc_albedo_type=input_dir+'tmp'+'{:00004d}'.format(ia)+'mc_albedo_type'+source+'.dat'
            
            UVS.inp['mc_albedo_spectral_file']=mc_albedo_spectral_file 
            UVS.inp['mc_albedo_type']=mc_albedo_type
            dx = 1.0 # 1 km horizontal resolution
            dy = 1.0
            f = open(mc_albedo_spectral_file,'w')
            fat = open(mc_albedo_type,'w')
            f.write('{:d} {:d} {:f} {:f}\n'.format(Nx, Ny, dx, dy))
            ix=1           
            iclines=0
            for iac in iacrosses:
                iy=1
                for ial in ialongs:
                    irec = ACM3D.index_construction[iac, ial]  # Teste å bytte ut iac, ial med irec, irec
                    # BG: modification for reduced swat-length
                    if "Orbit_05926C" in SceneName:   irec -= 2700 
                    elif "Orbit_06888C" in SceneName: irec -= 2527 
                    elif "Orbit_07277C" in SceneName: irec -= 2527
                    elif "Orbit_06331C" in SceneName: irec -= 2636

                    ###################### THIS IS NEW TEST - REMOVE #####################
                    irec = ial
                    iacr = iac
                    ######################################################################

                    itype_index = 'itype_index_{:00004d}_{:00004d}_{:00004d}_{:00004d}.dat'.format(ia, iac, ial, irec)
                    mc_albedo_type_file = input_dir+'tmp'+'{:00004d}_{:00004d}_{:00004d}_{:00004d}'.format(ia, iac, ial, irec)+'mc_albedo_spectral_type'+source+'.dat'
                    fatf = open(mc_albedo_type_file,'w')
                    f.write('{:d} {:d} {:s}\n'.format(ix, iy, itype_index))
                    fat.write('{:s} {:s}\n'.format(itype_index, mc_albedo_type_file))

                    if source=='solar':
                        # ESO:
                        UVvisalb = ACMCOM.albedo_diffuse_radiation_surface_visible[iacr,irec]
                        UVNIRalb = ACMCOM.albedo_diffuse_radiation_surface_near_infrared[iacr,irec]

                        fatf.write('{:f} {:f}\n'.format(200, UVvisalb ))
                        fatf.write('{:f} {:f}\n'.format(700, UVvisalb ))
                        fatf.write('{:f} {:f}\n'.format(701, UVNIRalb ))
                        fatf.write('{:f} {:f}\n'.format(4500, UVNIRalb ))
                    elif source=='thermal':
                        nwvl = ACMCOM.wavelengths_thermal_surface_emissivity.shape[0]
                        cmtonm=1e-07
                        for iwvl in np.arange(nwvl-1,-1,-1):
                            #print(ACMCOM.wavelengths_thermal_surface_emissivity[iwvl] * 1e-04)
                            # This should not happen. Is there some kind of inconsistency in the
                            # synthetic data

                            if ACMCOM.surface_emissivity_type_index[iacr,irec]>26:
                                # ESO:
                                # print("WARN: irec>=26 for ia {}".format(ia))
                                albedo=0.0
                            #elif ACMCOM.surface_emissivity_type_index[0,irec]<0:
                            # ESO:
                                # print("WARNING: surface_emissivity_type_index = {} > 26 for ia {}, irec {}".format(ACMCOM.surface_emissivity_type_index[iacr,irec], ia, irec))
                            elif ACMCOM.surface_emissivity_type_index[iacr,irec]<0:
                                albedo=0.0
                                # print("WARNING, surface_emissivity_type_index = {} < 0  for ia {}, irec {}".format(ACMCOM.surface_emissivity_type_index[iacr,irec], ia, irec))
                            else:
                                #tms
                                albedo = 1-ACMCOM.surface_emissivity_table[ACMCOM.surface_emissivity_type_index[iacr,irec]-1,iwvl]
                                

                            wvl = 1./(ACMCOM.wavelengths_thermal_surface_emissivity[iwvl] * cmtonm)

                            fatf.write('{:10.3f} {:8.5f}\n'.format(wvl, albedo))

                        # Add one more longer wavelength to comply with Fu wavelength grid. Assume albedo is the same.
                        fatf.write('{:10.3f} {:8.5f}\n'.format(110000, albedo))
                    fatf.close()    
                    iclines=iclines+1
                    iy=iy+1
                ix=ix+1
            f.close()
            fat.close()
        
    # Elevation, add elevation file if surface not at 0.0
    if elevation and  RTdimension == '3D':
        
        mc_elevation_file=input_dir+'tmp'+'{:00004d}'.format(ia)+'elevation.dat'
        UVS.inp['mc_elevation_file']= mc_elevation_file

        f = open(mc_elevation_file, 'w')
        f.write('{:d} {:d} {:f} {:f}\n'.format(Nx+1, Ny+1, dx, dy))
        ix=1           
        iclines=0
        elevation0s=[]
        for iac in iacrosses:
            iy=1
            for ial in ialongs:
                # irec = ACM3D.index_construction[iac, ial] #BG: this is not in use!
                # elevation = GetElevation(ACM3D.latitude[iac, ial], ACM3D.longitude[iac, ial])
                # ESO: passing ia for debugging:
                # dted.errors.NoElevationDataError: Specified location is not contained within DTED file: (54.6N,1.0E)

                elevation = GetElevation(ACM3D.latitude[iac, ial], ACM3D.longitude[iac, ial], ia)
               
                  
                if elevation< 0.0001: elevation=0.0001
                if iy==1: elevation0=elevation
                if ix==1: elevation0s.append(elevation)
                #                print('ACM3D, lat, lon', iac, ial, ACM3D.latitude[iac, ial], ACM3D.longitude[iac, ial], elevation)
                iclines=iclines+1

                f.write('{:d} {:d} {:e}\n'.format(ix, iy, elevation/1000))
                # Elevation is at corners, not at grid centers, and must be periodic. Since
                # corners are not known, use elevation at grid center and add last elevation
                # data by reusing first elevation.
                if iy == Ny:
                    f.write('{:d} {:d} {:e}\n'.format(ix, iy+1, elevation0/1000))
                    if ix==1: elevation0s.append(elevation0)
                    
                iy=iy+1
            ix=ix+1

        # See previous comment why this.
        elevation0s = np.array(elevation0s)
        ix = Nx+1
        iy=1
        for ial in ialongs:
            f.write('{:d} {:d} {:e}\n'.format(ix, iy, elevation0s[iy-1]/1000))
            if iy == Ny:
                f.write('{:d} {:d} {:e}\n'.format(ix, iy+1, elevation0s[iy]/1000))
            iy=iy+1



    # Solar geometry
    # Set solar geometry here to make sure azimuth shift due to 3D geometry is included
    UVS.inp['sza']=sza[0]
    UVS.inp['phi0']=phi0  # Azimuth is only needed for 3D, but it does not hurt for 1D.





    return UVS

def tmp_output_filename(iia, scene_name):
    """
    Return the filename to be used for temporary output files.

    """
    pass
    













####################################################################################################################################
# NOTE:
# 
# TO DO:
#     
#####################################################################################################################################

if __name__ == "__main__":

    if my_rank == 0:
        start_time = datetime.now(timezone.utc) 
        print("Run started at", start_time.isoformat())
            
    idx_source = 2
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 

    idx_scene = [7]
    # idx_scene = [3,4,5,6,7,8,11]
    # idx_scene = [3,4,5,6,7,8]
    # idx_scene =[3,4,5,6,7,8, 11]
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



    Test    = False
    want_ps = False      # BG: if want DISORT pseudospherical 
    verbose = False
    want_3D = True       # BG: if want MYSTIC
    want_small_buffer = False    # Use this if want faster execution without much loss of 3D effects
     
    if want_3D and want_small_buffer: buffer_str = 'Small Buffer (13 x 13)'
    elif want_3D:                     buffer_str = 'Large Buffer (25 x 21)'
    else:                             buffer_str = ''
    
    # BG: To distinguish minor modifications to .nc files
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
    # additional_spesifications += '_All'
            # 3D-Cloud impact
    # additional_spesifications += '_wc'           # INFO: large-buffer
    # additional_spesifications += '_wc_test_atm_0'
            # TEST; THEN REMOVE
    additional_spesifications += '_wc_test_new_3D_surface'


    surface = True      # BG: = False -> default albedo = 0     
    wccloud = True      
    iccloud = True       
    aerosol = True        
       
    # See list above where surface, wccloud etc are set to true or false
    processes = list('NNNN')  # N = Not included
    if surface:
        processes[0]='S'
    if wccloud:
        processes[1]='W'
    if iccloud:
        processes[2]='I'
    if aerosol:
        processes[3]='A'



    if want_3D:
        rte_solver = 'montecarlo' #'mystic' 
        RTdimension = '3D'
    else:
        if want_ps:
            rte_solver = 'disort \npseudospherical'
        else: 
            rte_solver = 'disort'  # 'twostr'        
        RTdimension ='1D' 







    # ---------------------------------- Paths -----------------------------------------------
    pathL2TestProducts = '/xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/' # EarthCARE data
            # ESO: for slurm/scratch, use a relative path. the data in
            #      /xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/ are copied to SCRATCH//EarthCARE_Real
            #pathL2TestProducts = "./EarthCARE_Real/"
            # ESO: I also link the above directory to my home run directory so I can test without
            #     using scratch;
            #     cd NEVAR; ln -s /xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/ .

    RTInpOutPath = './tmpRTIO/'    # Folder to store the final results
    RTOutNetcdfPath = './RESULTS/'  # Folder name for netcdf result files
    if Test:
        RTInpOutPath = './tmpRTIOtest/'    # Folder to store the final results        
        RTOutNetcdfPath = './RESULTStest/'     # Folder name for netcdf result files ¤ BG: changed this one!

        

    # Ensure folders exist on all nodes  ############# THIS IS FOR MULTIPLE NODE RUNS ##################
    # os.makedirs(RTInpOutPath, exist_ok=True)
    # os.makedirs(RTOutNetcdfPath, exist_ok=True)

    # ESO: These are not yet at scratch, neither are the DEM files:
    uvspecpath = '/xnilu_wrk2/projects/NEVAR/libRadtran/bin/' # Folder to RTM code
    RTMdata = '/xnilu_wrk2/projects/NEVAR/libRadtran/data/' #RTM input data

 
    
    libRad_version= 'v01_'+rte_solver+'_'+RTdimension+'_'+"".join(processes)
    # libRad_version= 'vpseudospherical_'+rte_solver+'_'+RTdimension+'_'+"".join(processes)
    # --------------------------------------------------------------------------------------------








    # Start processing (generate and read EartCare input files)
    for SceneName in SceneNames:
        Product ='ALL_3D_'
        ProductPath = '*'+Product+'*'
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')        
        ProductFile = sorted(glob.glob(ProductFile))[0]        #tms: Does it mean that the first file (if several) is selected?

        if verbose: print('ProductFile', ProductFile)
        ACM3D = ReadEC.Scene(Name=SceneName, verbose=verbose)  #tms: Make a class with name, verbose (initial) + various def (e.g. read, plot)
        ACM3D.ReadEarthCAREh5(ProductFile, verbose=verbose)                              # "Self" refers to the current instance of the class

        # Get nadir pixel index
        iacr = ACM3D.nadir_pixel_index
        if my_rank == 0:
            print("\n\n\n=================================================================================================================")
            print("Nadir Pixel Index    = ", iacr)
            print("SceneName            = ", SceneName)
            print("Specs                = ", rte_solver + ' - ' + RTdimension + ' - ' + additional_spesifications + ' - ' + buffer_str)
           

        Product ='ACM_COM'
        ProductPath = '*'+Product+'*'
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        if verbose: print('ProductFile', ProductFile)
        ACMCOM = ReadEC.Scene(Name=SceneName, verbose=verbose)        
        ACMCOM.ReadEarthCAREh5(ProductFile, verbose=verbose, ACM3D=ACM3D)       #tms: use ACM3D to get same lat/lon as ACM_3D?
        
                
        Product ='AM__ACD'      #Ångström exponent
        ProductPath = '*'+Product+'*'
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        if verbose: print('ProductFile', ProductFile)
        AMACD = ReadEC.Scene(Name=SceneName, verbose=verbose)
        AMACD.ReadEarthCAREh5(ProductFile, verbose=verbose)
        AMACD.InterpolateAngstrom(ACMCOM)

        Product ='ACM_RT_'      #1D and 3D RTM fluxes
        ProductPath = '*'+Product+'*'
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        if verbose: print('ProductFile', ProductFile)
        ACMRT = ReadEC.Scene(Name=SceneName, verbose=verbose)
        ACMRT.ReadEarthCAREh5(ProductFile, verbose=verbose)
        ACMRT.SetExtent()
         
        Product ='BMA_FLX'      #BBR fluxes
        ProductPath = '*'+Product+'*'
        ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
        ProductFile = sorted(glob.glob(ProductFile))[0]
        if verbose: print('ProductFile', ProductFile)
        BMAFLX = ReadEC.Scene(Name=SceneName, verbose=verbose)
        BMAFLX.ReadEarthCAREh5(ProductFile, Resolution='StandardResolution', verbose=verbose)  #tms: "Resolution" is probably not relevant as long as 
        #                                                                                             data file/path includes "libRad" in the name
        #                                                                                             (Or "StandaResolution" is record in ntcdf-file)
                                                                                                

              
        SceneNamelibRad='libRad'
        libRad = ReadEC.Scene(Name=SceneNamelibRad, verbose=verbose)    
        libRad.latitude_active = ACMCOM.latitude_active                 #tms: ACM_COM has a paramter called latitude_active (latitude under active sensor)
        libRad.solar_eup = np.zeros(libRad.latitude_active.shape)       
        libRad.solar_eup_std = np.zeros(libRad.latitude_active.shape)
        libRad.thermal_eup = np.zeros(libRad.latitude_active.shape)
        libRad.thermal_eup_std = np.zeros(libRad.latitude_active.shape)




        # --------------------------- Domain for full / test run ------------------------------------------------   
        # Loop over all along_track indices and do RT for each
        start = 150
        end = ACMCOM.latitude_active.shape[0] - 150 # BG: subtract 150 to not get index out of grid

                            # # Modification used for Ice-Habit testing
                            # if additional_spesifications == '_GHM' or additional_spesifications == '_SC' or additional_spesifications == '_RA':
                            #     if SceneName == 'Orbit_06888C':   # Svalbard
                            #         start = int(end - end / 6)      # 71N
                            #         end   = int(end - end / 150)    # 68N
                            #     elif SceneName == 'Orbit_06518D': # USA
                            #         start = int(end / 2)            # 46N
                            #         end   = int(end / 1.5)          # 38N
                            #     elif SceneName == 'Orbit_06886E': # Africa
                            #         start = int(end / 6)            # 15N
                            #         end   = int(end / 2)            # 1N

                            #     elif SceneName == 'Orbit_07277C': # Svalbard
                            #         start = int(end - end / 4)      # 73N
                            #         end   = int(end - end / 10)     # 70N
                            #     elif SceneName == 'Orbit_06907D': # USA
                            #         start = int(end / 2)            # 46N
                            #         end   = int(end / 1.6)          # 40N      
                            #     elif SceneName == 'Orbit_06497E': # Africa
                            #         start = int(end / 6)            # 15N
                            #         end   = int(end / 2.3)          # 5N

                            #     # print(f'Latitudes            = {ACMCOM.latitude_active[start]} -> {ACMCOM.latitude_active[end]}')
                            
        # # Modification used for WC-3D-Bias testing
        if 'wc' in additional_spesifications:
            lat = ACMCOM.latitude_active
            if SceneName == 'Orbit_06497E': # Africa
                target_start_lat, target_end_lat = 3, 5
            elif SceneName == 'Orbit_06518D': # USA
                target_start_lat, target_end_lat = 40,42
            elif SceneName == 'Orbit_06888C': # USA
                target_start_lat, target_end_lat = 68,69
            elif SceneName == 'Orbit_06331C': # USA
                target_start_lat, target_end_lat = 76,78


            start, end = int(np.nanargmin(np.abs(lat - target_start_lat))),  int(np.nanargmin(np.abs(lat - target_end_lat)))
            if start > end: tmp = start; start = end; end = tmp

            # print(f'Latitudes            = {ACMCOM.latitude_active[start]:.2f} -> {ACMCOM.latitude_active[end]:.2f}')
            
            




        if Test == False:
            ialongs = (
                list(range(start, end, 1)) # BG: change 10 -> 1 if want all pixels
            )
        else: 
            # ialongs = [int(end/2)] #[300, 600, 900, 1200] #[1000, 2000, 3000, 4000]
    
            target_lat = 0
            target_idx = int(np.nanargmin(np.abs(ACMCOM.latitude_active - target_lat)))
            ialongs = [target_idx]
            print(target_idx)
            print(f'Latitude            = {ACMCOM.latitude_active[target_idx]:.2f}')

        
        
        # ----------------------------------------------------------------------------------------------------------
       





        # The problem with the simple MPI approach below is that some processes complete much
        # faster than others. If we set aside the root process to distribute tasks, we get
        # something like this:
        num_tasks = len(ialongs)
        master_only_proc=False

        if my_rank == 0:
            # If we run on 1 process only, the master process is not handling worker processes.
            if np_mpi == 1:
                task_index = 0
                master_only_proc=True
                print("master_only_proc", master_only_proc)
            else: 
                # Master process
                task_index = 0
                finished_workers = 0
                num_workers = np_mpi - 1
                status = MPI.Status()

                # Initial distribution
                for i in range(1, np_mpi):
                    # print(f"[MASTER] Initial task distribution done. task_index={task_index}, finished_workers={finished_workers}, numtasks={num_tasks}")
                    if task_index < num_tasks:
                        comm.send(task_index, dest=i, tag=1)
                        task_index += 1
                    else:
                        comm.send(None, dest=i, tag=0)  # No more tasks
                        finished_workers += 1

                # Receive results and send new tasks
                while finished_workers < num_workers:
                    # Receive (task_index, result1, result2, result3, result4)
                    # print(f"[MASTER] Waiting for results. finished_workers={finished_workers}, num_workers={num_workers}")
                    data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    # print(f"[MASTER] Received data from rank {status.Get_source()}: {data}")
                    sourceMPI = status.Get_source()
                    idx, r1, r2, r3, r4 = data
                    libRad.solar_eup[idx] = r1
                    libRad.solar_eup_std[idx] = r2
                    libRad.thermal_eup[idx] = r3
                    libRad.thermal_eup_std[idx] = r4                

                    if task_index < num_tasks:
                        # print(f"[MASTER] Sending new task {task_index} to rank {sourceMPI}")
                        comm.send(task_index, dest=sourceMPI, tag=1)
                        task_index += 1
                    else:
                        # print(f"[MASTER] Sending termination signal to rank {sourceMPI}")
                        comm.send(None, dest=sourceMPI, tag=0)  # No more tasks
                        finished_workers += 1
                        # print(f"[MASTER] finished_workers incremented: {finished_workers}")

                

        if my_rank>0 or master_only_proc:
            # Worker process
            # ESO: added this to clear tmp out files on new run
            first_iter=True
            
            while True:
                if my_rank>0:
                    # print(f"[WORKER {my_rank}] Waiting for task...")
                    task_index = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
                    # print(f"[WORKER {my_rank}] Received task_index={task_index}")
                    if task_index is None:
                        print(f"[WORKER {my_rank}] Received termination signal. Exiting.")
                        # print("     Process {} has no remaining tasks".format(my_rank))
                        break

                    try:
                        ia = ialongs[task_index]
                    except Exception as E:
                        print("Process {} has no remaining tasks".format(my_rank))
                        break
                elif master_only_proc:
                    ia=ialongs[task_index]
                    print("\n\n |----------------------------------- ia", ia," ------------------------------------|")
                
        # for inum, ia in enumerate(ialongs):
        #     # Go to next iteration if another process is handling this ia
        #     if inum % np_mpi != my_rank:
        #         continue
            # print("ia", ia)
                print("     Process {} started calculating ia={}".format(my_rank, ia))
                latitude_wanted = ACMCOM.latitude_active[ia]
                indlatBMAFLX = np.unravel_index(np.argmin(np.abs(BMAFLX.latitude - latitude_wanted), axis=None), BMAFLX.latitude.shape)
                BMAFLX.indlatBMAFLX=indlatBMAFLX

                sza = BMAFLX.solar_zenith_angle[BMAFLX.indlatBMAFLX,1]  # Nadir view is in element one

                # --- BG: ADD THIS BLOCK ---
                eup_solar = np.nan
                eup_solar_std = np.nan
                eup_thermal = np.nan
                eup_thermal_std = np.nan
                # ----------------------
                
                # BG: try to rather set to nan above!
                # zout=edir_solar=eup_solar=eup_solar_std=0.0
                # edir_thermal=eup_thermal=eup_thermal_std=0.0

                BufferOK = Check3DBufferSize(ACM3D, ia)
                if not BufferOK:
                    source='Buffer'
                    # print(f'     [WORKER {my_rank}] WARNING: Not enough 3D buffer for ia = {ia}, source = {source}. Skipping.')
                else:
                    for source in sources:
                        try: 
                            RTM=True
                            if source=='solar':
                                if sza >= 90:
                                    RTM=False

                                mol_abs_param = 'kato2' #'reptran course' #
                            elif source=='thermal':
                                mol_abs_param = 'fu' 


                            if RTM:
                                UVS = UVspec.UVspec()   #tms: UVspec.py is an external program that defines input and runs libRadtran
                                UVS.inp['rte_solver']=rte_solver
                                #tms: "SetRTM" is a function defined above - NB! Check path
                                UVS = SetRTM(UVS, ia, iacr, ACM3D=ACM3D, AMACD=AMACD, ACMCOM=ACMCOM, ACMRT=ACMRT, BMAFLX=BMAFLX,
                                            iccloud=iccloud, wccloud=wccloud, surface=surface, aerosol=aerosol,
                                            mc_basename_path=RTInpOutPath, RTdimension = RTdimension, input_dir=RTInpOutPath,
                                            mol_abs_param=mol_abs_param, data_dir=RTMdata, source=source, SceneName=SceneName)
                                if UVS.status == 'OK':
                                    # print(f'[WORKER {my_rank}] Performing RTE ia = {ia}, source = {source}')
                                    # uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+'.inp'
                                    # This was not really needed, ia is unique to each process
                                    uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+str(my_rank)+'.inp'                            
                                    uvspecOutputFile=uvspecInputFile.replace('inp','out')
                                    UVS.WriteInputFile(uvspecInputFile)
                                    # print(f"[WORKER {my_rank}] Finished SetRTM for ia={ia}, source={source}")

                                    #tms:Run libradtran from UVspec.py
                                    ##################################
                                    # #BG: comment out and try bellow stuff: UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)
                                    # print(f"[WORKER {my_rank}] Finished SingleRun for ia={ia}, source={source}")
                                    try:
                                        # print(f"[WORKER {my_rank}] About to run SingleRun for ia={ia}, source={source}")
                                        UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)
                                        # print(f"[WORKER {my_rank}] Finished SingleRun for ia={ia}, source={source}")
                                    except Exception as e:
                                        print(f"[WORKER {my_rank}] Exception in SingleRun for ia={ia}, source={source}: {e}")

                                    if UVS.inp['rte_solver']=='montecarlo':
                                        # mc_flx, eup_std = UVspec.ReadMCOut(UVS.mc_basename+'_'+source, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'
                                        # ESO: "source" added twice?
                                        # mc_flx, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                        try:
                                            mc_flx, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                            # print(f"[WORKER {my_rank}] Finished ReadMCOut for ia={ia}, source={source}")
                                        except (FileNotFoundError, ValueError) as e:
                                            print(f"[WORKER {my_rank}] WARNING: Could not read MC output for ia={ia}, source={source}. Skipping. Error: {e}")
                                            continue  # Go to the next iteration


                                        if source=='solar':
                                            eup_solar_std=eup_std
                                        elif source=='thermal':
                                            eup_thermal_std=eup_std

                                    # print(f'[WORKER {my_rank}] Still here')
                                    #print('Tove:', uvspecOutputFile)
                                    # ESO: Debugging
                                    try:
                                        # print(f"[WORKER {my_rank}] Starting ReadRTOut for ia={ia}, source={source}")
                                        sza, zout, edir, eup = UVspec.ReadRTOut(uvspecOutputFile, mol_abs_param=mol_abs_param)
                                        # print(f"[WORKER {my_rank}] Finished ReadRTOut for ia={ia}, source={source}")
                                        if UVS.inp['rte_solver']=='montecarlo':
                                            print(f"                                            [WORKER {my_rank:2}] ia={ia:4}, {source:8}, zout={zout:.1f}, eup={eup:8.2f} +- {eup_std}")
                                        else:
                                            print(f"                                            [WORKER {my_rank:2}] ia={ia:4}, {source:8}, zout={zout:.1f}, eup={eup:8.2f}")
                                    except ValueError as e:
                                        print("Error [WORKER {}]: could not read file {} with mol_abs_param {}".
                                            format(my_rank, uvspecOutputFile, mol_abs_param))
                                        print(f"Error [WORKER {my_rank}]: ", e)
                                        # pdb.set_trace()

                                    if source=='solar':
                                        eup_solar=eup
                                        edir_solar=edir
                                    elif source=='thermal':
                                        eup_thermal=eup

                                    if RTdimension == '3D':
                                        # Remove albedo files, otherwise far too many (hundred of thousands) files are in the folder
                                        if surface: UVS.removefiles( UVS.inp['mc_albedo_type'])
                                        # ESO: keep files for testing MPI
                                        # But hopefully, "removefiles" only removes files of the
                                        # current run
                                        # pass

                                    # print(f'[WORKER {my_rank}] Still here, now with edir={edir}, eup={eup}')
                                    

                                else:
                                    print(UVS.status)
                                    # print(f'[WORKER {my_rank}] WARNING: UVS.status not OK for ia={ia}, source={source}. Skipping.')
                                    zout=edir=eup=eup_std=0.0
                            else:
                                zout=edir_solar=eup_solar=eup_solar_std=0.0
                                edir_thermal=eup_thermal=eup_thermal_std=0.0



                            if rte_solver=='montecarlo':
                                if verbose: print(" |--------- OUT", rte_solver, RTdimension, ia, source, latitude_wanted, eup, "---------|\n\n")  # BG: print statemnt
                                libRad.solar_eup_std[ia] = eup_solar_std
                                libRad.thermal_eup_std[ia] = eup_thermal_std
                            else:
                                # print('OUT', ia, source, latitude_wanted, sza, zout, edir_solar, eup_solar, eup_thermal)
                                if verbose: print(" |--------- OUT", rte_solver, RTdimension, ia, source, latitude_wanted, eup, "---------|\n\n")   # BG: print statement
                        except Exception as e:
                            print(f"[WORKER {my_rank}] Unexpected exception for ia={ia}: {e}")
                            
                libRad.solar_eup[ia] = eup_solar
                libRad.thermal_eup[ia] = eup_thermal

                # ESO: Write the results to text file, to avoid redoing the whole simulation
                # if the job is aborted.
                if rte_solver=='montecarlo':
                    if first_iter:
                        first_iter=False
                        file_mode= 'w'
                    else:
                        file_mode= 'a'
                        
                        # BG: Need to change this! I THINK!
                        # with open(os.path.join(RTOutNetcdfPath, SceneNames[0] + "-" +str(my_rank)),file_mode) as f:
                    with open(os.path.join(RTOutNetcdfPath, SceneName + "-" +str(my_rank)),file_mode) as f:
                        f.write("{:6d} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n".format(ia, eup_solar, eup_solar_std, eup_thermal, eup_thermal_std))

                # Send results to root process, unless we run with 1 process
                if my_rank > 0:
                    try:
                        # print(f"[WORKER {my_rank}] About to send results for ia={ia} to master.")
                        comm.send((ia, eup_solar, eup_solar_std, eup_thermal, eup_thermal_std), dest=0)
                        # print(f"[WORKER {my_rank}] Sent results for ia={ia} to master.")
                    except Exception as e:
                        print(f"[WORKER {my_rank}] Exception when sending results for ia={ia}: {e}")
                elif master_only_proc:
                    libRad.solar_eup_std[ia] = eup_solar_std
                    libRad.thermal_eup_std[ia] = eup_thermal_std
                    libRad.solar_eup[ia] = eup_solar
                    libRad.thermal_eup[ia] = eup_thermal
                    task_index += 1
                    if task_index > len(ialongs) - 1:
                        break
                    


            # tms: Function in ReadEarthCAREL2.py that writes the output to a netcdf file                                 
            # tms libRad.WriteNetcdf(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+source+'_'+SceneName+'.nc')

       


        if my_rank == 0:
            # ############### THIS IS NEW ######################
            # for rank in range(1, np_mpi):
            #     # If a worker did not receive termination, send it now
            #     # (You may want to track which ranks already got termination, but this works for most cases)
            #     comm.send(None, dest=rank, tag=0)
            #     # print(f"[MASTER] (Final sweep) Sending termination signal to rank {rank}")
            # ##################################################

            # print("[MASTER] Writing NetCDF file...")
            src = ','.join([i for i in sources])
            libRad.WriteNetcdf(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+SceneName + additional_spesifications + '.nc')
            # print("[MASTER] NetCDF file written.")

    if my_rank == 0:
        tt = datetime.now(timezone.utc) - start_time
        print("Run finished. It took {:.3f} hours \n\n".format(tt.total_seconds()/3600))

    MPI.Finalize()
    
