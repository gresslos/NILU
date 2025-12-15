import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit   # non-linear fits

FONTSIZE=15
INFOSIZE=13
FIGSIZE=(8,7)


def poly_fit(x, y, deg=1):
    """
    Polynomial least-squares fit:
      y ≈ p_0*x^deg + ... + p_deg
    Returns:
      coeffs  - np.array of polynomial coefficients (highest power first)
      R2      - coefficient of determination
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Polynomial fit
    coeffs = np.polyfit(x, y, deg)
    y_hat = np.polyval(coeffs, x)

    # R^2 via correlation (user wants corrcoef)
    r = np.corrcoef(y, y_hat)[0, 1]
    R2 = r**2

    return coeffs, R2


# ---------- LOGARITHMIC + LINEAR: F(N) = a*ln(N+1) + b*N + c ----------
def log_lin_model(x, a, b, c):
    return a * np.log(x + 1.0) + b * x + c

def log_lin_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    p0 = [1.0, 0.0, np.mean(y)]   # initial guess
    popt, _ = curve_fit(log_lin_model, x, y, p0=p0, maxfev=10000)
    a, b, c = popt

    y_hat = log_lin_model(x, a, b, c)
    r = np.corrcoef(y, y_hat)[0, 1]
    R2 = r**2
    return (a, b, c), R2


# ---------- EXP-SAT + LINEAR: F(N) = A - B*exp(-k*N) + C*N ----------

def exp_sat_lin_model(x, A, B, k, C):
    return A - B * np.exp(-k * x) + C * x

def exp_sat_lin_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    A0 = y[-1]
    B0 = y[0] - y[-1]
    k0 = 1.0 / max(x.max(), 1.0)
    C0 = 0.0

    popt, _ = curve_fit(exp_sat_lin_model, x, y,
                        p0=[A0, B0, k0, C0], maxfev=20000)
    A, B, k, C = popt

    y_hat = exp_sat_lin_model(x, A, B, k, C)
    r = np.corrcoef(y, y_hat)[0, 1]
    R2 = r**2
    return (A, B, k, C), R2




#######################################################################################################
if __name__ == "__main__":
    SceneNames = ['Orbit_05378D',#0           # Marocco - Norway           # Previousy 'Arctic_05378D'
                  'Orbit_05458F',             # Chile
                  'Orbit_05926C',             # Old Greenland (13.06.2026)

                  'Orbit_06888C',#3           # Svalbard (14.08.2025)     
                  'Orbit_07277C',             # Svalbard (08.09.2025)

                  'Orbit_06518D',#5           # USA (21.07.2025)                
                  'Orbit_06907D',             # USA (15.08.2025)                        

                  'Orbit_06497E',#7           # Africa (20.07.2025)                            
                  'Orbit_06886E',             # Africa (14.08.2025)                             

                  'Orbit_06600C',#9           # Greenland (27.07.2025)          
                  'Orbit_06662C',             # Greenland (31.07.2025)   

                  'Orbit_06331C',#11          # Greenland (09.07.2025)       

                  'Orbit_07883D' #12          # Norway (FILEFJELL) 
    ]
    idx_scene = [5,8,12]
    orbits = SceneNames = [SceneNames[i] for i in idx_scene]

    property_want = [
        'bias',
        'std'
    ][0]

    
    
    buffers = ['MCIPA', 'MiniBuffer (5,5)', 'SmallBuffer (13x13)', 'MediumBuffer (13x13)', 'LargeBuffer (25x21)', 'MegaBuffer (31x31)', 'GigaBuffer (41x41)']
    pixels  = [   0,             5*5,               13*13,                 17*17,              25*21,               31*31,                  41*41      ]
    bias    = [   9.4,           6.6,                6.8,                   7.2,                 7.8,                 8.6,                    9.4        ]
    std     = [   66.9,          47.8,              41.9,                  41.7,                42.3,                44.0,                   46.5       ]

   
    idx_list = list(range(0, len(buffers)))
    pixels = np.asarray(pixels)[idx_list]
    bias   = np.asarray(bias)[idx_list]
    std    = np.asarray(std)[idx_list]
    buffers = [buffers[i] for i in idx_list]



    x = pixels
    xlabel = r"Buffer Area [pixels$^{2}$]"


    if property_want == 'bias':     y = bias; ylabel = r"$\Delta F$ Bias [W/m$^2$]"; title = "Bias [MYSTIC - BMAFLX] vs Buffer Size with Polynomial Fits"
    elif property_want == 'std':    y = std; ylabel = r"$\Delta F$ STD [W/m$^2$]"; title = "STD [MYSTIC - BMAFLX] vs Buffer Size with Polynomial Fits"
    
    
    want_pol = True
    want_log = True
    want_exp = True
   



    x_fit = np.linspace(x.min(), x.max(), 300)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
    plt.scatter(x, y, label="Data", zorder=5, c='r')



    



    # ---------- POLYNOMIAL FITS (deg 1–3) ----------
    if want_pol: 
        degrees = [1,2] #, 3, 4]
        for deg in degrees:
            coeffs, R2 = poly_fit(x, y, deg)
            y_fit = np.polyval(coeffs, x_fit)

             # print details
            print(f"\nPolynomial degree {deg}")
            print("Coefficients (highest power first):")
            a_list = coeffs[::-1] # reverse coeffs -> [a0, a1, a2, ...]
            for i, a in enumerate(a_list):
                print(f"  a_{i} = {a:.4f}")
            print(f"R² = {R2:.4f}")

            if deg == 1: plt.plot(x_fit, y_fit, label=f"deg {deg} y = {a_list[1]:.4f}x + {a_list[0]:.4f}   (R² = {R2:.4f})")
            else: plt.plot(x_fit, y_fit, label=f"deg {deg} (R² = {R2:.4f})")

           
        

    # ---------- LOG + LINEAR FIT ----------
    if want_log: 
        (a_log, b_log, c_log), R2_log = log_lin_fit(x, y)
        y_fit_log = log_lin_model(x_fit, a_log, b_log, c_log)
        ax.plot(x_fit, y_fit_log, '--', label=f"log+lin (R² = {R2_log:.2f})")

        print("\nLog+linear fit: F(N) = a*ln(N+1) + b*N + c")
        print(f"  a = {a_log:.2f}")
        print(f"  b = {b_log:.2f}")
        print(f"  c = {c_log:.2f}")
        print(f"R² = {R2_log:.4f}")

    # ---------- EXP-SAT + LINEAR FIT ----------
    if want_exp: 
        (A_exp, B_exp, k_exp, C_exp), R2_exp = exp_sat_lin_fit(x, y)
        y_fit_exp = exp_sat_lin_model(x_fit, A_exp, B_exp, k_exp, C_exp)
        ax.plot(x_fit, y_fit_exp, ':', label=f"exp-sat+lin (R² = {R2_exp:.2f})")

        print("\nExp-sat+linear fit: F(N) = A - B*exp(-k*N) + C*N")
        print(f"  A = {A_exp:.2f}")
        print(f"  B = {B_exp:.2f}")
        print(f"  k = {k_exp:.4f}")
        print(f"  C = {C_exp:.4f}")
        print(f"R² = {R2_exp:.4f}")















    # ---------- PLOTTING STYLE ----------
    ax.set_xlabel(xlabel, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel, fontsize=INFOSIZE)

    # # Set plotting limits
    # ax.set_ylim(ymin,ymax)
    # ax.set_xlim(xmin, xmax)

    ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
    ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98, fontweight='bold')
    # remove the "Orbit_" prefix and join nicely
    orbit_names = [o.replace("Orbit_", "") for o in orbits]
    ax.set_title(f"Orbits: {', '.join(orbit_names)}", fontsize=INFOSIZE*.8)
  
    ax.legend(  
                # loc='upper left', 
                framealpha=0.7, 
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

    png_name = 'Data/figures/BufferSize_Regression.png'
    plt.savefig(png_name)
    plt.close()
    print(f"\nPlot saved to {png_name}")




