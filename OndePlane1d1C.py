####------OndePlane-----###
#Generation ondes planes et sommes
#auteur : Panayotis Akridas
#mail : pakridas@cyu.fr
#contributeur Claude de Anthropic AI pour consolider le code et ameliorer la representation graphique (9/5/2026 version non payante)
#date creation : 4 mai 2026
#version 4 : 10 mai 2026
###


import numpy
from matplotlib import pyplot

TWO_PI = 2 * numpy.pi

def Compute_length(x: float | numpy.ndarray | list) -> int:
    """Determine le nb d_elements dans la variable $x (1 si scalaire)."""
    if isinstance(x, (numpy.ndarray, list)):
        return len(x)
    return 1


def Get_simulation_type(x, t) -> str:
    """Information sur le type de simulation selon longueur de $x et $t"""
    len_x = Compute_length(x)
    len_t = Compute_length(t)
    if len_x > 1 and len_t == 1:
        msg = "espace" #"Calcul de l_onde plane pour x et t donnes (scalaires)"
    elif len_x == 1 and len_t > 1:
        msg = "temps" #"Calcul de l_onde plane pour t donne, sur un intervalle d'espace"
    elif len_x == 1 and len_t == 1:
        msg = "ici_maintenant mais sans interet"
    else:
        print("")
        print("         Desole JARVIS, mais je ne sais pas encore faire !")
        print("")
        exit()
#"Calcul de l_onde plane pour x donne, sur un intervalle de temps"
    return msg

def Check_momentum(k: float) -> float:
    """Verifie que le nombre d_onde est >0."""
    return abs(k)

def Compute_wavelength(k: float) -> float:
    """Calcule la long. onde  = 2*pi/k."""
    k = Check_momentum(k)
    return TWO_PI / k


def Compute_omega(k: float, speed: float = 1.0, dispersion_type: str = "linear") -> float:
    """
    Calcule la pulsation $omega a partir de la relation de dispersion (lineaire pour l_instant).
    Parameters
    ----------
    k : float
        Wave number [1/m].
    speed : float
        Phase velocity [speed] = L/T (used for linear dispersion).
    dispersion_type : {'linear', 'schrodinger'}
        Which relation to apply.

    Returns
    -------
    float
        pulsation $omega [omega] = 1/T.
    """
    k = Check_momentum(k)
    if dispersion_type == "linear":
        return speed * k
    elif dispersion_type == "schrodinger":
        hbar = 1.0546e-34   # [hbar] = M.L^2.T^{-1}
        mass = 9.109e-31    # [mass] = M
        return hbar * k**2 / (2 * mass)
    else:
        raise ValueError(f"Unknown dispersion_type: '{dispersion_type}'")


def Compute_plane_wave(k: float, x, t, dispersion_type: str = "linear") -> numpy.ndarray:
    """
    Calcule onde plane complexe en 1d.
    Parameters
    ----------
    k : float
        nombre d_ondes [k]=1/L.
    x : array-like
        Spatial positions [x]=L.
    t : float or array-like
        Time(s) [t] = T.
    dispersion_type : str
        Passed through to :func:`dispersion`.

    Returns
    -------
    np.ndarray
        Complex wave amplitude at each (x, t) point.
    """

    k = Check_momentum(k)
    omega = Compute_omega(k, dispersion_type=dispersion_type)
    x = numpy.asarray(x)
    t = numpy.asarray(t)
    phase = 1j * (omega * t - k * x)
    return numpy.exp(phase)
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def Check_oscillations(k: float, x: float, t: float) -> tuple[float, str]:
    """
    Count visible oscillations in [x_begin, x_end] and warn if too few/many.

    Returns
    -------
    (n_oscillations, message)
    """

    str_genre = Get_simulation_type(x,t)

    if str_genre == "espace":
        lam = Compute_wavelength(k)
        min_interval = min(x)
        max_interval = max(x)
    if str_genre == "temps":
        lam = Compute_period(k)
        min_interval = min(t)
        max_interval = max(t)
    print(str_genre)
    span = abs(max_interval - min_interval)
    
    n_osc = span / lam
    if span < lam:
        msg = "/!\\ Moins d'une oscillation visible -> agrandir l'intervalle ?"
    elif span > 6 * lam:
        msg = "/!\\ Plus de 6 oscillations visibles -> reduire l'intervalle ?"
    else:
        msg = ""

    return n_osc, msg


def Compute_period(k):
    omega = Compute_omega(k)
    period = TWO_PI/omega
    return period

def Compute_n_pts(k: float, factor: int = 500) -> int:
    """
    Suggest a number of spatial sample points for a pleasing plot.

    Parameters
    ----------
    k : float
        nombre d_onde [k] = 1L.
    factor : int
        Points per wavelength.
    """
    return int(factor * Compute_wavelength(k))


def Get_info(liste_k: list, x, t, n_waves: int) -> list[str]:
    """Build a human-readable summary of the simulation parameters."""

    lines = ["", f"Simulation de {n_waves} ondes planes"]

    min_k = min(liste_k)
    max_k = max(liste_k)

    str_genre = Get_simulation_type(x,t)

    if str_genre == "espace":
        str_variable = "x"
        min_interval = min(x)
        max_interval = max(x)
        n_pts = Compute_length(x)

    if str_genre == "temps":
        str_variable = "t"
        min_interval = min(t)
        max_interval = max(t)
        n_pts = Compute_length(t)
    print(str_variable)
    lines = [
            "", 
            f"Simulation de {n_waves} ondes planes :",
            f" - intervalle " + str_variable + " : " + str_genre + f"[{min_interval:.3g}, {max_interval:.3g}]",
            f" - avec {n_pts} points",
            f" - intervalle nombres d_ondes k : [{min_k:.4g}, {max_k:.4g}]",
            f" - instant t = {t}",
            ""]


    # if(len_x>1 and len_t==1):
    #     lines = lines + [
    #         f"  Intervalle espace : [{min(x):.3g} , {max(x):.3g}] "
    #         f"avec {len(x)} points",
    #         f"  Nombres d'ondes k : [{min(liste_k):.4g} , {max(liste_k):.4g}]",
    #         f"  Instant t         : {t}",
    #         ""]
    # if(len_x==11 and len_t>1):
    #     lines = lines + [
    #         f"  Intervalle espace : [{min(x):.3g} , {max(x):.3g}] "
    #         f"avec {len(x)} points",
    #         f"  Nombres d'ondes k : [{min(liste_k):.4g} , {max(liste_k):.4g}]",
    #         f"  Instant t         : {t}",
    #         ""]
    return lines


# ---------------------------------------------------------------------------
# Representation graphique
# ---------------------------------------------------------------------------

def Plot_waves(x, t, waves: numpy.ndarray, liste_k: list, liste_n_osc: list) -> None:
    """
    Plot each individual wave (real part) and their superposition.

    Parameters
    ----------
    plot_interval : array
        Spatial grid.
    waves : ndarray, shape (n_waves, n_pts)
        Complex wave amplitudes.
    liste_k : list
        Wave numbers (one per row in *waves*).
    liste_n_osc : list
        Number of oscillations per wave (for labels).
    """

    genre = Get_simulation_type(x, t)
    if genre ==  'espace':
        plot_interval = x
    if genre == 'temps':
        plot_interval = t

    n_waves = waves.shape[0]
    fig, axes = pyplot.subplots(n_waves + 1, 1,
                             figsize=(10, 2.5 * (n_waves + 1)),
                             sharex=True)
    fig.suptitle("Superposition d'ondes planes", fontsize=14)

    colors = pyplot.cm.viridis(numpy.linspace(0.1, 0.85, n_waves))

    for i, ax in enumerate(axes[:-1]):
        ax.plot(plot_interval, numpy.real(waves[i]), color=colors[i], linewidth=1.2)
        lam = Compute_wavelength(liste_k[i])
        ax.set_ylabel(f"onde {i}\n"
                      f"k={liste_k[i]:.3g}, λ={lam:.2f}\n"
                      f"n_osc={liste_n_osc[i]:.1f}",
                      fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    superposition = numpy.real(numpy.sum(waves, axis=0))
    axes[-1].plot(plot_interval, superposition, color="crimson", linewidth=1.5)
    axes[-1].set_ylabel("superposition", fontsize=8)
    axes[-1].set_xlabel("x [m]")
    axes[-1].tick_params(labelsize=8)
    axes[-1].grid(True, linewidth=0.4, alpha=0.5)

    pyplot.tight_layout()
    pyplot.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Wave number parameters -----------------------------------------
    k_centre = 1.0          # central wave number [1/m]
    delta_k  = k_centre / 10

    # --- Spatial grid ---------------------------------------------------
    x_ini, x_fin = 0, 100
    n_pts = Compute_n_pts(k_centre)
    x = numpy.linspace(x_ini, x_fin, n_pts)
    # --- Time instant ---------------------------------------------------
    t_ini = 1.0
    t = t_ini #numpy.linspace(x_ini, x_fin, n_pts)

    # --- Wave-number list -----------------------------------------------
    liste_k = [
        k_centre - 1.789 * delta_k,
        k_centre - 1.200 * delta_k,
        k_centre - 1.000 * delta_k,
        k_centre,
        k_centre + 1.000 * delta_k,
        k_centre + 1.200 * delta_k,
        k_centre + 1.789 * delta_k,
        k_centre + 2.050 * delta_k,
        k_centre - 2.103 * delta_k,
    ]
    n_waves = 3   # how many waves to actually use (≤ len(liste_k))

    # --- Print simulation type ------------------------------------------
    print(Get_simulation_type(x, t))

    # --- Print summary --------------------------------------------------
    for line in Get_info(liste_k[:n_waves], x, t, n_waves):
        print(line)

    # --- Generate waves -------------------------------------------------
    waves = numpy.zeros((n_waves, n_pts), dtype=complex) # creation d_un tableau rempli de 0, et de taille=$n_waves x $ n_pts 
    liste_n_osc: list[float] = []
    liste_msg:   list[str]   = []

    for i in range(n_waves):
        k_i = liste_k[i]
        waves[i] = Compute_plane_wave(k_i, x, t)
        n_osc, msg = Check_oscillations(k_i, x, t)
        liste_n_osc.append(n_osc)
        liste_msg.append(msg)

    # --- Per-wave report ------------------------------------------------
    for i in range(n_waves):
        lam = Compute_wavelength(liste_k[i])
        line = (f"Onde n°{i}  λ={lam:.2f} m  "
                f"n_osc={liste_n_osc[i]:.2f}")
        if liste_msg[i]:
            line += f"  {liste_msg[i]}"
        print(line)

    # --- Plot -----------------------------------------------------------
    Plot_waves(x, t,  waves, liste_k[:n_waves], liste_n_osc)
