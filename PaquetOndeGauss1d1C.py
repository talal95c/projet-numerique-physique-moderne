####------PaquetOndesGaussien-----###
#Evolution d_un paquet d_ondes gaussien quantique (relation de dispersion Sch)
#auteur : Panayotis Akridas
#mail : pakridas@cyu.fr
#contributeur : Claude de Anthropic AI exclusivement pour la partie graphique (animation) (16/5/2026 version non payante)
#date creation : 16 mai 2026
#version 2 : 17 mai 2026
#todo : beaucoup de verification et d_amelioration pour eviter des paquets d_ondes non normalises
###

import numpy
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import scipy

K0    = 1   # nombre d_onde central
SIGMA = 0.2   # largeur gaussienne dans l_espace des k
X0    = -20   # position initiale du paquet d_onde
HBAR  = 1#   # cte de Planck
HBAR2 = HBAR*HBAR 
MASS  = 1 #masse de la particule
HALF = .5

# Intervalle espace
X_MIN, X_MAX = -20, 100
N_X          = 2000
x            = numpy.linspace(X_MIN, X_MAX, N_X)


def Compute_re_norm_gaussian(x, mu, sigma):
    """
    Calcul d_une gaussienne de moyenne (centree sur) mu et d_ecart-type sigma a l_aide d_une fonction scipy
    Remarque  : ne fonctionne pas si les parametres sont complexes
    Interet : sans doute assez faible mais renvoie une gaussienne toujours normalisee a 1
    """
    return norm.pdf(x, mu, sigma)

def Compute_gaussian(x, mu, sigma):
    """
    Calcul d_une gaussienne non normalisee a 1, de moyenne (centree sur) mu et d_ecart-type sigma
    !!! Remarque 1 : non normalisee a 1
    Remarque 2 : fonctionne avec des complexes
    A faire 1 : a normaliser ?
    A faire 2 : verifier que pour x reel le resultat est coherent avec Compute_re_norm_gaussian
    """
    return numpy.exp(-(x-mu)**2/(2 * sigma**2))

def Compute_phase_velocity(k0 = K0):
    """
    Calcule la vitesse de phase
    """
    return HBAR*k0/(2*MASS)

def Compute_group_velocity(k0 = K0):
    """
    Calcule la vitesse de groupe
    """
    return Compute_phase_velocity(2*k0)

def Check_normalization(function_array, x_interval):
    """
    Calcule l_integrane de $function_array
    doit renvoyer 1 si $function_array est une densite de probabilite
    Methode : integration numerique par la methode de simpson (peut etre lourd numeriquement mais plus precis)
    """
    x_min = min(x_interval)
    x_max = max(x_interval)
    return scipy.integrate.simpson(function_array, x_interval)


def Compute_gaussian_wp_as_cct(x, t, sigma=SIGMA, k0=K0, x0=X0):
    """
        Calcule l_amplitude de probabilite d_un paquet d_onde gaussien au cours du temps avec l_expression de CCT [(16-a) p.61]
        Remarque 1 : un terme en exp(i $phi) est absent => sans importance sur la densite
        Remarque 2 : pas utilise
    """
    a = 1/sigma
    a2 = a**2
    terme = 2*1j*HBAR*t/MASS
    amplitude = numpy.power(2*a2/(pi*(a2**2 + 4*HBAR2*t**2/(MASS**2))),.25)
    mu = HBAR*k0*t/MASS
    new_sigma = numpy.sqrt((a2+2*1j*HBAR*t/MASS)/2)
    return amplitude * numpy.exp(1j*k0*x) * Compute_gaussian(x, mu, new_sigma)

def Compute_gaussian_wp(x, t, sigma=SIGMA, k0=K0, x0=X0):
    """
        Calcule l_amplitude de probabilite d_un paquet d_onde gaussien au cours du temps avec l_expression donnee en TD
    """
    a = 1/sigma
    a2 = a**2
    terme = MASS*a2+2*1j*HBAR*t
    inv_terme = 1/terme
    amplitude = numpy.sqrt(2*MASS*a*inv_terme/(numpy.sqrt(2*pi)))
    return amplitude * numpy.exp(inv_terme*MASS*(a2*k0+2*1j*x)**2/4) * Compute_gaussian(k0,0,numpy.sqrt(2)/a)



def Compute_ini_gaussian2_wp_as_cct(x, sigma=SIGMA, k0=K0):
    """
        Calcule la densite de probabilite *initiale* du paquet d_onde gaussien avec l_expression de CCT [(10) p.60]
        Interet : verification/debug
    """
    sigma2 = sigma**2
    phase = numpy.exp(1j*k0*x)
    arg_exp = -sigma2*x**2    
    amplitude = numpy.power(2*sigma2/pi,0.25)
    return amplitude*phase*numpy.exp(arg_exp)

def Compute_analytic_gaussian_position(t, k0 = K0):
    """
        Calcul analytique de la position moyenne de la particule libre <=> max du paquet d_ondes gaussienn avec l_expression de CCT [(19) p.62]
        Remarque : facilement demontrable a partir de l_expression generale de la densite gaussienne => pas de ref explicite a CCT
        Interet : verification/debug
    """
    return Compute_group_velocity(k0)*t

def Compute_analytic_gaussian_spreading(t, sigma=SIGMA):
    """
        Calcul analytique de la dispersion de la densite de probabilite 
        Interet : verification/debug
    """ 
    return HALF/sigma * numpy.sqrt(1+(2*HBAR*t*sigma**2/MASS)**2)

def Compute_analytic_gaussian_height(t, sigma=SIGMA):
    """
        Calcul analytique de la dispersion de la densite de probabilite 
        Interet : verification/debug
    """ 
    a=1/sigma
    a2 = a**2
    return numpy.sqrt(2/(pi*a2*(1+(2*HBAR*t/(MASS*a2))**2)))

def Compute_numeric_height(array_function):
    """
        Calcul numerique de la hauteur de la densite de probabilite
        Remarque : sans interet si array est complex
        Interet : verification/debug
    """
    return max(array_function)


def Compute_numeric_position(array_function, x):
    """
        Calcul numerique de la position moyenne de la particule libre directement a partir de la densite <=> max du paquet d_ondes gaussien 
        Remarque 1 : $array_function doit etre la densite de probabilite
        Remarque 2 : fonction valable pour toute densite unimodale <=> un seul max
        Remarque 3 : peut differer de Compute_analytic si $x n_est pas assez grand
        Interet : verification/debug
    """
    index_max = numpy.argmax(array_function) 
    return x[index_max]

def Compute_numeric_moment(array_function, x, moment):
    """
        Calcul numerique du moment de $array_function
        Interet 1 : fonction generale utilisee dans Compute_numeric_spreading
        Remarque : sans interet pour une gaussienne puisque elle est entierement determinee a partir de la moyenne et de sigma !
    """
    array_moment = array_function*numpy.power(x, moment)
    return Check_normalization(array_moment, x)

def Compute_numeric_spreading(array_function, x):
    """
        Calcul numerique de la dispersion de la particule libre directement a partir de la densite <=> max du paquet d_ondes 
        Remarque 1 : $array_function doit etre la densite de probabilite
        Remarque 2 : fonction valable pour toute densite unimodale <=> un seul max
        Remarque 3 : peut differer de Compute_analytic si $x n_est pas assez grand
        Interet : verification/debug 
    """    
    mean_x  = Compute_numeric_moment(array_function, x, 1)
    mean_x2 = Compute_numeric_moment(array_function, x, 2)
    spreading = numpy.sqrt(mean_x2 - mean_x**2) 
    return spreading


# ── Animation ──────────────────────────────────────────────────────────────────

def animate_wavepacket(t_max=30.0, n_frames=600, interval=10000000, kind=2):
    """
    Remarque : peut etre utilise pour representer la partie reelle et imaginaire de psi. Pas le temps d_ameliorer cette partie
    """
    times = numpy.linspace(0, t_max, n_frames)

    #Calcul des donnees pour toutes les images
    print("Calcul des donnees pour toutes les images")
    all_norm      = numpy.empty(n_frames)
    all_position  = numpy.empty(n_frames)    
    all_spreading = numpy.empty(n_frames)
    all_height    = numpy.empty(n_frames)
    all_density   = numpy.empty((n_frames, len(x)))    

    if kind>1:
        all_re_psi = numpy.empty((n_frames, len(x)))
        all_im_psi = numpy.empty((n_frames, len(x)))


    for i, t_val in enumerate(times):
        psi         = Compute_gaussian_wp(x, t_val)
        density     = numpy.abs(psi)**2

        all_density[i]   = density
        all_norm[i]      = Check_normalization(density, x)
        all_position[i]  = Compute_analytic_gaussian_position(t_val)
        all_spreading[i] = Compute_analytic_gaussian_spreading(t_val)
        all_height[i]    = Compute_analytic_gaussian_height(t_val)

        if kind>1:
            all_re_psi[i] = numpy.real(psi)
            all_im_psi[i] = numpy.imag(psi)

    print(f"Done.  norm at t=0: {all_norm[0]:.6f},  at t={t_max}: {all_norm[-1]:.6f}")

    # Calcul des limites du graph
    y_max_density = all_density[0].max() * 1.5
    if kind>1:
        y_max_re = numpy.abs(all_re_psi[0]).max() * 1.5
        y_max_im = numpy.abs(all_re_psi[0]).max() * 1.5
    if kind == 1:
        fig, ax_density = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    if kind == 2:
        fig, (ax_density, ax_re) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    if kind == 3:
        fig, (ax_density, ax_re, ax_im) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Evolution du paquet d_onde gaussien", fontsize=13)

    (line_density,) = ax_density.plot(x, all_density[0], color="#3266ad", lw=2)
#    (line_density2,) = ax_density.plot(x, all_density_pas[0], color="#ad5432", lw=1.25)
    ax_density.set_ylim(0, y_max_density)
    ax_density.set_xlim(X_MIN, X_MAX)
    ax_density.set_ylabel(r"$|\psi(x,t)|^2$", fontsize=11)
    time_text      = ax_density.text(0.02, 0.90, "temps t = 0.00",
                         transform=ax_density.transAxes, fontsize=10, color="#3266ad")
    norm_text      = ax_density.text(0.02, 0.76, f"norme (t) = {all_norm[0]:.4f}",
                         transform=ax_density.transAxes, fontsize=9,  color="gray")
    position_text  = ax_density.text(0.1, 0.90, "poistion du max : x0 = 0.00",
                         transform=ax_density.transAxes, fontsize=10, color="#3266ad")
    spreading_text = ax_density.text(0.3, 0.90, "dispersion : Delta x = 0.00",
                         transform=ax_density.transAxes, fontsize=10, color="#3266ad")
    height_text = ax_density.text(0.6, 0.90, "hauteur = 0.00",
                         transform=ax_density.transAxes, fontsize=10, color="#3266ad")

    # Si partie reele et/ou imag
      
    if kind >1 :
        (line_re,) = ax_re.plot(x, all_re_psi[0], color="#1D9E75", lw=2)
        ax_re.axhline(0, color="gray", lw=0.5, ls="--")
        ax_re.set_ylim(-y_max_re, y_max_re)
        ax_re.set_xlim(X_MIN, X_MAX)
        ax_re.set_ylabel(r"Re$[\psi(x,t)]$", fontsize=11)
        ax_re.set_xlabel("x", fontsize=12)
        if kind >2:
            (line_im,) = ax_im.plot(x, all_re_psi[0], color="#1D9E75", lw=2)
            ax_im.axhline(0, color="gray", lw=0.5, ls="--")
            ax_im.set_ylim(-y_max_im, y_max_im)
            ax_im.set_xlim(X_MIN, X_MAX)
            ax_im.set_ylabel(r"Im$[\psi(x,t)]$", fontsize=11)
            ax_im.set_xlabel("x", fontsize=12)

    plt.tight_layout()

    def update(frame):
        line_density.set_ydata(all_density[frame])
 #       line_density2.set_ydata(all_density_pas[frame])
        if kind > 1:
            line_re.set_ydata(all_re_psi[frame])
            if kind >2:
                line_im.set_ydata(all_im_psi[frame])
                
        time_text.set_text(f"t = {times[frame]:.2f}")
        norm_text.set_text(f"norme (t) = {all_norm[frame]:.4f}")
        position_text.set_text(f"poistion du max : x0 = {all_position[frame]:.4f}")
        spreading_text.set_text(f"dispersion : Delta x = {all_spreading[frame]:.4f}")
        height_text.set_text(f"hauteur = {all_height[frame]:.4f}")
        if kind == 1:
            return line_density, time_text, norm_text, position_text, spreading_text, height_text
        if kind == 2:    
            return line_density, line_re, time_text, norm_text, position_text, spreading_text
        if kind == 3:
            return line_density, line_re, line_im, time_text, norm_text, position_text, spreading_text
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=True,
        repeat=True,
    )

    # To save as MP4 (needs ffmpeg):
    # ani.save("wavepacket.mp4", writer="ffmpeg", dpi=150)

    plt.show()


if __name__ == "__main__":

    animate_wavepacket(t_max=40, n_frames=2000, interval=100, kind=1)
