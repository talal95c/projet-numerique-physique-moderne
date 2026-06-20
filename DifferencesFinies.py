####------DifferencesFinies-----###
# Resolution numerique de l equation de Schrodinger par differences finies
# Methode : Euler explicite avec schema leapfrog (separation Re/Im)
# Auteur : Groupe projet
# Date : juin 2026
# Phase 3 : Validation particule libre (V0 = 0)
# Phase 4 : Barriere de potentiel et temps de franchissement
###

import numpy
from numpy import pi, sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PaquetOndeGauss1d1C import (Compute_gaussian_wp, Compute_group_velocity,
                         Compute_analytic_gaussian_position,
                         Compute_analytic_gaussian_spreading,
                         Check_normalization, K0, SIGMA, HBAR, MASS)


# ─── Parametres de la grille ───────────────────────────────────────────────

X_MIN, X_MAX = -40, 120       # intervalle d espace (assez grand pour eviter les bords)
N_X          = 4000            # nombre de points en espace
dx           = (X_MAX - X_MIN) / (N_X - 1)

# Condition de stabilite : s = hbar*dt / (2*m*dx^2) < 0.5
# => dt < m*dx^2 / hbar
S_TARGET = 0.25                # parametre s cible (< 0.5 pour stabilite)
dt       = S_TARGET * 2 * MASS * dx**2 / HBAR
s        = HBAR * dt / (2 * MASS * dx**2)

x = numpy.linspace(X_MIN, X_MAX, N_X)


# ─── Fonctions derivees ────────────────────────────────────────────────────

def Compute_first_derivative(f, dx):
    """
    Calcule la derivee premiere par differences centrees.
    Retourne un tableau de meme taille que f.
    Aux bords : differences avant/arriere.
    """
    df = numpy.zeros_like(f)
    # Differences centrees pour les points interieurs
    df[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    # Bords
    df[0]  = (f[1] - f[0]) / dx
    df[-1] = (f[-1] - f[-2]) / dx
    return df


def Compute_second_derivative(f, dx):
    """
    Calcule la derivee seconde par differences centrees.
    f''(xi) = (f(xi+1) - 2*f(xi) + f(xi-1)) / dx^2
    """
    d2f = numpy.zeros_like(f)
    d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    return d2f


def Test_derivatives():
    """
    Test des derivees avec f(x) = x^2.
    Derivee premiere attendue : 2x
    Derivee seconde attendue : 2
    """
    x_test = numpy.linspace(-5, 5, 1000)
    dx_test = x_test[1] - x_test[0]
    f_test = x_test**2

    df_num = Compute_first_derivative(f_test, dx_test)
    df_exact = 2 * x_test

    d2f_num = Compute_second_derivative(f_test, dx_test)
    d2f_exact = 2.0 * numpy.ones_like(x_test)

    # Erreur relative (en evitant la division par zero au centre)
    mask = numpy.abs(df_exact) > 1e-10
    err_d1 = numpy.max(numpy.abs(df_num[mask] - df_exact[mask]) / numpy.abs(df_exact[mask]))

    err_d2 = numpy.max(numpy.abs(d2f_num[2:-2] - d2f_exact[2:-2]))

    print(f"Test derivee premiere : erreur relative max = {err_d1:.2e}")
    print(f"Test derivee seconde  : erreur absolue max  = {err_d2:.2e}")
    return err_d1, err_d2


# ─── Potentiel ──────────────────────────────────────────────────────────────

def Compute_barrier_potential(x, x_barrier, a_barrier, V0):
    """
    Potentiel barriere rectangulaire.
    V(x) = V0 si x_barrier <= x <= x_barrier + a_barrier
    V(x) = 0  sinon
    """
    V = numpy.zeros_like(x)
    V[(x >= x_barrier) & (x <= x_barrier + a_barrier)] = V0
    return V


# ─── Solveur de Schrodinger ─────────────────────────────────────────────────

def Solve_schrodinger(x, dt, n_steps, psi_init, V, save_every=10):
    """
    Resolution de l equation de Schrodinger par differences finies.
    Methode : schema leapfrog (mise a jour alternee Re/Im).

    Parameters
    ----------
    x : array
        Grille spatiale.
    dt : float
        Pas de temps.
    n_steps : int
        Nombre de pas de temps.
    psi_init : array (complex)
        Fonction d onde initiale.
    V : array
        Potentiel V(x).
    save_every : int
        Sauvegarder l etat tous les save_every pas.

    Returns
    -------
    times : array
        Instants sauvegardes.
    densities : array (n_saved, n_x)
        Densite de probabilite a chaque instant sauvegarde.
    norms : array
        Norme a chaque instant sauvegarde.
    positions : array
        Position du maximum a chaque instant sauvegarde.
    """
    n_x = len(x)
    dx_local = x[1] - x[0]
    s_local = HBAR * dt / (2 * MASS * dx_local**2)

    if s_local >= 0.5:
        print(f"ATTENTION : s = {s_local:.4f} >= 0.5 => instabilite probable !")
        print(f"  Reduire dt ou augmenter N_X")

    # Separation partie reelle et imaginaire
    re = numpy.real(psi_init).copy()
    im = numpy.imag(psi_init).copy()

    # Coefficient du potentiel
    V_coeff = dt * V / HBAR

    # Stockage des resultats
    n_saved = n_steps // save_every + 1
    times = numpy.zeros(n_saved)
    densities = numpy.zeros((n_saved, n_x))
    norms = numpy.zeros(n_saved)
    positions = numpy.zeros(n_saved)

    # Etat initial
    density = re**2 + im**2
    densities[0] = density
    norms[0] = Check_normalization(density, x)
    positions[0] = x[numpy.argmax(density)]
    idx_save = 1

    # Boucle temporelle
    for j in range(1, n_steps + 1):

        # Schema FDTD : mise a jour alternee
        # 1) Mettre a jour Im avec Re actuel
        im[1:-1] = (im[1:-1]
                    + s_local * (re[2:] + re[:-2] - 2 * re[1:-1])
                    - V_coeff[1:-1] * re[1:-1])

        # 2) Calculer la densite (avec Re ancien et Im nouveau)
        # density = re**2 + im * im_old  # approximation symplectique

        # 3) Mettre a jour Re avec Im nouveau
        re[1:-1] = (re[1:-1]
                    - s_local * (im[2:] + im[:-2] - 2 * im[1:-1])
                    + V_coeff[1:-1] * im[1:-1])

        # Conditions aux limites (Dirichlet)
        re[0] = re[-1] = 0.0
        im[0] = im[-1] = 0.0

        # Sauvegarde
        if j % save_every == 0 and idx_save < n_saved:
            density = re**2 + im**2
            densities[idx_save] = density
            norms[idx_save] = Check_normalization(density, x)
            positions[idx_save] = x[numpy.argmax(density)]
            times[idx_save] = j * dt
            idx_save += 1

    return times[:idx_save], densities[:idx_save], norms[:idx_save], positions[:idx_save]


# ─── Validation V0 = 0 ──────────────────────────────────────────────────────

def Validate_free_particle(t_max=30.0):
    """
    Compare le solveur numerique avec la solution analytique
    pour une particule libre (V0 = 0).
    """
    print("=" * 60)
    print("VALIDATION : Particule libre (V0 = 0)")
    print("=" * 60)
    print(f"  Parametres : hbar={HBAR}, m={MASS}, k0={K0}, sigma={SIGMA}")
    print(f"  Grille : [{X_MIN}, {X_MAX}], N_X={N_X}, dx={dx:.6f}")
    print(f"  Temps : dt={dt:.6e}, s={s:.4f}")

    # Condition initiale : paquet d ondes gaussien a t=0
    psi_init = Compute_gaussian_wp(x, 0.0)

    # Potentiel nul
    V = numpy.zeros_like(x)

    # Nombre de pas
    n_steps = int(t_max / dt)
    save_every = max(1, n_steps // 200)

    print(f"  Nombre de pas : {n_steps}, sauvegarde tous les {save_every} pas")
    print("  Calcul en cours...")

    times, densities, norms, positions = Solve_schrodinger(
        x, dt, n_steps, psi_init, V, save_every=save_every
    )

    print(f"  Termine. {len(times)} etats sauvegardes.")

    # Comparaison avec la solution analytique
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Validation : particule libre (V0 = 0)", fontsize=14)

    # 1. Densite a differents instants
    ax = axes[0, 0]
    t_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    for idx in t_indices:
        t_val = times[idx]
        # Solution numerique
        ax.plot(x, densities[idx], '-', label=f't={t_val:.1f} (num)', alpha=0.8)
        # Solution analytique
        psi_ana = Compute_gaussian_wp(x, t_val)
        density_ana = numpy.abs(psi_ana)**2
        ax.plot(x, density_ana, '--', label=f't={t_val:.1f} (ana)', alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("|Ψ|²")
    ax.set_title("Densite de probabilite")
    ax.legend(fontsize=7)
    ax.set_xlim(-30, 80)

    # 2. Conservation de la norme
    ax = axes[0, 1]
    ax.plot(times, norms, 'b-')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Norme = 1')
    ax.set_xlabel("t")
    ax.set_ylabel("Norme")
    ax.set_title("Conservation de la norme")
    ax.legend()

    # 3. Position du maximum
    ax = axes[1, 0]
    pos_analytique = [Compute_analytic_gaussian_position(t) for t in times]
    ax.plot(times, positions, 'b-', label='Numerique')
    ax.plot(times, pos_analytique, 'r--', label='Analytique (vg*t)')
    ax.set_xlabel("t")
    ax.set_ylabel("Position du max")
    ax.set_title("Position du maximum")
    ax.legend()

    # 4. Erreur relative sur la position
    ax = axes[1, 1]
    pos_ana_arr = numpy.array(pos_analytique)
    mask = pos_ana_arr > 1.0  # eviter division par zero
    if numpy.any(mask):
        erreur_rel = numpy.abs(positions[mask] - pos_ana_arr[mask]) / pos_ana_arr[mask]
        ax.plot(times[mask], erreur_rel, 'g-')
        ax.set_xlabel("t")
        ax.set_ylabel("Erreur relative")
        ax.set_title("Erreur relative sur la position")
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig("validation_V0_0.png", dpi=150)
    plt.close()
    print("  Figure sauvegardee : validation_V0_0.png")

    # Affichage des metriques
    print(f"\n  Norme initiale : {norms[0]:.6f}")
    print(f"  Norme finale   : {norms[-1]:.6f}")
    print(f"  Derive de norme: {abs(norms[-1] - norms[0]):.6e}")

    return times, densities, norms, positions


# ─── Simulation barriere de potentiel ───────────────────────────────────────

def Simulate_barrier(V0=0.5, a_barrier=5.0, x_barrier=30.0, t_max=80.0):
    """
    Simulation d un paquet d ondes gaussien rencontrant
    une barriere rectangulaire de potentiel.

    Parameters
    ----------
    V0 : float
        Hauteur de la barriere.
    a_barrier : float
        Largeur de la barriere.
    x_barrier : float
        Position du debut de la barriere.
    t_max : float
        Temps maximum de simulation.
    """
    print("=" * 60)
    print(f"SIMULATION : Barriere de potentiel")
    print(f"  V0 = {V0}, a = {a_barrier}, position = {x_barrier}")
    print("=" * 60)

    # Energie cinetique moyenne du paquet
    E_moy = HBAR**2 * K0**2 / (2 * MASS)
    print(f"  Energie moyenne du paquet : E = {E_moy:.4f}")
    if E_moy > V0:
        print(f"  Regime : E > V0 (transmission classique possible)")
    else:
        print(f"  Regime : E < V0 (effet tunnel)")

    # Condition initiale
    psi_init = Compute_gaussian_wp(x, 0.0)

    # Potentiel barriere
    V = Compute_barrier_potential(x, x_barrier, a_barrier, V0)

    # Resolution
    n_steps = int(t_max / dt)
    save_every = max(1, n_steps // 400)

    print(f"  Nombre de pas : {n_steps}")
    print("  Calcul en cours...")

    times, densities, norms, positions = Solve_schrodinger(
        x, dt, n_steps, psi_init, V, save_every=save_every
    )

    print(f"  Termine. {len(times)} etats sauvegardes.")

    return times, densities, norms, positions, V


def Compute_tunneling_time(x, times, densities, V, x_barrier, a_barrier):
    """
    Determine le temps de franchissement de la barriere
    en mesurant quand le maximum du paquet transmis
    depasse la position x_barrier + a_barrier.
    """
    x_sortie = x_barrier + a_barrier

    # On cherche le premier instant ou le maximum est au-dela de la barriere
    for i in range(len(times)):
        # Densite a droite de la barriere
        mask_droit = x > x_sortie
        if numpy.any(mask_droit):
            density_droit = densities[i][mask_droit]
            if numpy.max(density_droit) > 0.001 * numpy.max(densities[0]):
                # Le maximum a droite de la barriere
                idx_max_droit = numpy.argmax(density_droit)
                x_max_droit = x[mask_droit][idx_max_droit]
                return times[i], x_max_droit

    return None, None


def Measure_traversal_times(V0_values=None, a_values=None):
    """
    Etude parametrique du temps de franchissement.
    """
    if V0_values is None:
        V0_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if a_values is None:
        a_values = [2.0, 5.0, 8.0, 10.0]

    x_barrier = 30.0
    t_max = 100.0

    print("\n" + "=" * 60)
    print("ETUDE PARAMETRIQUE : temps de franchissement")
    print("=" * 60)

    # Etude en fonction de V0 (a fixe)
    a_fixed = 5.0
    tau_vs_V0 = []
    print(f"\n--- Influence de V0 (a = {a_fixed}) ---")
    for V0 in V0_values:
        times, densities, norms, positions, V = Simulate_barrier(
            V0=V0, a_barrier=a_fixed, x_barrier=x_barrier, t_max=t_max
        )
        tau, x_max = Compute_tunneling_time(x, times, densities, V, x_barrier, a_fixed)
        tau_vs_V0.append(tau)
        print(f"  V0 = {V0:.2f} => tau = {tau}")

    # Etude en fonction de a (V0 fixe)
    V0_fixed = 0.3
    tau_vs_a = []
    print(f"\n--- Influence de a (V0 = {V0_fixed}) ---")
    for a_val in a_values:
        times, densities, norms, positions, V = Simulate_barrier(
            V0=V0_fixed, a_barrier=a_val, x_barrier=x_barrier, t_max=t_max
        )
        tau, x_max = Compute_tunneling_time(x, times, densities, V, x_barrier, a_val)
        tau_vs_a.append(tau)
        print(f"  a = {a_val:.1f} => tau = {tau}")

    return V0_values, tau_vs_V0, a_values, tau_vs_a


def Plot_barrier_simulation(V0=0.3, a_barrier=5.0, x_barrier=30.0, t_max=80.0):
    """
    Representation graphique de la simulation avec barriere.
    """
    times, densities, norms, positions, V = Simulate_barrier(
        V0=V0, a_barrier=a_barrier, x_barrier=x_barrier, t_max=t_max
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Paquet d ondes et barriere (V0={V0}, a={a_barrier})", fontsize=14)

    # 1. Densite a differents instants
    ax = axes[0, 0]
    t_indices = numpy.linspace(0, len(times)-1, 8, dtype=int)
    colors = plt.cm.viridis(numpy.linspace(0.1, 0.9, len(t_indices)))
    for c_idx, idx in enumerate(t_indices):
        ax.plot(x, densities[idx], color=colors[c_idx],
                label=f't={times[idx]:.1f}', alpha=0.8)
    # Barriere (zone grisee)
    ax.axvspan(x_barrier, x_barrier + a_barrier, alpha=0.2, color='red',
               label=f'Barriere V0={V0}')
    ax.set_xlabel("x")
    ax.set_ylabel("|Ψ|²")
    ax.set_title("Evolution de la densite")
    ax.legend(fontsize=7)
    ax.set_xlim(X_MIN, X_MAX)

    # 2. Conservation de la norme
    ax = axes[0, 1]
    ax.plot(times, norms, 'b-')
    ax.axhline(y=1.0, color='r', linestyle='--')
    ax.set_xlabel("t")
    ax.set_ylabel("Norme")
    ax.set_title("Conservation de la norme")

    # 3. Position du maximum
    ax = axes[1, 0]
    vg = Compute_group_velocity()
    ax.plot(times, positions, 'b-', label='Numerique')
    ax.plot(times, vg * times, 'g--', label=f'Libre (vg*t)', alpha=0.5)
    ax.axhline(y=x_barrier, color='r', linestyle=':', label='Debut barriere')
    ax.axhline(y=x_barrier + a_barrier, color='r', linestyle='--',
               label='Fin barriere')
    ax.set_xlabel("t")
    ax.set_ylabel("Position du max")
    ax.set_title("Trajectoire du maximum")
    ax.legend(fontsize=8)

    # 4. Probabilite transmise et reflechie
    ax = axes[1, 1]
    prob_trans = []
    prob_refl = []
    for i in range(len(times)):
        mask_trans = x > x_barrier + a_barrier
        mask_refl = x < x_barrier
        pt = numpy.trapezoid(densities[i][mask_trans], x[mask_trans]) if numpy.any(mask_trans) else 0
        pr = numpy.trapezoid(densities[i][mask_refl], x[mask_refl]) if numpy.any(mask_refl) else 0
        prob_trans.append(pt)
        prob_refl.append(pr)
    ax.plot(times, prob_trans, 'g-', label='P(transmission)')
    ax.plot(times, prob_refl, 'r-', label='P(reflexion)')
    ax.plot(times, numpy.array(prob_trans) + numpy.array(prob_refl), 'b--',
            label='Total', alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("Probabilite")
    ax.set_title("Transmission et reflexion")
    ax.legend()

    plt.tight_layout()
    plt.savefig("simulation_barriere.png", dpi=150)
    plt.close()
    print("  Figure sauvegardee : simulation_barriere.png")

    return times, densities, norms, positions


# ─── Animation ──────────────────────────────────────────────────────────────

def Animate_barrier(V0=0.3, a_barrier=5.0, x_barrier=30.0, t_max=80.0):
    """
    Animation de la propagation du paquet d ondes a travers la barriere.
    """
    times, densities, norms, positions, V = Simulate_barrier(
        V0=V0, a_barrier=a_barrier, x_barrier=x_barrier, t_max=t_max
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Effet tunnel : V0={V0}, a={a_barrier}", fontsize=13)

    y_max = numpy.max(densities[0]) * 1.3
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("|Ψ(x,t)|²")

    # Barriere
    ax.axvspan(x_barrier, x_barrier + a_barrier, alpha=0.15, color='red')
    ax.axvline(x=x_barrier, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=x_barrier + a_barrier, color='red', linestyle=':', alpha=0.5)

    line, = ax.plot(x, densities[0], 'b-', lw=2)
    time_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=10)
    norm_text = ax.text(0.02, 0.84, '', transform=ax.transAxes, fontsize=9, color='gray')

    def update(frame):
        line.set_ydata(densities[frame])
        time_text.set_text(f't = {times[frame]:.2f}')
        norm_text.set_text(f'norme = {norms[frame]:.4f}')
        return line, time_text, norm_text

    ani = animation.FuncAnimation(fig, update, frames=len(times),
                                  interval=30, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Test des derivees
    print("\n>>> Test des derivees")
    Test_derivatives()

    # Validation particule libre
    print("\n>>> Validation particule libre (V0 = 0)")
    Validate_free_particle(t_max=30.0)

    # Simulation avec barriere
    print("\n>>> Simulation barriere de potentiel")
    Plot_barrier_simulation(V0=0.3, a_barrier=5.0, x_barrier=30.0, t_max=80.0)

    # Pour l animation (decommentez) :
    # Animate_barrier(V0=0.3, a_barrier=5.0, x_barrier=30.0, t_max=80.0)
