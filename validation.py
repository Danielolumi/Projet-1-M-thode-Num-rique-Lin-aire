"""
validation.py
=============
Auteur : Daniel Olumi — GCVIV0185-7 Méthodes numériques linéaires, ULiège 2026-2027

Tests objectifs et structurés du code :

  1. precision_vitesses   : ordre de précision des vitesses déduites du potentiel
  2. translation_canal    : canal rectiligne, vitesse uniforme -> solution exacte = translation
  3. convergence          : erreur en fonction de dx -> ordre de précision du schéma
  4. stabilite_cfl_2d     : advection en diagonale, CFL 2D juste en dessous / au dessus de 1
  5. amplification       : facteur d'amplification théorique (analyse de Von Neumann)
  6. comparer_reference_prof : l'appel au code du professeur reproduit son out_pot.npy
"""

import numpy as np

from domaine import Domaine, domaine_carre_ouvert
from potentiel import potentiel_depuis_matrice
from vitesses import vitesses_aux_bords
from volumes_finis import simuler


# ----------------------------------------------------------------------
# 1. Précision des vitesses déduites du potentiel
# ----------------------------------------------------------------------
def precision_vitesses(liste_n=(10, 20, 40, 80, 160)):
    """phi exact = exp(x) cos(y) (fonction harmonique : son laplacien est nul).
    On l'échantillonne aux centres, on calcule les vitesses aux faces comme dans le code,
    et on compare à la dérivée exacte au MILIEU de chaque face.
    Renvoie dx et l'erreur max sur les faces intérieures."""
    dxs, err = [], []
    for n in liste_n:
        dom = Domaine(np.ones((n, n), dtype=bool), 1.0 / n, 1.0 / n)  # carré fermé
        X, Y = dom.coordonnees()
        phi = np.exp(X) * np.cos(Y)
        u, v = vitesses_aux_bords(dom, phi)
        xf = np.arange(n + 1) * dom.dx                # abscisses des faces verticales
        yc = (np.arange(n) + 0.5) * dom.dy
        XF, YC = np.meshgrid(xf, yc)
        u_exact = np.exp(XF) * np.cos(YC)            # dphi/dx exact au milieu de la face
        err.append(np.abs(u[:, 1:-1] - u_exact[:, 1:-1]).max())
        dxs.append(dom.dx)
    return np.array(dxs), np.array(err)


# ----------------------------------------------------------------------
# 2. Canal rectiligne : translation exacte
# ----------------------------------------------------------------------
def profil_gauss(s, s0=0.25, sigma=0.05):
    return np.exp(-0.5 * ((s - s0) / sigma) ** 2)


def profil_creneau(s, a=0.15, b=0.35):
    return ((s >= a) & (s <= b)).astype(float)


def canal_vertical(n, largeur=3):
    """Canal vertical de n mailles de long (entrée en haut, sortie en bas, comme l'exige
    le code du professeur), potentiel calculé par laplace_pot.
    Le potentiel de sortie est choisi pour que la vitesse vaille exactement -1 (vers le bas) :
    phi varie linéairement de 0 à (n-1) dy entre les centres des mailles d'entrée et de sortie."""
    tab = np.zeros((n, largeur + 2), dtype=np.int32)    # colonnes 0 et largeur+1 = murs
    tab[:, 1:-1] = 1
    dy = 1.0 / n
    return potentiel_depuis_matrice(tab, phi_entree=0.0, phi_sortie=(n - 1) * dy, dx=dy)


def translation_canal(n, profil=profil_gauss, t_final=0.4, **opts):
    """Advection dans le canal vertical. s = distance depuis l'entrée (s = 1 - y).
    Solution exacte : c(s, t) = c0(s - t). Renvoie (s, c_num, c_exact, res, v)."""
    dom, phi = canal_vertical(n)
    u, v = vitesses_aux_bords(dom, phi)
    _, Y = dom.coordonnees()
    S = 1.0 - Y
    res = simuler(dom, u, v, profil(S), t_final, c_entree=0.0, n_sauvegardes=1,
                  verbose=False, **opts)
    j = dom.nx // 2                                   # colonne centrale
    s = S[:, j]
    c_exact = profil(s - 1.0 * t_final)
    return s, res['c'][-1][:, j], c_exact, res, v


def convergence(liste_nx=(50, 100, 200, 400, 800), profil=profil_gauss, **opts):
    """Erreur L1 et Linf en fonction de dx, et ordre observé entre deux maillages."""
    dxs, e1, einf = [], [], []
    for nx in liste_nx:
        x, c, ce, res, _ = translation_canal(nx, profil, **opts)
        dx = 1.0 / nx           # taille d'une maille (le canal mesure 1)
        dxs.append(dx)
        e1.append(np.sum(np.abs(c - ce)) * dx)
        einf.append(np.abs(c - ce).max())
    dxs, e1, einf = map(np.array, (dxs, e1, einf))
    ordre = np.log(e1[:-1] / e1[1:]) / np.log(dxs[:-1] / dxs[1:])
    return dxs, e1, einf, ordre


# ----------------------------------------------------------------------
# 3. Stabilité : condition CFL 2D
# ----------------------------------------------------------------------
def stabilite_cfl_2d(sigma, n=40, n_pas=400, schema='euler', reconstruction='constante',
                     a=1.0, b=1.0):
    """Advection en diagonale (u = a, v = b uniformes) d'un créneau carré.
    sigma = a dt/dx + b dt/dy (nombre de Courant 2D) est imposé exactement.
    Renvoie l'évolution de max|c| au cours du temps."""
    dom = domaine_carre_ouvert(n)
    u = np.full((n, n + 1), a)
    v = np.full((n + 1, n), b)
    X, Y = dom.coordonnees()
    c0 = ((np.abs(X - 0.3) < 0.1) & (np.abs(Y - 0.3) < 0.1)).astype(float)
    dt = sigma / (a / dom.dx + b / dom.dy)
    from volumes_finis import pas_euler, pas_rk2
    avance = pas_rk2 if schema == 'rk2' else pas_euler
    c = c0.copy()
    cmax = [np.abs(c).max()]
    for _ in range(n_pas):
        c, _ = avance(dom, c, u, v, dt, reconstruction=reconstruction, limiteur=None,
                      c_entree=0.0)
        cmax.append(np.abs(c).max())
        if cmax[-1] > 1e8:
            break
    return np.array(cmax), c, dom


# ----------------------------------------------------------------------
# 4. Analyse de Von Neumann (théorique)
# ----------------------------------------------------------------------
def symbole(theta, lam, reconstruction='constante'):
    """Valeur propre z = dt * L(e^{i k x}) / e^{i k x} de l'opérateur spatial (1D, a > 0).
    constante : L = -a (c_i - c_{i-1}) / dx
    lineaire  : L = -a (c_{i+1} + 3 c_i - 5 c_{i-1} + c_{i-2}) / (4 dx)   (cf. cours VF)"""
    e = np.exp(-1j * theta)
    if reconstruction == 'constante':
        return -lam * (1 - e)
    return -lam * (np.exp(1j * theta) + 3 - 5 * e + e**2) / 4


def amplification(lam, reconstruction='constante', schema='euler', n_theta=721):
    """max sur theta de |rho| ; rho = 1 + z (Euler) ou 1 + z + z^2/2 (RK2)."""
    theta = np.linspace(-np.pi, np.pi, n_theta)
    z = symbole(theta, lam, reconstruction)
    rho = 1 + z if schema == 'euler' else 1 + z + z**2 / 2
    return np.abs(rho).max()


def carte_amplification_2d(n=81, n_theta=73):
    """Upwind constant + Euler en 2D : max |rho|^2 - 1 dans le plan (lambda_x, lambda_y)."""
    lx = np.linspace(0, 1, n)
    LX, LY = np.meshgrid(lx, lx)
    th = np.linspace(-np.pi, np.pi, n_theta)
    TX, TY = np.meshgrid(th, th)
    res = np.empty_like(LX)
    for k in range(n):
        for m in range(n):
            rho = 1 - LX[k, m] * (1 - np.exp(-1j * TX)) - LY[k, m] * (1 - np.exp(-1j * TY))
            res[k, m] = (np.abs(rho) ** 2).max() - 1
    return LX, LY, res


# ----------------------------------------------------------------------
# 5. Fidélité de l'appel au code du professeur
# ----------------------------------------------------------------------
def comparer_reference_prof(image='donnees/laby.png', reference='donnees/out_pot_prof.npy'):
    """Refait, via potentiel.py, exactement le calcul du script du professeur
    (pleine résolution, phi_in = +100, phi_out = -100) et compare au fichier out_pot.npy
    fourni dans son dépôt. Renvoie l'écart maximal (doit être ~1e-10)."""
    from potentiel import lire_image
    ref = np.load(reference)                       # convention matricielle + bordure de 0
    dom, phi = potentiel_depuis_matrice(lire_image(image, 1), 100.0, -100.0)
    nous = np.flipud(np.nan_to_num(phi, nan=0.0))  # retour à la convention matricielle
    return np.abs(nous - ref[1:-1, 1:-1]).max()
