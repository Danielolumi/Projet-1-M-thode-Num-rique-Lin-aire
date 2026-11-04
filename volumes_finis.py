"""
volumes_finis.py
================
Auteur : Daniel Olumi — GCVIV0185-7 Méthodes numériques linéaires, ULiège 2026-2027

Étape 3 : advection d'un constituant c par la méthode des Volumes Finis.

Équation (forme CONSERVATIVE, à privilégier en VF) :
    dc/dt + d(u c)/dx + d(v c)/dy = 0

Intégration sur une maille de surface S = dx dy (théorème de Green/divergence) :
    S dc/dt + somme des flux sortants à travers les 4 bords = 0
    dc/dt = -[ (F_E - F_W) / dx + (G_N - G_S) / dy ]          ("bilan des échanges / S")
avec   F = u * c_bord  (flux en x par unité de longueur de face)
       G = v * c_bord  (flux en y)

Étapes à chaque pas de temps (structure type d'un code VF, cf. cours) :
  1. reconstruction de c sur les bords (constante ou linéaire)
  2. calcul d'UN flux unique par bord : décentrement amont ("upwind")
  3. bilan des flux dans chaque maille
  4. mise à jour en temps (Euler explicite ou Runge-Kutta à 2 pas)
"""

import numpy as np

from domaine import INTERIEUR, ENTREE


# ======================================================================
# 1. Reconstruction
# ======================================================================
def pentes(dom, c, reconstruction='constante', limiteur=None):
    """Gradients directionnels de c dans chaque maille : gx = dc/dx, gy = dc/dy.

    constante : gx = gy = 0  (c est uniforme dans la maille)
    lineaire  : gradient centré sur les 2 mailles voisines
                gx = (c_E - c_W) / (2 dx)
                Si une des deux voisines n'existe pas (mur, bord), on revient à une
                reconstruction constante dans cette direction (cf. "Domaine limité").
    limiteur  : None ou 'barth' (Barth-Jespersen directionnel) -> la valeur reconstruite
                sur un bord ne peut pas sortir de [min, max] des 3 mailles (pas de
                nouvel extremum -> pas d'oscillation).
    """
    gx = np.zeros_like(c)
    gy = np.zeros_like(c)
    if reconstruction == 'constante':
        return gx, gy

    # valeurs des voisines Est, Ouest, Nord, Sud (0 si hors du domaine)
    cE = np.zeros_like(c); cE[:, :-1] = c[:, 1:]
    cW = np.zeros_like(c); cW[:, 1:] = c[:, :-1]
    cN = np.zeros_like(c); cN[:-1, :] = c[1:, :]
    cS = np.zeros_like(c); cS[1:, :] = c[:-1, :]

    # la pente n'est calculée que si les DEUX faces opposées sont intérieures
    ok_x = (dom.type_u[:, :-1] == INTERIEUR) & (dom.type_u[:, 1:] == INTERIEUR)
    ok_y = (dom.type_v[:-1, :] == INTERIEUR) & (dom.type_v[1:, :] == INTERIEUR)
    gx = np.where(ok_x, (cE - cW) / (2 * dom.dx), 0.0)
    gy = np.where(ok_y, (cN - cS) / (2 * dom.dy), 0.0)

    if limiteur == 'barth':
        gx = _limiter(c, cW, cE, gx, 0.5 * dom.dx)
        gy = _limiter(c, cS, cN, gy, 0.5 * dom.dy)
    return gx, gy


def _limiter(c, c_moins, c_plus, g, demi):
    """Barth-Jespersen 1D : on multiplie la pente par phi in [0, 1] tel que
    c +/- demi*g reste entre le min et le max de (c_moins, c, c_plus)."""
    cmax = np.maximum(np.maximum(c_moins, c), c_plus)
    cmin = np.minimum(np.minimum(c_moins, c), c_plus)
    theta = np.abs(demi * g)                  # écart maximal entre centre et bord
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.minimum(1.0, np.minimum((cmax - c) / theta, (c - cmin) / theta))
    phi = np.where(theta > 0, phi, 1.0)
    return phi * g


# ======================================================================
# 2. Flux aux bords (décentrement amont)
# ======================================================================
def flux(dom, c, u, v, reconstruction='constante', limiteur=None, c_entree=1.0):
    """Flux numériques F (ny, nx+1) et G (ny+1, nx) sur toutes les faces.

    Chaque face voit deux valeurs reconstruites : cL (côté gauche/bas) et cR (côté
    droit/haut). Le flux doit être UNIQUE : on prend la valeur venant de l'amont
        F = u * cL  si u > 0      ;      F = u * cR  si u < 0
    Bords ouverts : la valeur extérieure ("condition limite faible") vaut c_entree
    pour une ENTREE et 0 pour une SORTIE ; elle n'est utilisée que si le fluide entre.
    Faces MUR : u = 0 donc F = 0 automatiquement.
    """
    dx, dy = dom.dx, dom.dy
    gx, gy = pentes(dom, c, reconstruction, limiteur)
    ny, nx = c.shape

    # ---- direction x ----
    cL = np.empty((ny, nx + 1))      # valeur du côté GAUCHE de chaque face verticale
    cR = np.empty((ny, nx + 1))      # valeur du côté DROIT
    cL[:, 1:] = c + 0.5 * dx * gx    # bord Est de la maille (i, j)  -> face j+1, côté gauche
    cR[:, :-1] = c - 0.5 * dx * gx   # bord Ouest de la maille (i, j) -> face j, côté droit
    cL[:, 0] = np.where(dom.type_u[:, 0] == ENTREE, c_entree, 0.0)    # extérieur à gauche
    cR[:, -1] = np.where(dom.type_u[:, -1] == ENTREE, c_entree, 0.0)  # extérieur à droite
    F = np.where(u > 0, u * cL, u * cR)

    # ---- direction y ----
    cB = np.empty((ny + 1, nx))      # côté BAS de chaque face horizontale
    cH = np.empty((ny + 1, nx))      # côté HAUT
    cB[1:, :] = c + 0.5 * dy * gy
    cH[:-1, :] = c - 0.5 * dy * gy
    cB[0, :] = np.where(dom.type_v[0, :] == ENTREE, c_entree, 0.0)
    cH[-1, :] = np.where(dom.type_v[-1, :] == ENTREE, c_entree, 0.0)
    G = np.where(v > 0, v * cB, v * cH)
    return F, G


# ======================================================================
# 3. Bilan
# ======================================================================
def bilan(dom, c, u, v, reconstruction='constante', limiteur=None, c_entree=1.0):
    """Renvoie dc/dt dans chaque maille et le flux de masse NET ENTRANT par les bords."""
    F, G = flux(dom, c, u, v, reconstruction, limiteur, c_entree)
    dcdt = -((F[:, 1:] - F[:, :-1]) / dom.dx + (G[1:, :] - G[:-1, :]) / dom.dy)
    dcdt[~dom.masque] = 0.0
    entrant = ((F[:, 0].sum() - F[:, -1].sum()) * dom.dy
               + (G[0, :].sum() - G[-1, :].sum()) * dom.dx)
    return dcdt, entrant


# ======================================================================
# 4. Intégration temporelle
# ======================================================================
def pas_euler(dom, c, u, v, dt, **opts):
    """Euler explicite :  c^{n+1} = c^n + dt * L(c^n)   (ordre 1 en temps)."""
    L, q = bilan(dom, c, u, v, **opts)
    return c + dt * L, dt * q


def pas_rk2(dom, c, u, v, dt, **opts):
    """Runge-Kutta à 2 pas (Heun) = deux Euler successifs puis moyenne (ordre 2) :
        c*      = c^n + dt L(c^n)
        c**     = c*  + dt L(c*)
        c^{n+1} = (c^n + c**) / 2
    """
    L1, q1 = bilan(dom, c, u, v, **opts)
    c_etoile = c + dt * L1
    L2, q2 = bilan(dom, c_etoile, u, v, **opts)
    return 0.5 * (c + c_etoile + dt * L2), 0.5 * dt * (q1 + q2)


def simuler(dom, u, v, c0, t_final, cfl=0.9, schema='rk2', reconstruction='lineaire',
            limiteur=None, c_entree=1.0, n_sauvegardes=10, verbose=True):
    """Intègre l'équation d'advection de t = 0 à t_final.

    cfl : fraction du pas de temps maximal autorisé  (dt = cfl * dt_max)
    Renvoie un dictionnaire :
        't'      : instants sauvegardés,      'c' : champs sauvegardés
        'masse'  : masse dans le domaine à chaque pas,  'apport' : masse entrée cumulée
        'temps'  : instants de chaque pas,   'dt' : pas de temps utilisé
    """
    from vitesses import pas_de_temps_max
    dt_max = pas_de_temps_max(dom, u, v)
    dt = cfl * dt_max
    n_pas = int(np.ceil(t_final / dt - 1e-12))
    dt = t_final / n_pas                      # on tombe exactement sur t_final
    avance = pas_rk2 if schema == 'rk2' else pas_euler
    opts = dict(reconstruction=reconstruction, limiteur=limiteur, c_entree=c_entree)

    c = np.where(dom.masque, c0, 0.0).astype(float)
    S = dom.aire
    instants = set(np.linspace(0, n_pas, n_sauvegardes + 1).round().astype(int).tolist())
    res = {'t': [0.0], 'c': [c.copy()], 'temps': [0.0], 'masse': [c.sum() * S],
           'apport': [0.0], 'dt': dt, 'dt_max': dt_max, 'n_pas': n_pas}
    apport = 0.0
    for n in range(1, n_pas + 1):
        c, dq = avance(dom, c, u, v, dt, **opts)
        apport += dq
        res['temps'].append(n * dt)
        res['masse'].append(c.sum() * S)
        res['apport'].append(apport)
        if n in instants:
            res['t'].append(n * dt)
            res['c'].append(c.copy())
        if not np.isfinite(c).all() or np.abs(c).max() > 1e6:
            if verbose:
                print(f"  !! INSTABILITÉ détectée au pas {n} (t = {n*dt:.4g}) : arrêt")
            res['t'].append(n * dt); res['c'].append(c.copy())
            res['instable'] = True
            break
    else:
        res['instable'] = False
    for cle in ('temps', 'masse', 'apport', 't'):
        res[cle] = np.array(res[cle])
    if verbose:
        print(f"  {schema.upper()} / {reconstruction}{' + '+limiteur if limiteur else ''} : "
              f"{n} pas, dt = {dt:.3e} (CFL = {dt/dt_max:.2f})")
    return res
