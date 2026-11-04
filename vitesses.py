"""
vitesses.py
===========
Auteur : Daniel Olumi — GCVIV0185-7 Méthodes numériques linéaires, ULiège 2026-2027

Étape 2 : vitesses NORMALES sur chaque bord des mailles, déduites du potentiel.

    u = dphi/dx ,  v = dphi/dy          (énoncé : u_i = dphi/dx_i)

Face intérieure entre deux centres distants de dx : différence centrée
    u_{i,j-1/2} = (phi_{i,j} - phi_{i,j-1}) / dx         -> erreur O(dx^2)
Face de mur : u = 0 (paroi imperméable).
Face extérieure d'une maille d'entrée/sortie : dans le code du professeur, le potentiel
est imposé AU CENTRE de ces mailles ; aucune valeur de phi n'existe au-delà. La vitesse
de cette face est donc obtenue par la CONTINUITÉ de la maille : ce qui entre par la
face extérieure = ce qui sort par ses autres faces (divergence nulle).

Ce module contient aussi :
  * divergence(dom, u, v)  : vérification de  du/dx + dv/dy = 0  maille par maille
  * pas_de_temps_max(...)  : pas de temps maximal autorisé par la condition CFL 2D
"""

import numpy as np

from domaine import INTERIEUR, ENTREE, SORTIE


def vitesses_aux_bords(dom, phi):
    """Renvoie u (ny, nx+1) et v (ny+1, nx) : vitesses normales aux faces."""
    ny, nx = dom.ny, dom.nx
    dx, dy = dom.dx, dom.dy
    u = np.zeros((ny, nx + 1))
    v = np.zeros((ny + 1, nx))

    # 1) faces intérieures : différence centrée entre les deux centres voisins
    I, J = np.nonzero(dom.type_u == INTERIEUR)
    u[I, J] = (phi[I, J] - phi[I, J - 1]) / dx
    I, J = np.nonzero(dom.type_v == INTERIEUR)
    v[I, J] = (phi[I, J] - phi[I - 1, J]) / dy

    # 2) faces extérieures des mailles d'entrée/sortie : continuité de la maille
    #    B = bilan (divergence) de la maille calculé avec la face extérieure encore à 0
    ouverte_u = (dom.type_u == ENTREE) | (dom.type_u == SORTIE)
    ouverte_v = (dom.type_v == ENTREE) | (dom.type_v == SORTIE)
    B = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
    # bord haut : (v_haut - v_bas)/dy + ... = 0  ->  v_haut = -B * dy
    v[-1, :] = np.where(ouverte_v[-1, :], -B[-1, :] * dy, v[-1, :])
    # bord bas  : (v_haut - v_bas)/dy + ... = 0  ->  v_bas = +B * dy
    v[0, :] = np.where(ouverte_v[0, :], B[0, :] * dy, v[0, :])
    # bords gauche / droit (même raisonnement en x)
    u[:, 0] = np.where(ouverte_u[:, 0], B[:, 0] * dx, u[:, 0])
    u[:, -1] = np.where(ouverte_u[:, -1], -B[:, -1] * dx, u[:, -1])
    # 3) faces MUR : u = v = 0 (déjà à zéro)
    return u, v


def vitesses_aux_centres(u, v):
    """Moyenne des deux faces opposées -> vitesse au centre (pour l'affichage)."""
    uc = 0.5 * (u[:, :-1] + u[:, 1:])
    vc = 0.5 * (v[:-1, :] + v[1:, :])
    return uc, vc


def divergence(dom, u, v):
    """Divergence discrète dans chaque maille :  (u_E - u_W)/dx + (v_N - v_S)/dy.

    C'est exactement le bilan "ce qui sort - ce qui entre" divisé par l'aire :
    elle doit être nulle (fluide incompressible). NaN dans les murs."""
    div = (u[:, 1:] - u[:, :-1]) / dom.dx + (v[1:, :] - v[:-1, :]) / dom.dy
    return np.where(dom.masque, div, np.nan)


def pas_de_temps_max(dom, u, v):
    """Pas de temps maximal pour la condition CFL 2D :
           dt * ( |u|/dx + |v|/dy ) <= 1     dans chaque maille
    (on prend dans chaque maille la plus grande vitesse de ses deux faces)."""
    umax = np.maximum(np.abs(u[:, :-1]), np.abs(u[:, 1:]))
    vmax = np.maximum(np.abs(v[:-1, :]), np.abs(v[1:, :]))
    somme = umax / dom.dx + vmax / dom.dy
    return 1.0 / somme.max()


def debits_bords(dom, u, v):
    """Débits entrant et sortant (par unité de profondeur) à travers les bords ouverts."""
    q_entree = q_sortie = 0.0
    dx, dy = dom.dx, dom.dy
    for code in (ENTREE, SORTIE):
        # flux positif = entrant dans le domaine
        q = (np.sum(u[:, 0][dom.type_u[:, 0] == code]) * dy
             - np.sum(u[:, -1][dom.type_u[:, -1] == code]) * dy
             + np.sum(v[0, :][dom.type_v[0, :] == code]) * dx
             - np.sum(v[-1, :][dom.type_v[-1, :] == code]) * dx)
        if code == ENTREE:
            q_entree = q
        else:
            q_sortie = -q
    return q_entree, q_sortie
