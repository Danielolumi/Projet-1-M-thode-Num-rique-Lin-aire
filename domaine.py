"""
domaine.py
==========
Auteur : Daniel Olumi — GCVIV0185-7 Méthodes numériques linéaires, ULiège 2026-2027

Définition du domaine de calcul (grille régulière de mailles carrées/rectangulaires).

Conventions :
  * Le domaine est découpé en ny lignes et nx colonnes de mailles.
  * Une maille est repérée par (i, j) :
        i = numéro de ligne   -> direction y (i = 0 en BAS, i augmente vers le haut)
        j = numéro de colonne -> direction x (j = 0 à GAUCHE, j augmente vers la droite)
  * Le centre de la maille (i, j) est en  x = (j + 0.5) dx ,  y = (i + 0.5) dy.
  * Les inconnues (potentiel phi, concentration c) sont stockées AU CENTRE des mailles
    -> tableaux de forme (ny, nx).
  * Les vitesses NORMALES sont stockées SUR LES BORDS (faces) des mailles :
        u[i, j]  : face verticale à GAUCHE de la maille (i, j)   -> forme (ny, nx+1)
        v[i, j]  : face horizontale en DESSOUS de la maille (i, j) -> forme (ny+1, nx)
    (c'est une grille "décalée" : centres pour les scalaires, bords pour les vitesses)

Chaque face reçoit un "type" :
    MUR       (0) : paroi imperméable (vitesse normale nulle, flux nul)
    INTERIEUR (1) : face entre deux mailles fluides
    ENTREE    (2) : face de bord du domaine par où le fluide entre
    SORTIE    (3) : face de bord du domaine par où le fluide sort
"""

import numpy as np

# Codes des types de faces
MUR = 0
INTERIEUR = 1
ENTREE = 2
SORTIE = 3


class Domaine:
    """Grille régulière + masque fluide/mur + types des faces."""

    def __init__(self, masque, dx=1.0, dy=1.0):
        # masque[i, j] = True si la maille (i, j) est du fluide, False si c'est un mur
        self.masque = np.asarray(masque, dtype=bool)
        self.ny, self.nx = self.masque.shape
        self.dx = float(dx)
        self.dy = float(dy)
        self._calculer_types_faces()

    # ------------------------------------------------------------------
    def _calculer_types_faces(self):
        """Toutes les faces sont MUR, sauf celles qui séparent deux mailles fluides."""
        m = self.masque
        self.type_u = np.full((self.ny, self.nx + 1), MUR, dtype=int)
        self.type_v = np.full((self.ny + 1, self.nx), MUR, dtype=int)
        # face u[i, j] (1 <= j <= nx-1) sépare la maille (i, j-1) et la maille (i, j)
        self.type_u[:, 1:-1][m[:, :-1] & m[:, 1:]] = INTERIEUR
        # face v[i, j] (1 <= i <= ny-1) sépare la maille (i-1, j) et la maille (i, j)
        self.type_v[1:-1, :][m[:-1, :] & m[1:, :]] = INTERIEUR

    # ------------------------------------------------------------------
    def ouvrir(self, cote, type_bord, debut=None, fin=None):
        """Ouvre une partie d'un côté du domaine (ENTREE ou SORTIE).

        cote      : 'gauche', 'droite', 'bas' ou 'haut'
        type_bord : ENTREE ou SORTIE
        debut,fin : indices (j pour bas/haut, i pour gauche/droite) de la zone ouverte.
                    None -> toutes les mailles fluides de ce côté sont ouvertes.
        """
        m = self.masque
        if cote == 'gauche':
            faces, fluide = self.type_u[:, 0], m[:, 0]
        elif cote == 'droite':
            faces, fluide = self.type_u[:, -1], m[:, -1]
        elif cote == 'bas':
            faces, fluide = self.type_v[0, :], m[0, :]
        elif cote == 'haut':
            faces, fluide = self.type_v[-1, :], m[-1, :]
        else:
            raise ValueError("cote doit valoir 'gauche', 'droite', 'bas' ou 'haut'")
        zone = np.zeros_like(fluide)
        zone[slice(debut, fin)] = True
        # 'faces' est une vue sur type_u/type_v : la modifier modifie le domaine
        faces[zone & fluide] = type_bord

    # ------------------------------------------------------------------
    def coordonnees(self):
        """Coordonnées (x, y) des centres de mailles, tableaux (ny, nx)."""
        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy
        return np.meshgrid(x, y)

    @property
    def aire(self):
        """Aire d'une maille S = dx * dy."""
        return self.dx * self.dy

    def __repr__(self):
        return (f"Domaine({self.ny} x {self.nx} mailles, dx={self.dx:g}, dy={self.dy:g}, "
                f"{self.masque.sum()} mailles fluides, "
                f"{np.sum(self.type_u == ENTREE) + np.sum(self.type_v == ENTREE)} faces d'entrée, "
                f"{np.sum(self.type_u == SORTIE) + np.sum(self.type_v == SORTIE)} faces de sortie)")


# ======================================================================
#  Domaine de test (le domaine "labyrinthe" est construit dans potentiel.py
#  à partir du résultat du code du professeur)
# ======================================================================
def domaine_carre_ouvert(n, cote=1.0):
    """Carré entièrement fluide : entrées à gauche et en bas, sorties à droite et en haut
    (utilisé pour une advection en diagonale)."""
    d = cote / n
    dom = Domaine(np.ones((n, n), dtype=bool), d, d)
    dom.ouvrir('gauche', ENTREE)
    dom.ouvrir('bas', ENTREE)
    dom.ouvrir('droite', SORTIE)
    dom.ouvrir('haut', SORTIE)
    return dom
