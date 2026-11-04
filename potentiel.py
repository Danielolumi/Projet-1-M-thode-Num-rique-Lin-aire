"""
potentiel.py
============
Auteur : Daniel Olumi — GCVIV0185-7 Méthodes numériques linéaires, ULiège 2026-2027

Étape 1 du projet : champ de potentiel des vitesses phi.

Le Laplacien est résolu par le CODE DU PROFESSEUR (laplace_only.py, fonction
laplace_pot, dépôt gitlab.uliege.be/HECE/labyrinth_public), utilisé SANS modification.
Ce module ne fait que :
  1. préparer la matrice de domaine (lecture de l'image, réduction de résolution,
     suppression des poches isolées) ;
  2. appeler laplace_pot ;
  3. convertir son résultat vers nos conventions (objet Domaine, axe y vers le haut).

Rappel de ce que fait laplace_pot (code du professeur) :
  * convention MATRICIELLE : [0, 0] en haut à gauche, i vers le BAS ;
  * 0 = mur (hors domaine), 1 = fluide ;
  * entrée = mailles fluides de la LIGNE SUPÉRIEURE, sortie = mailles fluides de la
    LIGNE INFÉRIEURE : potentiel imposé DANS ces mailles (condition "forte", Dirichlet) ;
  * ailleurs : somme(phi_voisins fluides) - (nb de voisins fluides) * phi_P = 0
    (Laplacien à 5 points ; un mur n'est pas compté -> dphi/dn = 0, paroi imperméable) ;
  * renvoie (domaine, potentiel) entourés d'une bordure de 0 ;
    domaine : 0 = mur, 1 = intérieur, 2 = maille de condition limite.
"""

import sys
import types

import numpy as np
from PIL import Image
from scipy import ndimage

# laplace_only.py utilise tqdm uniquement pour afficher des barres de progression.
# Si cette bibliothèque n'est pas installée, on la remplace par une version qui ne
# fait rien : le calcul est identique, seules les barres ne s'affichent pas.
try:
    import tqdm                          # noqa: F401
except ImportError:
    sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda iterable, *a, **k: iterable)

import laplace_only                      # code du professeur (non modifié)
from domaine import Domaine, ENTREE, SORTIE


def lire_image(chemin, facteur=1, seuil=200):
    """Lit l'image comme le fait le professeur (composante Rouge, seuil 200) :
    renvoie une matrice d'entiers 1 = fluide (blanc), 0 = mur (noir).

    facteur : réduction de résolution. Un bloc facteur x facteur pixels devient une maille,
              fluide seulement si TOUS ses pixels sont blancs (un mur ne peut pas disparaître).
    """
    rouge = np.asarray(Image.open(chemin).convert('RGB'))[:, :, 0]
    fluide = rouge > seuil
    if facteur > 1:   # taille complétée (par du blanc) jusqu'au multiple de facteur suivant
        ny = int(np.ceil(fluide.shape[0] / facteur)) * facteur
        nx = int(np.ceil(fluide.shape[1] / facteur)) * facteur
        blocs = np.pad(fluide, ((0, ny - fluide.shape[0]), (0, nx - fluide.shape[1])), constant_values=True).reshape(ny // facteur, facteur, nx // facteur, facteur)
        fluide = blocs.all(axis=(1, 3))
    return fluide.astype(np.int32)


def preparer(tableau):
    """Recadre la matrice sur la zone contenant des murs (comme laplace_pot) et supprime
    les poches de fluide non reliées à l'entrée ou à la sortie : dans ces poches aucun
    potentiel n'est imposé, la matrice du système serait singulière."""
    murs = np.where(tableau == 0)
    t = tableau[murs[0].min():murs[0].max() + 1, murs[1].min():murs[1].max() + 1].copy()
    etiquettes, _ = ndimage.label(t == 1)                 # zones fluides connexes
    utiles = set(etiquettes[0, :][t[0, :] == 1]) | set(etiquettes[-1, :][t[-1, :] == 1])
    t[~np.isin(etiquettes, list(utiles))] = 0
    return t


def potentiel_depuis_matrice(tableau, phi_entree=0.0, phi_sortie=1.0, dx=1.0):
    """Applique le code du professeur à une matrice 0/1 (convention matricielle) et
    renvoie (dom, phi) dans NOS conventions :
        dom : objet Domaine (i = 0 en BAS), faces extérieures des mailles d'entrée = ENTREE,
              des mailles de sortie = SORTIE
        phi : potentiel aux centres, tableau (ny, nx), NaN dans les murs
    """
    t = preparer(tableau)
    dom_prof, pot_prof = laplace_only.laplace_pot(t.copy(), outside_value=0, inside_value=1,
                                                  pot_in=phi_entree, pot_out=phi_sortie)
    # on retire la bordure de 0 ajoutée par laplace_pot
    dom_prof = dom_prof[1:-1, 1:-1]
    pot_prof = pot_prof[1:-1, 1:-1]
    # convention matricielle (i vers le bas) -> notre convention (i vers le haut = axe y)
    dom_prof = np.flipud(dom_prof)
    pot_prof = np.flipud(pot_prof)

    masque = dom_prof > 0                                   # 1 ou 2 = fluide
    dom = Domaine(masque, dx, dx)
    dom.ouvrir('haut', ENTREE)                               # au-dessus des mailles d'entrée
    dom.ouvrir('bas', SORTIE)                                # sous les mailles de sortie
    dom.cl = (dom_prof == 2)                                 # mailles à potentiel imposé
    phi = np.where(masque, pot_prof, np.nan)
    return dom, phi


def potentiel_labyrinthe(chemin, facteur=4, phi_entree=0.0, phi_sortie=1.0, taille=1.0):
    """Chaîne complète image -> (dom, phi). Le domaine a une largeur `taille`."""
    tableau = lire_image(chemin, facteur)
    nx = preparer(tableau).shape[1]
    return potentiel_depuis_matrice(tableau, phi_entree, phi_sortie, dx=taille / nx)
