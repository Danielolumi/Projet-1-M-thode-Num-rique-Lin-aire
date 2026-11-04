"""
main.py
=======
Auteur : Daniel Olumi — GCVIV0185-7 Méthodes numériques linéaires, ULiège 2026-2027

Script principal : enchaîne toute la chaîne de calcul sur le labyrinthe.
    1. potentiel des vitesses phi (code Laplace du professeur, différences finies)
    2. vitesses normales sur les bords des mailles + vérifications
    3. advection du constituant par volumes finis
Les figures sont affichées et enregistrées dans le dossier « resultats ».

Lancement : Spyder -> F5 ; VS Code -> triangle « Run Python File » ;
            terminal -> python main.py   (depuis ce dossier)
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from potentiel import potentiel_labyrinthe
from vitesses import vitesses_aux_bords, divergence, debits_bords, pas_de_temps_max
from volumes_finis import simuler
from affichage import carte, carte_vitesse, serie, bilan_masse

# ---------------- paramètres ----------------
IMAGE = os.path.join('donnees', 'laby.png')
FACTEUR = 4                    # 1 maille = 4 x 4 pixels de l'image
PHI_ENTREE, PHI_SORTIE = 0.0, 1.0
T_FINAL = 20.0                 # durée simulée
CFL = 0.9                      # dt = CFL * dt_max
DOSSIER = 'resultats'

# on travaille toujours depuis le dossier qui contient ce fichier
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(DOSSIER, exist_ok=True)

# 1) domaine et potentiel
dom, phi = potentiel_labyrinthe(IMAGE, facteur=FACTEUR,
                                phi_entree=PHI_ENTREE, phi_sortie=PHI_SORTIE)
print(dom)

# 2) vitesses normales aux bords + vérifications
u, v = vitesses_aux_bords(dom, phi)
div = divergence(dom, u, v)
q_in, q_out = debits_bords(dom, u, v)
print(f"max |div|        = {np.nanmax(np.abs(div)):.2e}")
print(f"débit entrant    = {q_in:.6f}   débit sortant = {q_out:.6f}")
print(f"dt max (CFL = 1) = {pas_de_temps_max(dom, u, v):.3e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
carte(dom, phi, r'Potentiel des vitesses $\phi$', ax=ax[0])
carte_vitesse(dom, u, v, ax=ax[1])
fig.tight_layout()
fig.savefig(os.path.join(DOSSIER, 'potentiel_vitesse.png'), dpi=120)

# 3) advection par volumes finis (labyrinthe initialement « propre », c = 1 à l'entrée)
c0 = np.zeros((dom.ny, dom.nx))
res = simuler(dom, u, v, c0, T_FINAL, cfl=CFL, schema='rk2',
              reconstruction='lineaire', limiteur='barth', c_entree=1.0)
ecart = np.abs(res['masse'] - res['masse'][0] - res['apport']).max()
print(f"conservation de la masse : écart max = {ecart:.1e}")
print(f"min c = {res['c'][-1].min():.2e}   max c = {res['c'][-1].max():.6f}")

fig = serie(dom, res, titre='Advection du constituant (RK2, reconstruction linéaire limitée)')
fig.savefig(os.path.join(DOSSIER, 'advection_labyrinthe.png'), dpi=120)
fig = bilan_masse(res)
fig.savefig(os.path.join(DOSSIER, 'bilan_masse.png'), dpi=120)
print(f"figures enregistrées dans le dossier « {DOSSIER} »")
plt.show()
