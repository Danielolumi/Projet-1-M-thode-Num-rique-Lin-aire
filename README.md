# Projet Volumes Finis — Advection dans un labyrinthe

**GCVIV0185-7 — Méthodes numériques linéaires en génie civil et géologique (ULiège, 2026-2027)**
Auteur : Daniel Olumi

Résolution de l'équation d'advection linéaire 2D, sous forme conservative,

$$\frac{\partial c}{\partial t}+\frac{\partial (uc)}{\partial x}+\frac{\partial (vc)}{\partial y}=0,$$

par la méthode des Volumes Finis, dans le champ de vitesse irrotationnel et incompressible
d'un labyrinthe, en deux versions : **Euler + reconstruction constante** et
**Runge-Kutta 2 + reconstruction linéaire** (avec limiteur de Barth-Jespersen). Le champ de vitesse est déduit du potentiel $\phi$ ($u=\partial\phi/\partial x$,
$v=\partial\phi/\partial y$) calculé par le code de Laplace fourni (différences finies).

## Organisation

| Fichier | Rôle |
|---|---|
| `laplace_only.py` | code fourni (dépôt `HECE/labyrinth_public`), utilisé sans modification |
| `potentiel.py` | lecture de l'image, appel de `laplace_pot`, passage à nos conventions |
| `domaine.py` | grille régulière, types de faces (mur / intérieur / entrée / sortie) |
| `vitesses.py` | vitesses normales aux bords, divergence, pas de temps CFL, débits |
| `volumes_finis.py` | reconstruction (constante / linéaire / limitée), flux amont, bilan, Euler, RK2 |
| `validation.py` | tests : précision des vitesses, translation, convergence, stabilité, Von Neumann |
| `affichage.py` | figures et animation (GIF) |
| `main.py` | chaîne complète sur le labyrinthe |
| `projet_VF.ipynb` | **notebook de présentation** : résultats, validation et analyse |
| `rapport.pdf` | rapport synthétique |
| `donnees/laby.png` | image du labyrinthe |
| `donnees/out_pot_prof.npy` | potentiel de référence du dépôt fourni (contrôle de l'appel) |

## Utilisation

Bibliothèques : `numpy`, `scipy`, `matplotlib`, `pillow`, `ipykernel` (et `tqdm`, facultatif) :

```
pip install -r requirements.txt
```

**Notebook** `projet_VF.ipynb` (VS Code avec l'extension Jupyter, ou `jupyter notebook`) : exécuter
les cellules dans l'ordre depuis le dossier du projet. Toutes les figures et l'animation
(`ecoulement.gif`, environ 1 minute) sont recalculées ; exécution complète en quelques minutes.

Chaîne complète en script (figures enregistrées dans `resultats/`) :

```
python main.py
```

## Principaux résultats

* divergence discrète du champ de vitesse nulle à la précision machine ; débit entrant = débit sortant ;
* conservation de la masse du constituant à $10^{-15}$ près (forme conservative) ;
* ordre de convergence 1 (reconstruction constante) et 2 (reconstruction linéaire + RK2) ;
* stabilité sous la condition CFL 2D $\Delta t\,(|u|/\Delta x+|v|/\Delta y)\le 1$ ; reconstruction
  linéaire + Euler explicite instable ;
* le constituant suit le chemin entrée → sortie, les culs-de-sac (vitesse nulle) restent propres.
