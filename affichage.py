"""
affichage.py
============
Auteur : Daniel Olumi — GCVIV0185-7 Méthodes numériques linéaires, ULiège 2026-2027

Fonctions graphiques (matplotlib) pour visualiser les résultats.
"""

import numpy as np
import matplotlib.pyplot as plt

from vitesses import vitesses_aux_centres


def _etendue(dom):
    """Limites [xmin, xmax, ymin, ymax] du domaine pour imshow."""
    return [0, dom.nx * dom.dx, 0, dom.ny * dom.dy]


def carte(dom, champ, titre='', ax=None, cmap='viridis', vmin=None, vmax=None, barre=True):
    """Affiche un champ (ny, nx) ; les murs sont en blanc."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    z = np.ma.masked_where(~dom.masque, champ)          # on masque les murs
    im = ax.imshow(z, origin='lower', extent=_etendue(dom), cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(titre)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if barre:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def carte_vitesse(dom, u, v, titre='Norme de la vitesse', ax=None, fleches=True, pas=None):
    """Norme de la vitesse aux centres + quelques flèches (quiver)."""
    uc, vc = vitesses_aux_centres(u, v)
    norme = np.hypot(uc, vc)
    ax = carte(dom, norme, titre, ax=ax, cmap='magma')
    if fleches:
        X, Y = dom.coordonnees()
        p = pas or max(1, dom.nx // 30)
        sel = (slice(None, None, p), slice(None, None, p))
        msk = dom.masque[sel] & (norme[sel] > 1e-3 * norme.max())
        ax.quiver(X[sel][msk], Y[sel][msk], uc[sel][msk], vc[sel][msk],
                  color='cyan', scale_units='xy', angles='xy',
                  scale=norme.max() / (p * dom.dx), width=0.003)
    return ax


def serie(dom, res, n_max=6, titre='', vmin=0, vmax=1):
    """Affiche plusieurs instants sauvegardés d'une simulation."""
    idx = np.linspace(0, len(res['c']) - 1, min(n_max, len(res['c']))).round().astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(3.2 * len(idx), 3.4))
    for ax, k in zip(np.atleast_1d(axes), idx):
        carte(dom, res['c'][k], f"t = {res['t'][k]:.3g}", ax=ax, vmin=vmin, vmax=vmax, barre=False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


def bilan_masse(res, titre=''):
    """Masse dans le domaine vs masse entrée cumulée, et leur écart (conservation)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(res['temps'], res['masse'], label='masse dans le domaine  M(t)')
    a1.plot(res['temps'], res['apport'], '--', label="masse nette entrée  ∫(Q_in - Q_out) dt")
    a1.set_xlabel('t'); a1.legend(); a1.set_title('Bilan de masse')
    ecart = res['masse'] - res['masse'][0] - res['apport']
    a2.plot(res['temps'], ecart)
    a2.set_xlabel('t'); a2.set_title('Écart  M(t) - M(0) - apport  (doit être ~1e-15)')
    fig.suptitle(titre); fig.tight_layout()
    return fig


def animer(dom, phi, u, v, res, fichier='ecoulement.gif', images_par_seconde=10, dpi=60):
    """Animation en trois panneaux (potentiel, |u|, concentration) enregistrée en GIF.

    Seule la concentration change d'une image à l'autre ; une image par instant
    sauvegardé dans res (paramètre n_sauvegardes de simuler).
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    uc, vc = vitesses_aux_centres(u, v)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    carte(dom, phi, r'Potentiel $\phi$', ax=axes[0])
    carte(dom, np.hypot(uc, vc), r'Norme de la vitesse $|\mathbf{u}|$', ax=axes[1], cmap='magma')
    carte(dom, res['c'][0], '', ax=axes[2], vmin=0, vmax=1)
    image_c = axes[2].images[0]
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
    fig.tight_layout()

    def mise_a_jour(k):
        image_c.set_data(np.ma.masked_where(~dom.masque, res['c'][k]))
        axes[2].set_title(f"Concentration, t = {res['t'][k]:.1f}")
        return (image_c,)

    anim = FuncAnimation(fig, mise_a_jour, frames=len(res['c']), blit=False)
    anim.save(fichier, writer=PillowWriter(fps=images_par_seconde), dpi=dpi)
    plt.close(fig)
    return fichier
