import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

N = 1200       
k0 = 2.0     
sigma = 1.0    


X = np.random.normal(0, sigma, (N, 3))

D = squareform(pdist(X))

S = np.sinc(k0 * D / np.pi) #uso funzione numpy cosi mi fa la riga di 1

print("Diagonalizzazione della matrice S...")
evals, evecs = eigh(S)

idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]



fig = plt.figure(figsize=(16, 10))


ax_spec = fig.add_subplot(2, 3, (1, 3))
ax_spec.plot(evals[:40], marker='o', linestyle='-', color='k', alpha=0.7)
ax_spec.set_title('Spettro degli Autovalori (Primi 40)', fontsize=14)
ax_spec.set_ylabel('Autovalore $\lambda$', fontsize=12)
ax_spec.set_xlabel('Indice $i$', fontsize=12)
ax_spec.grid(True, linestyle='--', alpha=0.5)

# Evidenziamo l linee di degenerazione teorica
ax_spec.axvspan(-0.5, 0.5, color='red', alpha=0.2, label='l=0 (Deg: 1)')
ax_spec.axvspan(0.5, 3.5, color='blue', alpha=0.2, label='l=1 (Deg: 3)')
ax_spec.axvspan(3.5, 8.5, color='green', alpha=0.2, label='l=2 (Deg: 5)')
ax_spec.axvspan(8.5,15.5,color = 'purple', alpha = 0.2, label = 'l=3 (deg=7)' )
ax_spec.legend()

def plot_eigenvector_3d(ax, evec_data, title):
    vmax = np.max(np.abs(evec_data))
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                    c=evec_data, cmap='bwr', vmin=-vmax, vmax=vmax, 
                    s=20, alpha=0.8, edgecolors='none')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


ax_0 = fig.add_subplot(2, 3, 4, projection='3d')
plot_eigenvector_3d(ax_0, evecs[:, 0], "Autovettore 0 (l=0, m=0)\nSimmetria Radiale (Orbitale s)")

ax_1 = fig.add_subplot(2, 3, 5, projection='3d')
plot_eigenvector_3d(ax_1, evecs[:, 1], "Autovettore 1 (l=1, m=-1)\nSingolo Piano Nodale (Orbitale p)")

ax_2 = fig.add_subplot(2, 3, 6, projection='3d')
plot_eigenvector_3d(ax_2, evecs[:, 2], "Autovettore 2 (l=1, m=0)\nSingolo Piano Nodale (Orbitale p)")

plt.tight_layout()
plt.show()