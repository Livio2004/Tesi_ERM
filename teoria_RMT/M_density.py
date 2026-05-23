import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
import linalg as lin
import seaborn as sns 


def genera_matrice_S(N, b0):
    X = np.random.randn(N, 3)
    tensore = X[:, np.newaxis]-X[np.newaxis, :]
    norma = np.linalg.norm(tensore, axis=-1)
    fattore = np.sqrt(N/b0)
    argomento = fattore*norma
    S = np.sinc(argomento/np.pi) #il sinc di numpy ha il pi greco
    return S

campionamenti = 10
N = 1000
b0_valori = [0.1,1,3,10]
fig, axes = plt.subplots(nrows=1, ncols = 4, figsize = (12,6))
fig.suptitle('Densità di autovalori con simulazione montecarlo', fontweight= 'bold')
for ax, b0 in zip(axes, b0_valori):
    tutti_autovalori = []
    for _ in range(campionamenti):
        # Esempio: matrice simmetrica casuale (Gaussian Orthogonal Ensemble)
        mat = genera_matrice_S(N,b0)
        autovalori, autovettori = eigh(mat)
        tutti_autovalori.extend(autovalori) #bisogna usare extend perchè append aggiunge ogni volta nuovi blocchi
    sns.kdeplot(tutti_autovalori,ax = ax, fill=True, color='darkblue', bw_adjust=0.5, label = f'b0 ={b0} ')
    ax.set_xlabel(r'$\lambda$ ')
    ax.set_ylabel(r'$p(\lambda)$')
    ax.legend()



plt.legend()
plt.tight_layout()
plt.show()


