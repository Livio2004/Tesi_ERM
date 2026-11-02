import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

def gaussian_orthogonal_ensemble(N):
    H = np.random.normal(0,1/np.sqrt(2*N), (N,N))
    np.fill_diagonal(H,0)
    H = H + np.diag(np.random.normal(0,1/np.sqrt(N),N))
    H_symmetry = (H+H.T)/2
    return H_symmetry

def simulazione_montecarlo(N,n):
    autovalori_tot = []
    for i in range(n):
        H = gaussian_orthogonal_ensemble(N)
        autovalori, autovettori = np.linalg.eig(H)
        autovalori_tot.extend(autovalori)

    fig = plt.figure(figsize=(10,6))
    plt.title(f'GOE spectral density simulation with KDE \n Matrix size = {N} Toys = {n}')
    sns.kdeplot(autovalori_tot, bw_adjust=0.5, label = 'kde of GOE spectral density' )
    plt.scatter(np.sqrt(2), 0, color = 'red', label=r'predicted right edge of semicircle($\sqrt{2}$)')
    plt.scatter(-np.sqrt(2), 0, color = 'red', label=r'predicted left edge of semicircle($-\sqrt{2}$)')
    plt.legend()
    plt.show()

simulazione_montecarlo(1000, 10 )