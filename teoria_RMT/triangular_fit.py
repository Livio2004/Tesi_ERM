import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
import linalg as lin
import seaborn as sns 
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde


def genera_matrice_S(N, b0):
    X = np.random.randn(N, 3)
    tensore = X[:, np.newaxis]-X[np.newaxis, :]
    norma = np.linalg.norm(tensore, axis=-1)
    fattore = np.sqrt(N/b0)
    argomento = fattore*norma
    S = np.sinc(argomento/np.pi) #il sinc di numpy ha il pi greco
    return S


def triangolare(x, a):
    # mettere i negatuvu a 0 per scipys
    res = (a - np.abs(x - 1)) / a**2
    return np.maximum(res, 0)

campionamenti = 10
N = 1000
b0_valori = [0.005]
b0_valori_fit = np.linspace(0.001,0.30, 20)
fig, ax = plt.subplots(nrows=1, ncols = len(b0_valori), figsize = (12,6))
fig.suptitle('Densità di autovalori triangolare con simulazione montecarlo', fontweight= 'bold')

tutti_autovalori = []
for _ in range(campionamenti):
    mat = genera_matrice_S(N,b0_valori[0])
    autovalori, autovettori = eigh(mat)
    tutti_autovalori.extend(autovalori) #bisogna usare extend perchè append aggiunge ogni volta nuovi blocchi

kde_matematico = gaussian_kde(tutti_autovalori)
bw_adjust = 0.5
kde_matematico.covariance_factor = lambda: kde_matematico.scotts_factor() * bw_adjust
kde_matematico._compute_covariance() # Ricalcola la covarianza interna con il nuovo fattore

centri_bin = np.linspace(min(tutti_autovalori), max(tutti_autovalori), 800)
conteggi = kde_matematico(centri_bin)

sns.kdeplot(tutti_autovalori, ax = ax , fill=True, color='darkblue', bw_adjust=0.5, label = f'b0 ={b0_valori[0]} ')
ax.set_xlabel(r'$\lambda$ ')
ax.set_ylabel(r'$p(\lambda)$')
ax.legend()

popt, pcov = curve_fit(triangolare, centri_bin, conteggi, p0=[1.0] )
print(popt)
print('valore expected =', np.sqrt((3*b0_valori[0])/2))
ax.hist(autovalori, label = '1 run',  bins = 40, density = True, histtype = 'step' )
ax.plot(centri_bin, triangolare(centri_bin,*popt), label = 'fit')




plt.legend()
plt.tight_layout()
plt.show()

a_fit = []

for b0 in b0_valori_fit:
    tutti_autovalori = []
    for _ in range(campionamenti):
        # Esempio: matrice simmetrica casuale (Gaussian Orthogonal Ensemble)
        mat = genera_matrice_S(N,b0)
        autovalori, autovettori = eigh(mat)
        tutti_autovalori.extend(autovalori) #bisogna usare extend perchè append aggiunge ogni volta nuovi blocchi
    kde_matematico = gaussian_kde(tutti_autovalori)
    bw_adjust = 0.5
    kde_matematico.covariance_factor = lambda: kde_matematico.scotts_factor() * bw_adjust
    kde_matematico._compute_covariance() # Ricalcola la covarianza interna con il nuovo fattore

    centri_bin = np.linspace(min(tutti_autovalori), max(tutti_autovalori), 800)
    conteggi = kde_matematico(centri_bin)
    popt, pcov = curve_fit(triangolare, centri_bin, conteggi, p0=[1.0] )
    a_fit.append(popt[0])

def retta(x, m, q):
    return m*x+q


popt, pcov = curve_fit(retta, np.sqrt(b0_valori_fit), a_fit)

plt.scatter(np.sqrt(b0_valori_fit), a_fit)
plt.plot(np.sqrt(b0_valori_fit), retta(np.sqrt(b0_valori_fit), *popt), label = f'fit retta con m = {popt[0]:.3f} e q = {popt[1]:.3f}')

plt.xlabel('sqrt(b0)')
plt.ylabel('a')



plt.legend()
plt.tight_layout()
plt.show()









