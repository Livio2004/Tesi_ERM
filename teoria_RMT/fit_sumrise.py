import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
import linalg as lin
import seaborn as sns 
from scipy.special import gamma
from scipy.optimize import curve_fit

def wigner_sumrise(s, q , r): #la vogliamo confrontare con wigner-dyson nel bulk centrale
    a = (r*(gamma((q+2)/r))**(q+1))/gamma((q+1)/r)**(q+2)
    b = (gamma((q+2)/r)/gamma((q+1)/r))**(r)

    return (a*s**q)*np.exp(-1*b*s**r)

def genera_matrice_S(N, b0):
    X = np.random.randn(N, 3)
    tensore = X[:, np.newaxis]-X[np.newaxis, :]
    norma = np.linalg.norm(tensore, axis=-1)
    fattore = np.sqrt(N/b0)
    argomento = fattore*norma
    S = np.sinc(argomento/np.pi) #il sinc di numpy ha il pi greco
    return S

N = 5000
b0_valori = [0.1,1,3,10]
fig, axes = plt.subplots(nrows=1, ncols = 4, figsize = (12,6))
fig.suptitle('fit wigner sumrise del bulk', fontweight = 'bold')

for ax, b0 in zip(axes, b0_valori):
    S = genera_matrice_S(N, b0)
    autovalori, autovettori = eigh(S)
    autovalori = np.sort(autovalori)
    numero_finestre = 30
    step = 50
    n_window = 400
    results = []
    for centro in range(0, len(autovalori)-numero_finestre, step):
        finestrella = autovalori[centro:centro+n_window]

        spacings = np.diff(np.sort(finestrella))
        spacings = spacings / np.mean(spacings) # RICORDARSOI MNORMALIZZAZIONE
        conteggi, bordi_bin = np.histogram(np.real(spacings),bins = 'sturges', range=(np.min((np.real(spacings))),np.max((np.real(spacings)))) , density = True)
        centri_bin = (bordi_bin[:-1] + bordi_bin[1:]) / 2


        try:
            popt, _ = curve_fit(wigner_sumrise, centri_bin, conteggi , p0 = [1.0,1.0])
            results.append((np.mean(finestrella), popt[0], popt[1]))
        except RuntimeError:
            continue #I fit che non convergono li togliamo tanto a noi ci interessa il plot paper
    
    results = np.array(results)
    ax.errorbar(results[:, 0], results[:, 1], label='q', fmt='o', markersize=3)
    ax.errorbar(results[:, 0], results[:, 2], label='r', fmt='s', markersize=3)
    ax.set_title(f"b0 = {b0}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

