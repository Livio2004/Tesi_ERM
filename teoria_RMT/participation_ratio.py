import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time
import linalg as lin

def genera_matrice_S(N, b0):
    X = np.random.randn(N, 3)
    tensore = X[:, np.newaxis]-X[np.newaxis, :]
    norma = np.linalg.norm(tensore, axis=-1)
    fattore = np.sqrt(N/b0)
    argomento = fattore*norma
    S = np.sinc(argomento/np.pi) #il sinc di numpy ha il pi greco
    return S
     

#X = np.random.randn(2,2)
#print(X)
#voglio accedere alle differenze di questi valori 
#print(X[np.newaxis, :])
#print(X[np.newaxis, :]-X[:, np.newaxis])


N = [300,1000,1000]
b0 = 1 

S = [genera_matrice_S(N[i], b0) for i in range(len(N))]
'''
auto = lin.linear_system(S)
autovalori, autovettori,history = auto.eigensolver_QR_numpy(1000)
'''
autovalori1, autovettori1 = eigh(S[0])
autovalori2, autovettori2 = eigh(S[1])
autovalori3, autovettori3 = eigh(S[2])

ipr3 = np.sum(autovettori3**4, axis=0) 
pr_normalizzato = 1 / (N[2] * ipr3)

plt.figure(figsize=(10, 6))


plt.scatter(autovalori3, pr_normalizzato, alpha=0.6, color='darkblue', s=10, label='N = 1000')

plt.xlabel(r'Autovalori $\lambda$')
plt.ylabel(r'IPR')
plt.title(' Participation Ratio in funzione degli autovalori')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()