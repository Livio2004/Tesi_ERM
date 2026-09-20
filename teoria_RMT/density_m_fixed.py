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
'''
autovalori1 = autovalori1/N[0]
autovalori2 = autovalori2/N[1]
autovalori3 = autovalori3/N[2]
'''
plt.figure(figsize = (10,6))
conteggi, bordi_bin = np.histogram(np.real(autovalori1), density = True,  bins='sturges', range=(np.min((np.real(autovalori1))),np.max((np.real(autovalori1)))))
centri_bin = (bordi_bin[:-1] + bordi_bin[1:]) / 2
plt.scatter(centri_bin, conteggi,  marker = 'o', alpha = 0.8, label = f'N = {N[0]}')
plt.hist(np.real(autovalori2), histtype='step', density = True, bins='sturges', range=(np.min((np.real(autovalori2))),np.max((np.real(autovalori2)))), label =f'N = {N[1]}', alpha = 0.4 )
plt.hist(np.real(autovalori3),  bins='sturges', density = True, range=(np.min((np.real(autovalori3))),np.max((np.real(autovalori3)))), label = f'N = {N[2]}', alpha = 0.4)
plt.axvline(1, ls = '--', label = 'expected mean', color = 'blue')
plt.axvline(np.mean(autovalori3), ls = '--', label = f'numerical mean(con N= {N[2]})', color = 'purple')

plt.legend()

plt.xlabel(r'$\lambda$')
plt.ylabel(r'$p(\lambda)$')



plt.show()