import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm


N = 1000
M = 10000 #numero realizzazioni
k = 5

# Scelgo v UNA VOLTA SOLA e lo fisso, può assumere qualsiasi valore
v = np.random.uniform(-1, 1, N)

X = np.zeros(M, dtype=complex)


for m in range(M):

    #disordine dato dalle r 
    r = np.random.  uniform(0, 1, N)

    u = np.exp(1j * k * r)

    X[m] = np.sum(v * u) / np.sqrt(N)



Q = np.sum(v**2) / N
print(Q)
x = X.real
sigma = np.sqrt(Q/ 2)

xx = np.linspace(x.min(), x.max(), 500)

#plot
plt.hist(x, bins=80, density=True, alpha=0.6, label="simulazione")
plt.plot(xx, norm.pdf(xx, loc=0, scale=sigma), linewidth=2,label="Gaussiana teorica")
plt.xlabel(r"$\mathrm{Re}(X_k^1)$")
plt.ylabel("densità")
plt.legend()
plt.show()