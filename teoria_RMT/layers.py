import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import norm 


def function(phi_o = 0.2, difference= (float)):
    return phi_o -difference

def matrix_M(N, L, toys = 50):
    all_eigvals = []
    for i in range(toys):
        r = np.random.uniform(-1/2, 1/2  , (N, L))
        diff = np.abs(r[:, None, :] - r[None, :, :])
        diff = np.minimum(diff, 1.0 - diff)
        M = np.sum(function(difference = diff), axis = 2) / L
        eigvals = np.linalg.eigvalsh(M)
        all_eigvals.extend(eigvals)
    return all_eigvals

N_values = [1000,1000,500,100] 
L_values = [10,100,500,1000]
colors = ['blue', 'orange', 'green', 'red']

plt.figure(figsize=(12, 7))
for N, L, color in zip(N_values, L_values, colors):

    alpha = L / N
    eigvals = np.array(matrix_M(N, L))
    plt.hist(
        eigvals,
        bins=10000,
        density=True,
        alpha=0.5,
        label=fr"ERM SIMULATION ($N={N}$, 50 toys)" + "\n" +
              fr"$\alpha = {alpha:.2f}$"
    )



plt.xlim(0, 2)
plt.yscale('log')
plt.xlabel(r"$\lambda$")
plt.ylabel("Density")
plt.legend()
plt.show()