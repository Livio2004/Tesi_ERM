import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import norm
from scipy.optimize import brentq




#scrivo una funzione per trovare gli intervalli proibiti al risolvente reale 
#con questo script li vedo solo graficamente
#poi studio con derivate
def eigenvalues_real(real_resolvent, alpha, fourier ):
    frac = np.sum((fourier*alpha)/(alpha-fourier*real_resolvent))
    return frac + 1/real_resolvent
def first_derivative(real_resolvent, alpha, fourier):
    frac = np.sum((np.power(fourier, 2)*alpha)/(alpha-real_resolvent*fourier)**2)
    return frac - 1/np.power(real_resolvent, 2)

#N_values = np.array([800,1000,500,100])
#L_values = np.array([50,100,500,1000])
N_values = np.array([1000,1000,500,100])
L_values = np.array([10,100,500,1000])
alpha = L_values/N_values

phi_0 = 0.2
k_modes = np.arange(1, 50)
Gamma_k = (1.0 - (-1.0)**k_modes) / (2.0 * (np.pi * k_modes)**2)
Gamma_k = np.concatenate([Gamma_k, Gamma_k]) #k negativi e positivi uguali
gamma_hat_0 = phi_0 -1/4
g_min = -30
g_max = 30


def get_poles(alpha, gamma_k, g_min, g_max):
    gamma_nonzero = gamma_k[np.abs(gamma_k) > 1e-14]
    poles = alpha/gamma_nonzero
    poles = poles[(poles>g_min) & (poles<g_max)]
    poles = np.append(poles, 0.0)
    return np.unique(np.sort(poles))



colors = ['blue', 'orange', 'green', 'red']
plt.figure(figsize=(12,7))
for alpha_i, color in zip(alpha, colors):
    poles = get_poles(alpha_i, Gamma_k, g_min,g_max)
    print(f'poles with L/N={alpha_i}  are ', poles)
    for pole in poles:
        plt.axvline(pole, color=color, linestyle='--', alpha=0.5)
    boundaries = np.concatenate(([g_min], poles, [g_max]))
    #togliamo i poli artificialmente 
    eps = 1e-5
    first_branch = True
    for left, right in zip(
        boundaries[:-1],
        boundaries[1:]
    ):
        # non tocchiamo i poli
        a = left + eps
        b = right - eps
        if a >= b:
            continue
        g_values = np.linspace(a,b,1000)
        #uso g_values per trovare zeri della derivata
        der_values = np.array([first_derivative(g,alpha_i, Gamma_k) for g in g_values])
        for j in range(len(g_values)-1):
            if der_values[j]*der_values[j+1]<0:
                root = brentq(first_derivative,g_values[j], g_values[j+1], args = (alpha_i, Gamma_k))
                estremo = eigenvalues_real(root, alpha_i, Gamma_k)
                plt.scatter(root, estremo, color = color)
                plt.axhline(estremo, color = color, linestyle='--', alpha = 0.5)
                print(f"alpha={alpha_i:.4f}, "f"G*={root:.6f}, "f"z*={estremo:.6f}")        
        z = np.array([eigenvalues_real(g_real_i, alpha_i, Gamma_k ) for g_real_i in g_values])
        plt.plot(g_values, z, color = color, label = fr'$z(G_r)$ with $\alpha$ = {alpha_i:.2f}' if first_branch else None)
        first_branch=False


plt.ylim(-1, 3)
plt.xlabel(fr'$G_r$')
plt.ylabel(fr'$z$')
plt.legend()
plt.savefig('zeri.png')
plt.show()

