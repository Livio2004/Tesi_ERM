import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.optimize import root
from scipy.optimize import brentq
plt.rcParams.update({'font.size': 18})


def resolvent_system(x, z, alpha, gamma_k):
    gi, gr = x

    frac1 = np.sum((alpha*gamma_k**2)/((alpha-gr*gamma_k)**2 +(gi*gamma_k)**2))
    frac2 = np.sum((gamma_k*alpha**2)/((alpha-gr*gamma_k)**2 +(gi*gamma_k)**2))

    F1 = frac1 -1/(np.power(gr,2)+np.power(gi,2))
    F2 = frac2 - z
    
    return np.array([F1,F2])

intervals = np.array([0])
gr_guess = np.array([])

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



poles = get_poles(alpha[1], Gamma_k, g_min,g_max)
for pole in poles:
    boundaries = np.concatenate(([g_min], poles, [g_max]))
    #togliamo i poli artificialmente 
    eps = 1e-5
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
    der_values = np.array([first_derivative(g,alpha[1], Gamma_k) for g in g_values])
    for j in range(len(g_values)-1):
        if der_values[j]*der_values[j+1]<0:
            roo = brentq(first_derivative,g_values[j], g_values[j+1], args = (alpha[1], Gamma_k))
            estremo = eigenvalues_real(roo, alpha[1], Gamma_k)
            intervals = np.append(estremo, intervals)
            gr_guess = np.append(roo, gr_guess)
            print(f"alpha={alpha[1]:.4f}, "f"G*={roo:.6f}, "f"z*={estremo:.6f}")        
    z = np.array([eigenvalues_real(g_real_i, alpha[1], Gamma_k ) for g_real_i in g_values])
    

intervals = np.sort(intervals)
print(intervals)
print(gr_guess)
gr_values = []
gr_values1 = []
gi_values = []
gi_values1 = []
rho_values = []
rho_values1 = []
z_valori = np.linspace(0.02, intervals[1]-1e-4,1000)
z_valori_1 = np.linspace(intervals[2]+0.02, intervals[3]-1e-4,1000)
guess_g = np.array([1e-2, gr_guess[0]])
guess_g_1 = np.array([1e-2, gr_guess[1]])


for z,z1 in zip(z_valori, z_valori_1):
    sol = root(resolvent_system, guess_g,args=(z, alpha[1], Gamma_k))
    sol1 = root(resolvent_system, guess_g_1,args=(z1, alpha[1], Gamma_k))
    if not sol.success:
        print('fallimento a z=', z)
        break
    if not sol1.success:
        print('fallimento a z', z1)
    gi, gr = sol.x
    gi1, gr1 = sol1.x

    gr_values.append(gr)
    gr_values1.append(gr1)
    gi_values.append(gi)
    gi_values1.append(gi1)
    rho_values.append(-gi/np.pi)
    rho_values1.append(-gi1/np.pi)

    #continuation per trovqare soluzione convergente
    guess_g = np.array([gi, gr])
    guess_g_1 = np.array([gi1, gr1])
    

plt.figure(figsize=(12,7))
plt.plot(z_valori, rho_values)
plt.plot(z_valori_1, rho_values1)

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



eigvals = np.array(matrix_M(N_values[1], L_values[1]))
plt.hist(
        eigvals,
        bins=10000,
        density=True,
        alpha=0.5,
        label=fr"ERM SIMULATION ($N={N_values[1]}$, 50 toys)" + "\n" +
              fr"$\alpha = {alpha[1]:.2f}$"
    )



plt.xlim(0, 2)
plt.yscale('log')
plt.xlabel(r"$\lambda$")
plt.ylabel("Density")
plt.legend()
plt.show()