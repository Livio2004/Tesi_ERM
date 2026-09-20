import numpy as np
import matplotlib.pyplot as plt

N = 1500
eta = 1e-3  # regolatore  per la convergenza (guardare Monasson)
lambda_vals = np.linspace(-0.2, 0.8, 3000)
L = 1 #lunghezza toro
num = 100
all_eigvals = []
phi_0 = 0.2
for _ in range(num):
    r = np.random.uniform(-0.5, 0.5, N)
    diff = np.abs(r[:, None] - r[None, :])
    diff = np.minimum(diff, 1.0 - diff)
    M = (phi_0 - diff) / N       
    eigvals = np.linalg.eigvalsh(M)
    all_eigvals.extend(eigvals)

eigvals = np.array(all_eigvals)


# 2risolvente
k_modes = np.arange(1, 50)
Gamma_k = (1.0 - (-1.0)**k_modes) / (2.0 * (np.pi * k_modes)**2) #gamma con la funzione scelta 
# sia negatuvi sia positivi
Gamma_k = np.concatenate([Gamma_k, Gamma_k])
gamma_hat_0 = phi_0 -1/4

rho_analitica = []
s_guess = -1j / eta  # stima iniziale
# parametro per convergenza
alpha = 0.2

for z_real in lambda_vals:
    z = z_real - 1j * eta
    s = s_guess
    if s_guess is None:
        s = 1.0 / z
    else:
        s = s_guess

    for _ in range(300):
        sum_term = np.sum(Gamma_k / (1.0 - s * Gamma_k)) / N
        s_next = 1.0 / (z-sum_term)
        s = (1.0 - alpha) * s + alpha * s_next
    
    #utilizziamo la s come una bnuova guess
    s_guess = s
    rho_analitica.append(np.imag(s) / np.pi)
# plot
plt.figure(figsize=(12, 7))

plt.hist(eigvals, bins=200, density=True, alpha=0.5, color='blue', label="Simulazione ERM (N=1500 100 realizzazioni)")
plt.plot(lambda_vals, rho_analitica, 'r-', linewidth=2, label="Soluzione Analitica (Bulk)")
plt.axvline(x=Gamma_k[0], color='k', linestyle='--', label=r"$\hat{\Gamma}(\pm 1)$")
#autovalore omesso dal bulk
plt.axvline(gamma_hat_0, color='orange', linestyle='-.', label=r"$\hat{\Gamma}(0)$ (Autovalore rimosso)")
plt.yscale('log')
plt.ylim(1e-2, 5e2) # Tagliamo il grafico per escludere il rumore numerico inferiore
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\rho(\lambda)$ (Scala Logaritmica)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()