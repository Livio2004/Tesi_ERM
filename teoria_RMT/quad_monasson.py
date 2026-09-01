import numpy as np
import matplotlib.pyplot as plt

N = 1200
eta = 1e-3  # regolatore  per la convergenza (guardare Monasson)
lambda_vals = np.linspace(-0.2, 0.8, 3000)

np.random.seed(42)
r = np.random.uniform(-0.5, 0.5, N)
diff = np.abs(r[:, None] - r[None, :])
diff = np.minimum(diff, 1 - diff) # condizioni periodiche vogliamo invarianza per traslazione
M = 1 - 4 * diff**2
eigvals = np.linalg.eigvalsh(M) / N

# 2risolvente
k_modes = np.arange(1, 50)
Gamma_k = 2 * (-1)**(k_modes + 1) / (np.pi**2 * k_modes**2) #gamma con la funzione scelta 
# sia negatuvi sia positivi
Gamma_k = np.concatenate([Gamma_k, Gamma_k]) 

rho_analitica = []
for z_real in lambda_vals:
    z = z_real + 1j * eta
    s = -1j #guess per risoluzione equazione
    # fixed-point iteration per trovare s(z)
    for _ in range(100):
        sum_term = np.sum(Gamma_k / (1 + s * Gamma_k)) /N
        s = 1 / (sum_term - z)
    rho_analitica.append(np.imag(s) / np.pi)

# plot
plt.figure(figsize=(12, 7))

plt.hist(eigvals, bins=200, density=True, alpha=0.5, color='blue', label="Simulazione ERM (N=1200)")
#plt.plot(eigvals, np.full_like(eigvals, 1e-2), '|', color='midnightblue', markersize=10, label="Autovalori discreti esatti")
plt.plot(lambda_vals, rho_analitica, 'r-', linewidth=2, label="Soluzione Analitica (Bulk)")
plt.axvline(x=Gamma_k[0], color='k', linestyle='--', label=r"$\hat{\Gamma}(\pm 1)$ termodinamico")
#autovalore omesso dal bulk
plt.axvline(x=2/3, color='orange', linestyle='-.', label=r"$\hat{\Gamma}(0)$ (Autovalore estensivo)")
plt.yscale('log')
plt.ylim(1e-2, 5e2) # Tagliamo il grafico per escludere il rumore numerico inferiore
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\rho(\lambda)$ (Scala Logaritmica)")
plt.legend()
plt.title("Confronto Spettro ERM: L=1 rivela i picchi discreti nel rumore")
plt.grid(True, alpha=0.3)
plt.show()