import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ive
from scipy.spatial.distance import cdist

# Parametri: regime in cui il confronto teoria/simulazione è ben visibile.
M = 1000
eta = 0.06
cases = [(400, 20, 11), (1000, 12, 22)]  # (N, realizzazioni, seed)

def c_spectrum(M, tol=1e-14, lmax=1000):
    """
    c_l = exp(-M)*sqrt(pi/(2M))*I_{l+1/2}(M).
    scipy.special.ive(nu,M)=exp(-M)*I_nu(M), quindi è numericamente stabile.
    """
    cs = []
    mass = 0.0
    for ell in range(lmax + 1):
        c = np.sqrt(np.pi/(2*M)) * ive(ell + 0.5, M)
        cs.append(float(c))
        mass += (2*ell + 1)*c
        if ell > 5 and abs(1-mass) < tol and (2*ell+1)*c < tol:
            break
    return np.asarray(cs)

def solve_G(lams, N, M, eta, damping=0.15, tol=1e-12, max_iter=20000):
    """
    Risolve il fixed point derivato:
      G = [z - sum_l (2l+1)c_l/(1-N c_l G)]^{-1},
      z = lambda - i eta.
    """
    cs = c_spectrum(M)
    deg = 2*np.arange(len(cs)) + 1
    out = np.empty(len(lams), dtype=complex)
    prev = None

    for k, lam in enumerate(lams):
        z = lam - 1j*eta
        G = 1/z if prev is None else prev
        for _ in range(max_iter):
            Sigma = np.sum(deg * cs / (1 - N*cs*G))
            candidate = 1/(z - Sigma)
            newG = (1-damping)*G + damping*candidate
            if abs(newG-G) < tol:
                G = newG
                break
            G = newG
        # scegliamo il ramo fisico E-J: Im G > 0 per Im z < 0
        out[k] = G
        prev = G
    return out, cs

def simulate_eigenvalues(N, M, reps, seed):
    """
    Simula il kernel originale:
       S_ij = sin(sqrt(M)|x_i-x_j|)/(sqrt(M)|x_i-x_j|),
       x_i ~ N(0,I_3).
    """
    rng = np.random.default_rng(seed)
    eigs = []
    trace_check = []
    for _ in range(reps):
        X = rng.normal(size=(N, 3))
        r = cdist(X, X)
        S = np.sinc(np.sqrt(M)*r/np.pi)
        vals = np.linalg.eigvalsh(S)
        eigs.append(vals)
        trace_check.append(np.trace(S)/N)
    return np.concatenate(eigs), np.asarray(trace_check)

rows = []
curves = {}

for N, reps, seed in cases:
    eigs, trace_check = simulate_eigenvalues(N, M, reps, seed)
    xmax = max(5.5, np.quantile(eigs, 0.999)*1.08)
    lam = np.linspace(0, xmax, 520)

    Gth, cs = solve_G(lam, N, M, eta)
    Gemp = np.array([np.mean(1/(x - 1j*eta - eigs)) for x in lam])

    rho_th = Gth.imag/np.pi
    rho_emp = Gemp.imag/np.pi

    rows.append({
        "N": N,
        "M": M,
        "realizzazioni": reps,
        "eta": eta,
        "RMSE rho_eta": np.sqrt(np.mean((rho_emp-rho_th)**2)),
        "MAE rho_eta": np.mean(np.abs(rho_emp-rho_th)),
        "media autovalori": eigs.mean(),
        "min autovalore": eigs.min(),
        "max autovalore": eigs.max(),
        "somma (2l+1)c_l": np.sum((2*np.arange(len(cs))+1)*cs),
        "N_eff": 1/np.sum((2*np.arange(len(cs))+1)*cs**2),
    })
    curves[N] = (lam, Gth, Gemp, eigs)

# Grafico 1: densità smussata
fig, ax = plt.subplots(figsize=(8, 5))
for N, _, _ in cases:
    lam, Gth, Gemp, _ = curves[N]
    ax.plot(lam, Gemp.imag/np.pi, label=f"empirica N={N}")
    ax.plot(lam, Gth.imag/np.pi, "--", label=f"teoria N={N}")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\rho_\eta(\lambda)=\mathrm{Im}\,G(\lambda-i\eta)/\pi$")
ax.set_title(fr"Kernel originale vs equazione del risolvente, $M={M:g}$, $\eta={eta:g}$")
ax.grid(alpha=0.25)
ax.legend()
plt.show()



