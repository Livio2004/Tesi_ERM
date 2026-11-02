import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import root, brentq

def resolvent_system(x,z,alpha,gamma_k):
    gi,gr=x
    den=(alpha-gr*gamma_k)**2+(gi*gamma_k)**2
    frac1=np.sum(alpha*gamma_k**2/den)
    frac2=np.sum(gamma_k*alpha**2/den)
    F1=frac1-1/(gr**2+gi**2)
    F2=frac2-z
    return np.array([F1,F2])

def eigenvalues_real(real_resolvent,alpha,fourier):
    frac=np.sum(fourier*alpha/(alpha-fourier*real_resolvent))
    return frac+1/real_resolvent

def first_derivative(real_resolvent,alpha,fourier):
    frac=np.sum(fourier**2*alpha/(alpha-real_resolvent*fourier)**2)
    return frac-1/real_resolvent**2

N_values=np.array([1000,1000,500,100])
L_values=np.array([10,100,500,1000])
#alpha=L_values/N_values
alpha=np.array([1e-1,5e-2,1e-2,5e-3])

phi_0=0.2
k_modes=np.arange(1,50)
Gamma_k=(1.0-(-1.0)**k_modes)/(2.0*(np.pi*k_modes)**2)
Gamma_k=np.concatenate([Gamma_k,Gamma_k])
gamma_hat_0=phi_0-1/4

g_min=-1000
g_max=1000

def get_poles(alpha,gamma_k,g_min,g_max):
    gamma_nonzero=gamma_k[np.abs(gamma_k)>1e-14]
    poles=alpha/gamma_nonzero
    poles=poles[(poles>g_min)&(poles<g_max)]
    poles=np.append(poles,0.0)
    return np.unique(np.sort(poles))

def function(phi_o=0.2,difference=float):
    return phi_o-difference

def matrix_M(N,L,toys=50):
    all_eigvals=[]
    for i in range(toys):
        r=np.random.uniform(-1/2,1/2,(N,L))
        diff=np.abs(r[:,None,:]-r[None,:,:])
        diff=np.minimum(diff,1.0-diff)
        M=np.sum(function(difference=diff),axis=2)/L
        eigvals=np.linalg.eigvalsh(M)
        all_eigvals.extend(eigvals)
    return all_eigvals

def find_seed(z,alpha,gamma_k,g_left,g_right):
    gi_guesses=[-1e-3,-1e-2,-0.1,-0.5,-1,-2,-5,-10,-20,-50,-100]
    gr_guesses=[g_left,g_right,(g_left+g_right)/2,0]

    for gi0 in gi_guesses:
        for gr0 in gr_guesses:
            sol=root(resolvent_system,[gi0,gr0],args=(z,alpha,gamma_k))

            if not sol.success:
                continue

            gi,gr=sol.x
            residual=np.linalg.norm(resolvent_system(sol.x,z,alpha,gamma_k))

            if residual<1e-7 and abs(gi)>1e-7:
                if gi>0:
                    gi=-gi
                return np.array([gi,gr])

    return None

def solve_band(z_min,z_max,g_left,g_right,alpha,gamma_k,npoints=1000):
    delta=min(1e-4,(z_max-z_min)*1e-4)
    z_left=z_min+delta
    z_right=z_max-delta
    z_mid=(z_left+z_right)/2

    seed=find_seed(z_mid,alpha,gamma_k,g_left,g_right)

    if seed is None:
        print("Nessun seed trovato:",alpha,z_min,z_max)
        return np.array([]),np.array([])

    n_left=npoints//2
    n_right=npoints-n_left

    z_values_left=np.linspace(z_mid,z_left,n_left)
    z_values_right=np.linspace(z_mid,z_right,n_right)

    z_solution_left=[]
    rho_left=[]
    guess=seed.copy()

    for z in z_values_left:
        sol=root(resolvent_system,guess,args=(z,alpha,gamma_k))

        if not sol.success:
            break

        gi,gr=sol.x

        if gi>0:
            gi=-gi

        if abs(gi)<1e-10:
            break

        z_solution_left.append(z)
        rho_left.append(-gi/np.pi)
        guess=np.array([gi,gr])

    z_solution_right=[]
    rho_right=[]
    guess=seed.copy()

    for z in z_values_right[1:]:
        sol=root(resolvent_system,guess,args=(z,alpha,gamma_k))

        if not sol.success:
            break

        gi,gr=sol.x

        if gi>0:
            gi=-gi

        if abs(gi)<1e-10:
            break

        z_solution_right.append(z)
        rho_right.append(-gi/np.pi)
        guess=np.array([gi,gr])

    z_solution=np.concatenate((np.array(z_solution_left)[::-1],np.array(z_solution_right)))
    rho_solution=np.concatenate((np.array(rho_left)[::-1],np.array(rho_right)))

    return z_solution,rho_solution


fig,axs=plt.subplots(2,2,figsize=(12,8))
axs=axs.flatten()

fig_sc,axs_sc=plt.subplots(2,2,figsize=(12,8))
axs_sc=axs_sc.flatten()

x_theory=np.linspace(-1,1,1000)
y_theory=np.sqrt(1-x_theory**2)

Gamma_unique=np.unique(Gamma_k[Gamma_k>1e-14])

for index in range(len(alpha)):
    edge_data=[]

    poles=get_poles(alpha[index],Gamma_k,g_min,g_max)
    boundaries=np.concatenate(([g_min],poles,[g_max]))
    eps=1e-5

    for left,right in zip(boundaries[:-1],boundaries[1:]):
        a=left+eps
        b=right-eps

        if a>=b:
            continue

        g_values=np.linspace(a,b,3000)
        der_values=np.array([
            first_derivative(g,alpha[index],Gamma_k)
            for g in g_values
        ])

        for j in range(len(g_values)-1):
            if not np.isfinite(der_values[j]) or not np.isfinite(der_values[j+1]):
                continue

            if der_values[j]*der_values[j+1]<0:
                roo=brentq(
                    first_derivative,
                    g_values[j],
                    g_values[j+1],
                    args=(alpha[index],Gamma_k)
                )

                estremo=eigenvalues_real(
                    roo,
                    alpha[index],
                    Gamma_k
                )

                if estremo>0 and not any(abs(estremo-e[0])<1e-7 for e in edge_data):
                    edge_data.append((estremo,roo))

    edge_data.sort(key=lambda x:x[0])

    intervals=np.array([e[0] for e in edge_data])
    gr_guess=np.array([e[1] for e in edge_data])

    print("\nalpha =",alpha[index])
    print("intervals =",intervals)
    print("gr_guess =",gr_guess)

    spectral_intervals=[]

    for j in range(0,len(intervals)-1,2):
        spectral_intervals.append((
            intervals[j],
            intervals[j+1],
            gr_guess[j],
            gr_guess[j+1]
        ))

    print("spectral intervals:")

    for z_left,z_right,g_left,g_right in spectral_intervals:
        print(f"[{z_left:.6f}, {z_right:.6f}]")

        z_values,rho_values=solve_band(
            z_left,
            z_right,
            g_left,
            g_right,
            alpha[index],
            Gamma_k,
            npoints=1000
        )

        if len(z_values)==0:
            continue

        axs[index].plot(z_values,rho_values,lw=2)

        z_center=(z_left+z_right)/2
        Gamma_est=alpha[index]*z_center

        Gamma_mode=Gamma_unique[
            np.argmin(np.abs(Gamma_unique-Gamma_est))
        ]

        m=2

        x=(alpha[index]*z_values/Gamma_mode-1)/(2*np.sqrt(m*alpha[index]))

        y=np.pi*Gamma_mode*rho_values/(
            np.sqrt(m)*alpha[index]**1.5
        )

        mask=(x>=-1.2)&(x<=1.2)&np.isfinite(y)

        axs_sc[index].plot(
            x[mask],
            y[mask],
            lw=1.8,
            label=fr"$\Gamma_k={Gamma_mode:.4g}$"
        )

        mask_rmse=(np.abs(x)<1)&np.isfinite(y)

        if np.sum(mask_rmse)>0:
            y_monasson=np.sqrt(1-x[mask_rmse]**2)
            rmse=np.sqrt(np.mean((y[mask_rmse]-y_monasson)**2))

            print(
                "alpha =",
                alpha[index],
                "Gamma =",
                Gamma_mode,
                "RMSE =",
                rmse
            )

    eigvals=np.array(
        matrix_M(
            N_values[index],
            L_values[index]
        )
    )

    axs[index].hist(
        eigvals,
        bins=10000,
        density=True,
        alpha=0.5,
        label=fr"ERM SIMULATION ($N={N_values[index]}$, 50 toys)"+"\n"+
              fr"$\alpha={alpha[index]:.2f}$"
    )

    axs[index].set_xlim(0,2)
    axs[index].set_yscale("log")
    axs[index].set_xlabel(r"$\lambda$")
    axs[index].set_ylabel("Density")
    axs[index].set_title(
        fr"$\alpha={alpha[index]:.2f}$, $N={N_values[index]}$, $L={L_values[index]}$"
    )
    axs[index].legend()

    axs_sc[index].plot(
        x_theory,
        y_theory,
        "k--",
        lw=2,
        label=r"$\sqrt{1-x^2}$"
    )

    axs_sc[index].set_xlim(-1.1,1.1)
    axs_sc[index].set_ylim(0,None)
    axs_sc[index].set_xlabel(r"$x$")
    axs_sc[index].set_ylabel(
        r"$\frac{\pi\Gamma_k}{\sqrt{2}\alpha^{3/2}}\rho$"
    )
    axs_sc[index].set_title(
        fr"Monasson collapse, $\alpha={alpha[index]:.2f}$"
    )
    axs_sc[index].legend()

fig.tight_layout()
fig_sc.tight_layout()

plt.show()