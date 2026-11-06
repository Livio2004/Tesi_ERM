import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import root, brentq
plt.rcParams.update({'font.size': 18})

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

alpha = np.logspace(-3, 1, 10)
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


plt.figure(figsize=(12,6))
plt.title(r'$\alpha$ vs $z$')
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alpha)))
label_i = True
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
        der_values=np.array([first_derivative(g,alpha[index],Gamma_k) for g in g_values])

        for j in range(len(g_values)-1):
            if not np.isfinite(der_values[j]) or not np.isfinite(der_values[j+1]):
                continue

            if der_values[j]*der_values[j+1]<0:
                roo=brentq(first_derivative,g_values[j],g_values[j+1],args=(alpha[index],Gamma_k))
                estremo=eigenvalues_real(roo,alpha[index],Gamma_k)

                if estremo>0 and not any(abs(estremo-e[0])<1e-7 for e in edge_data):
                    edge_data.append((estremo,roo))

    edge_data.sort(key=lambda x:x[0])

    intervals=np.array([e[0] for e in edge_data])
    gr_guess=np.array([e[1] for e in edge_data])

    print("\nalpha =",alpha[index])
    print("intervals =",intervals)
    print("gr_guess =",gr_guess)


    for j in range(0,len(intervals)-1,2):
        z_min = intervals[j]
        z_max = intervals[j+1]
        plt.vlines(x=alpha[index],ymin=z_min,ymax=z_max, color=colors[index],linewidth=2,alpha=0.8, label = 'forbidden intervals' if label_i else None )
        label_i = False
   

plt.xlabel(r'$\alpha$')
plt.ylabel(r'$z$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('alpha_i.png')
plt.show()




  