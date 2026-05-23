import numpy as np
import matplotlib.pyplot as plt 
from numpy import linalg as LA
from scipy.stats import norm


def check_symmetry(A):
        for i in range(A.shape[0]):
            for j in range(i+1,A.shape[0]):
                    if A[i,j]!=A[j,i]:
                        print("La matrice non è simmetrica")
                        return False
                
        print("la matrice è simmetrica")
        return True

def check_hermitian(A):
        for i in range(A.shape[0]):
            for j in range(i+1,A.shape[0]):
                    if A[i,j]!=A[j,i].conj():
                        print("La matrice non è simmetrica")
                        return False
                
        print("la matrice è hermitiana")
        return True

n = 8
H = np.random.normal(0,1, size = (n,n))+ 1j * np.random.normal(0,1, size=(n,n))


H_simmetrica = (H+H.conj().T)/2

print(H_simmetrica)

check_symmetry(H_simmetrica)

check_hermitian(H_simmetrica)

T = 50000
eigenvalues = np.array(0)
#in questo caso GOE gaussian orthoganal ensemble
for i in range (T):
    H = np.random.normal(0,1, size= (n,n))
    H_simmetrica = (H+H.T)/2
    eigen_singol = LA.eig(H_simmetrica)[0]
    eigenvalues = np.concatenate((eigenvalues, eigen_singol), axis = None)

plt.hist(eigenvalues, bins = 'sturges', density= True, label= 'GOE N=8 T =50000')
plt.legend()
plt.show()

def compute_gaussian_jpdf(matrix):

    N_squared = matrix.size
    exponent = -0.5 * np.sum(matrix**2)
    normalization = (1 / np.sqrt(2 * np.pi))**N_squared
    
    return normalization * np.exp(exponent)

#  Definiamo la dimensione della matrice (N x N)
N = 100
H = np.random.randn(N, N)
H_sym = (H + H.T) / np.sqrt(2)
entries = H_sym.flatten()
rho_H = compute_gaussian_jpdf(H_sym)

print(f"\nValore della jpdf rho[H]: {rho_H:.2e}")

plt.hist(entries, bins=50, density=True, color='lightgray', edgecolor='black', alpha=0.7)
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, 0, 1), 'r-', lw=3, label='Gaussiana Teorica')
plt.title("Distribuzione delle entrate H_ij")
plt.xlabel("Valore di H_ij")
plt.ylabel("Frequenza")
plt.legend()

plt.tight_layout()
plt.show()
