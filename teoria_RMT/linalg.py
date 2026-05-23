import numpy as np
import matplotlib.pyplot as plt
import time 
import sys 
from numba import jit 
from sympy import *


class linear_system:
    def __init__(self, A, b=None):
        self.A = np.array(A, dtype=np.complex128)
        if b is None:
            self.b = np.array(self.A.shape[0])
        self.b = np.array(b, dtype=np.float64)
        self.n = self.A.shape[0]
        self.L = None
        self.P = 0
        self.Q = None
        self.R = None

    def check_symmetry(self):
        for i in range(self.n):
            for j in range(i+1,self.n):
                    if self.A[i,j]!=self.A[j,i]:
                        print("La matrice non è simmetrica")
                        return False
                
        print("la matrice è simmetrica")
        return True
    
    def cholesky_reduction(self):
        if not self.check_symmetry():
            raise ValueError("La matrice non è simmetrica!")
        
        L = np.zeros((self.n, self.n))
        for i in range(self.n):
          for j in range(0, i+1):
            s = np.sum(np.power(L[i,:j],2))
            if i==j :
                 L[i,j]=np.sqrt(self.A[i,i]-s)
            if i >j :
                 s_ij = np.sum(L[i, :j]*L[j, :j] )
                 L[i,j]=(1/L[j,j])*(self.A[i,j]-s_ij)
        self.L = L
        return self.L
    
    def backward_substition(self, A, b):
    #Checkiamo se la matrice A è triangolare superiore
        if not np.allclose(A, np.triu(A)):
            raise ValueError("La matrice A deve essere triangolare superiore")
        x = np.zeros(self.n)
        for i in range(self.n-1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:]))/A[i,i]
        return x
    
    def forward_substition(self, A,b):
        x = np.zeros(self.n)
        for i in range(self.n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]))/A[i,i]
        return x

    def gaussian_elimination(self, A, b):
    #bisogna fare il ciclo sulle colonne e poi sulle righe
        for j in range(self.n):
            for i in range(j+1,self.n):
                c = A[i,j]/A[j, j]
                A[i,j:]=A[i, j:]-c*A[j,j:]
                b[i]=b[i]-c*b[j]
        return A, b

    def gaussian_elimination_pivoting(self, A, b):
        #bisogna fare il ciclo sulle colonne e poi sulle righe
        for j in range(self.n):
            pivot_riga = np.argmax(A[j:,j])+j
            print(pivot_riga)

        if (np.abs(A[pivot_riga, j])<10e-12):
            print('ciao')
            raise ValueError('La matrice è singolare ')
        
        if (pivot_riga != j):
            A[[j,pivot_riga]]= A[[pivot_riga, j]]
            b[[j,pivot_riga]]=b[[pivot_riga,j]]

        for i in range(j+1,self.n):
            c = A[i,j]/A[j, j]
            A[i,j:]=A[i, j:]-c*A[j,j:]
            print(A)
            b[i]=b[i]-c*b[j]
        return A, b
    
    def solve_cholensky(self):
        #risoluzione generale del sistema lineare con cholesky
        if self.L is None:
            self.cholesky_reduction()
        
        # 1. Ly = b
        y = self.forward_substition(self.L, self.b)
        # 2. L^T x = y
        x = self.backward_substition(self.L.T, y)
        return x
    
    '''def householder(self, R):
        u = R[:,0].copy()
        n = np.sqrt(u@u)
        if (u[0] + n) > (u[0]-n):
            s = +1
        else:
            s = -1
        den = n**2 + s * u[0] * n
        u[0] += s * n
        return np.eye(len(R)) - np.outer(u,u) / den'''
    
    def householder(self, R):
        u = R[:, 0].copy().astype(complex)
        n = np.linalg.norm(u)
        
        if n == 0:
            return np.eye(len(R), dtype=complex)

        if np.abs(u[0]) > 1e-15:
            s = u[0] / np.abs(u[0])
        else:
            s = 1.0 
        u[0] += s * n
        
        norma_u_quadra = np.real(u.conj().T @ u)
        
        if norma_u_quadra < 1e-15:
            return np.eye(len(R), dtype=complex)
        return np.eye(len(R), dtype=complex) - 2.0 * np.outer(u, u.conj()) / norma_u_quadra

    def QR_decomposition(self, A):
        n = self.A.shape[0]
        R = A.copy()
        Q = np.eye(n)
        self.P = 0
        for i in range(n-1):
            P = np.eye(n, dtype = np.complex128)
            Pp = self.householder(R[i:, i:])
            P[i:,i:] = Pp
            R = P@R
            Q = Q@P
            self.P+=1
        self.Q = Q
        self.R = R
        return Q, R
    
    def solve_QR(self):
        Q, R = self.QR_decomposition()
        b = Q.T @ self.b
        x = self.backward_substition(R, b)
        return x

    
    def verifica_sistema_cholesky(self):
        verifica = np.dot(self.A, self.solve_cholensky())
        residuo= np.linalg.norm(verifica-self.b)
        print(f"Il residuo per il sistema con Cholensky vale {residuo}")
        return None
    
    def verifica_sistema_QR(self):
        verifica = np.dot(self.A, self.solve_QR())
        residuo= np.linalg.norm(verifica-self.b)
        print(f"Il residuo per il sistema con QR vale {residuo}")
        return None
    
    def cholesky_determinant(self):
        if self.L is None:
            self.cholesky_reduction()
        diag_elements = np.diag(self.L)
        # Calcoliamo il prodotto dei quadrati
        determinante = np.prod(diag_elements**2)
    
        return determinante
    def QR_determinant(self):
        #uso binet per q e R
        detQ = np.power(-1, self.P)
        detR = np.prod(np.diag(self.R))
        return detQ*detR
    
    def eigensolver_QR(self,N, tol = 1e-16):
        N = int(N)
        Ak = self.A.copy().astype(complex)
        Qk = np.eye(self.n, dtype= complex)
        history = []
        for i in range(N):
            Q,R = self.QR_decomposition(Ak)
            Ak = np.dot(R, Q)
            Qk = np.dot(Qk, Q)
            #diag = np.abs(np.diag(Ak))
            #subdiag = np.abs(np.diag(Ak, k=-1))
            tridig = np.abs(np.tril(Ak, k=-1))
            history.append(np.sort(np.diag(Ak)))
            #somma_adiacenti = diag[:-1] + diag[1:]
            if np.max(tridig)<= tol:
                print(f"Convergenza relativa raggiunta all'iterazione {i}")
                break
        autovalori = np.diag(Ak)    
        autovettori = []
        for j in range(Qk.shape[1]):
            autovettore_j = Qk[:, j]
            autovettori.append(autovettore_j)
        autovettori = np.array(autovettori)
        return autovalori, autovettori, np.array(history)
    
    def eigensolver_QR_numpy(self,N, tol = 1e-16):
        N = int(N)
        Ak = self.A.copy().astype(complex)
        Qk = np.eye(self.n, dtype= complex)
        history = []
        for i in range(N):
            Q,R = np.linalg.qr(Ak)
            Ak = np.dot(R, Q)
            Qk = np.dot(Qk, Q)
            #diag = np.abs(np.diag(Ak))
            #subdiag = np.abs(np.diag(Ak, k=-1))
            tridig = np.abs(np.tril(Ak, k=-1))
            history.append(np.sort(np.diag(Ak)))
            #somma_adiacenti = diag[:-1] + diag[1:]
            if np.max(tridig)<= tol:
                print(f"Convergenza relativa raggiunta all'iterazione {i}")
                break
        autovalori = np.diag(Ak)    
        autovettori = []
        for j in range(Qk.shape[1]):
            autovettore_j = Qk[:, j]
            autovettori.append(autovettore_j)
        autovettori = np.array(autovettori)
        return autovalori, autovettori, np.array(history)
    

    def residui_QR(self, max_iter=1000, tol=1e-16):
        target_vals,_, storia = self.eigensolver_QR(max_iter, tol)
        
        plt.figure(figsize=(10, 6))
        for a in range(self.n):
            residui = np.abs(storia[:, a] - target_vals[a])
            #per non far impazzire matplot che log0 va all'infinito
            residui[residui < 1e-16] = np.nan 
            plt.semilogy(residui, label=f'λ_{a} ≈ {target_vals[a]:.2f}')
        plt.xlabel('Iterazioni')
        plt.ylabel('Residui')
        plt.title('Convergenza degli Autovalori (Metodo QR)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()



class interpolazioni:
    def __init__(self, x, f):
        self.x = x
        self.y = f
        diff = x[:, None]-x[None,:]
        np.fill_diagonal(diff, 1.0)
        denominatori = np.prod(diff, axis=1)
        self.weights_lagrange = 1.0/denominatori

    def lagrange_interpolation(self,x):
        interpolazione = []
        for i in x:
            distanza = i - self.x + 1e-16
            numeratore = np.sum((self.y * self.weights_lagrange) / distanza)
            denominatore = np.sum(self.weights_lagrange / distanza)
            risultato = numeratore / denominatore
            interpolazione.append(risultato)
        return interpolazione
    
class discrete_fourier:
    def __init__(self, L, N):
        self.dx = L/N
        self.dp = (2 * np.pi) / L
        self.x = np.linspace(0,L,N, endpoint=False)
        self.N = N
        mezzo = N // 2
        k_standard = np.concatenate((np.arange(0, mezzo), np.arange(-mezzo, 0)))
        self.p = k_standard * self.dp
        self.p_ordinato = np.concatenate((self.p[mezzo:], self.p[:mezzo]))
        self.W = np.exp(-1j * self.p[:, None] * self.x[None, :])
        self.W_dagger = self.W.conj().T
#con i dx non va
    def fourier(self,fi):
        return np.dot(self.W,fi)*self.dx
    
    def antifourier(self,f_tilde):
        return np.dot(self.W_dagger,f_tilde)*(self.dp/(2*np.pi))
    
def fit_lineare_generale(X, y, C):
    W = np.linalg.inv(C)
    M = X.T @W@X
    risultati = X.T@W@y
    sistema = linear_system(M, risultati)
    L = sistema.cholesky_reduction()
    z = sistema.forward_substition(L, risultati)
    alpha = sistema.backward_substition(L.T, z)
    I = np.eye(len(alpha))
    L_inv = np.zeros_like(L)
    for i in range(len(alpha)):
        L_inv[:, i] = sistema.forward_substition(L, I[:, i])
    
    C_alpha = L_inv.T@L_inv
    residui = y - (X @ alpha)
    chi2 = residui.T @ W @ residui

    return alpha, C_alpha, chi2

class root_functions:
    def __init__(self, a, b, func):
        self.a = a
        self.b = b
        self.fa = func(a)
        self.fb = func(b)
        self.func = func
    
    def bisezione(self, tol = 1e-10, max_iter=1000):
        a = self.a
        b = self.b
        fa = self.fa
        fb = self.fb
        c = (a+b)/2
        ci = []

        if fa*fb>0:
            print('Devi scegliere due valori per cui hanno valori di segno opposto')
            return None
        else:
            for i in range(max_iter):
                if (np.abs(self.func(c))>tol):
                    c = (a+b)/2
                    ci.append(c)
                    if (self.func(c)*fa>0):
                        a = c
                    else :
                        b = c
                else:
                    return c, np.array(ci)
            print(f'Raggiunto numero massimo di iterazioni={max_iter}')
            return c, np.array(ci)
        

    def regula_falsi(self, tol=1e-10, max_iter = 1000):
        a = self.a  
        b = self.b
        fa = self.fa
        fb = self.fb
        if fa * fb > 0:
            print('Devi scegliere due valori per cui la funzione ha segno opposto.')
            return None
        c = (a * fb - b * fa) / (fb - fa)
        fc = self.func(c)
        ci = []
        for i in range(max_iter):
            if np.abs(fc) > tol:
                if fa * fc > 0:
                    a = c
                    fa = fc
                else:
                    b = c
                    fb = fc  
                c = (a * fb - b * fa) / (fb - fa)
                fc = self.func(c)
                ci.append(c)
            else:
                return c, np.array(ci)
        
        print(f'Raggiunto numero massimo di iterazioni={max_iter}')
        return c, np.array(ci)

    def controllo_iterazioni_bisezione(self,tol=1e-10):
        return np.ceil(np.log2((self.b-self.a)/tol)) 

def newton(x0, func,max_iter = 2000,  prec = 1e-10):
    history = [] 
    xs = symbols('x')
    if isinstance(func, Expr):
        print("Tratto la funzione come SIMBOLICA")
        der_sym = diff(func, xs)
        f_numerica= lambdify(xs, func, 'numpy')
        df_numerica = lambdify(xs, der_sym, 'numpy') 
    else:
        print("Tratto la funzione come NUMERICA")
        f_numerica = func
        # differenze finite del bruno lezione 2
        h = 1e-8
        df_numerica = lambda x: (func(x + h) - func(x)) / h
    x_n = x0
    for i in range(max_iter):
        f_val = f_numerica(x_n)
        if np.max(np.abs(f_val)) < prec:
            return x_n, np.array(history)
        df_val = df_numerica(x_n)
        if np.max(np.abs(df_val)) < 1e-15:
            print("La derivata è quasi nulla metodo non converge")
            return None, np.array(history)
        x_next = x_n - f_val / df_val
        history.append(x_next)
        if np.max(np.abs(x_next - x_n)) < prec:
            return x_next, np.array(history)
        x_n = x_next
        
    print(f"Raggiunto il numero massimo di iterazioni= {len(history)}")
    return x_n, np.array(history) 


def Hermite(n):
    if n==0 :
        print('I polinomi di hermite partono dal grado 1')
    else:

        H0= 1
        H00 = 0
        x = symbols('x')
        for i in range(n):
            Hfinale = 2*H0*x-2*(i)*H00
            H00=H0
            H0=Hfinale
        return simplify(Hfinale), np.flip(np.array(Poly((Hfinale)).all_coeffs())), lambdify(x,Hfinale, 'numpy')



def Hermite_coeff(n):
    if n ==0:
        print('I polinomi di hermite partono dal grado 1')
    else :
        H0 = np.zeros(n+1)
        H00 = np.zeros(n+1)
        H0[0]=1
        for i in range(n):
            H= 2*np.roll(H0,1)-2*i*H00
            H00=H0
            H0 = H
        return H

class equazioni_differenziali:
    def __init__(self, xi, df, y0, h):
        self.h = h
        self.func = df
        self.n_steps = int(round((xi[1] - xi[0]) / h)) + 1
        self.t = np.linspace(xi[0], xi[1]+h, self.n_steps)
        self.y0 = np.atleast_1d(y0)
        self.y = np.zeros((len(self.y0),len(self.t)), dtype = np.complex128)
        #print(np.size(self.y))

    def g(self, y0, i):
        oggetto = np.array(self.func(*(y0[:, i]),self.t[i]))
        return self.h*oggetto

    def forward_euler(self):
        t = self.t
        y = self.y
        y0 = self.y0
        y[:,0]= y0
        #print(self.t)
        #print(len(self.t))
        #print(y[:,0])
        for n in range(len(t)-1):
            y[:, n+1] = y[:, n]+self.g(y,n)
        #print(f'Eulero con {self.h} ha fatto {len(t)} iterazioni')
        return np.array(t), np.array(y) 
    
    def backward_euler(self):  #DA RIVEDERE PER GENERALIZZARE ARRAY
        t = self.t
        y = self.y
        y0 = self.y0
        y[:,0]= y0
        for n in range(len(t)-1):
            def funzione(incoglionita):
                return y[n]+self.h*self.func(t[n+1],incoglionita)-incoglionita
            solci_iter, _ = newton(y[n],funzione)
            #print(y)
            y[n+1]= solci_iter
        return t, y       
    
    def rk2(self):

        t = self.t
        y = self.y
        y0 = self.y0
        y[:,0]= y0
        h = self.h

        for n in range(len(t)-1):
            k1 = np.array(self.func(*y[:, n], t[n]))
            y_mid = y[:, n] + (h / 2) * k1
            k2 = np.array(self.func(*(y_mid),self.t[n]+0.5*h))
            y[:, n+1] = y[:, n]+h*k2
        #print(f'con rk2 con {h} ha fatto {len(t)} iterazioni')
        return t, y 

    def rk4(self):

        t = self.t
        y = self.y
        y0 = self.y0
        y[:,0]= y0
        h = self.h
        for n in range(len(t)-1):
            k1 = np.array(self.func(*y[:, n], t[n]))
            #print(self.func(*y[:,n],t[n]))
            #print(y[:, n])
            y_mid = y[:, n] + (h / 2) * k1
            k2 = np.array(self.func(*(y_mid),self.t[n]+0.5*h))
            y_mid2 = y[:, n] + (h / 2) * k2
            k3 = np.array(self.func(*(y_mid2),self.t[n]+0.5*h))
            y_mid3 = y[:, n] + h * k3
            k4 = np.array(self.func(*(y_mid3),self.t[n]+h))
            y[:, n+1] = y[:, n] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        #print(f'con rk4 con {h} ha fatto {len(t)} iterazioni')
        return t, y
    
    def symplectic_euler(self):
        t = self.t
        y = self.y
        y[:, 0] = self.y0
        h = self.h

        for n in range(len(t) - 1):
            derivate = self.func(*y[:, n], t[n])
            y[1, n+1] = y[1, n] + h * derivate[1]
            y[0, n+1] = y[0, n] + h * y[1, n+1]

        return t, y



    def rk4_schrodinger(self):

        t = self.t
        y = self.y
        y0 = self.y0
        y[:,0]= y0
        h = self.h
        for n in range(len(t)-1):
            k1 = np.array(self.func(y[:, n], t[n]))
            #print(self.func(*y[:,n],t[n]))
            #print(y[:, n])
            y_mid = y[:, n] + (h / 2) * k1
            k2 = np.array(self.func((y_mid),self.t[n]+0.5*h))
            y_mid2 = y[:, n] + (h / 2) * k2
            k3 = np.array(self.func((y_mid2),self.t[n]+0.5*h))
            y_mid3 = y[:, n] + h * k3
            k4 = np.array(self.func((y_mid3),self.t[n]+h))
            y[:, n+1] = y[:, n] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        #print(f'con rk4 con {h} ha fatto {len(t)} iterazioni')
        return t, y
    


        
def forward_euler(xi, df, y0, h):
    i =  xi[0]
    x = [0]
    solci = [y0]
    while i<=xi[-1]:
        solci_iter = y0+h*df(i,y0)
        solci.append(solci_iter)
        y0 = solci_iter
        i+=h
        x.append(i)
    print(f'con il forward algorithm con {h} ha fatto {len(x)} iterazioni')
    return np.array(x), np.array(solci)  
    
'''
def forward_euler(t_range, df, y0, h):
    t = np.arange(t_range[0], t_range[1] + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(len(t) - 1):
        y[n+1] = y[n] + h * df(t[n], y[n])
    return t, y              
    '''


def backward_euler(xi, df, y0, h):

    t = np.arange(xi[0], xi[1]+h, h)
    y = np.zeros(len(t))
    y[0]= y0
    for n in range(len(t)-1):
        def func(incoglionita):
            return y[n]+h*df(t[n+1],incoglionita)-incoglionita
        solci_iter, _ = newton(y[n],func)
        #print(y)
        y[n+1]= solci_iter
    return t, y       

def laplaciano(N, A = 0, B = 0):
    laplaciano = -2*np.eye(N) + np.eye(N, k = 1) + np.eye(N, k = -1) #approx derivata seconda
    laplaciano[N-1, 0] = B
    laplaciano[0, N-1] = A
    return laplaciano