# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Name> Trevor Wai
<Class> Section 1
<Date> 4/5/23
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    #Starting points
    x, lamb, mu = starting_point(A, b, c)
    m, n = A.shape
    #Problem 1 Function F
    def _F(x, lamb, mu):
        M = np.diag(mu)
        return np.concatenate((A.T@lamb + mu - c, A @ x - b, M @ x))
    
    #Problem 2 searh direction
    def _search_direction():
        zero = np.zeros((m,m))
        I = np.eye(n)
        DF = np.block([[np.zeros((n,n)), A.T, I], [A, zero, np.zeros_like(A)], [np.diag(mu), np.zeros((n,m)), np.diag(x)]])
        #Otherside of the equal sign
        b = -_F(x, lamb, mu) + np.concatenate((np.zeros(n), np.zeros(m), (1/10) * (x.T @ mu / n) * np.ones(n)))

        return la.lu_solve(la.lu_factor(DF), b)
    
    direction = _search_direction()

    #Problem 3 Compte the step size
    def _directional(direction):
        delta_x = direction[:n]
        delta_mu = direction[n+m:]
        #Alpha max and Delta max
        alpha = np.min(-mu / delta_mu, where=delta_mu<0, initial=1)
        delta = np.min(-x/delta_x, where=delta_x<0, initial=1)
        #Alpha, Delta min
        alpha = min([1, 0.95 * alpha])
        delta = min([1, 0.95 * delta])
        return alpha, delta
    
    for i in range(niter):
        direction = _search_direction()

        #Delta Values
        delta_x = direction[:n]
        delta_lamb = direction[n:n+m]
        delta_mu = direction[n+m:]

        #Alpha Delta Values
        alpha, delta = _directional(direction)

        #Update x, lambda, mu, optimal value, and nu
        x += delta * delta_x

        lamb += alpha * delta_lamb

        mu += alpha * delta_mu

        val = c.T @ x

        nu = (x.T @ mu) / n

        #Break if duality measure is less than tol
        if nu < tol:
            break

    
    return x, val
    


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    data = np.loadtxt(filename, dtype=float)

    #Initialize the vectors c and y
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]

    #Initialiing the Constraint matrix correctly
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    #Calcuate the solution by calling interior point function
    sol = interiorPoint(A, y, c, niter=10)[0]

    #Extract values of Beta
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]

    #Plots
    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    domain = np.linspace(0,10,200)
    plt.scatter(data[:,1], data[:,0], label='Data')
    plt.plot(domain, domain*slope + intercept, 'g', label='Least Squares')
    plt.plot(domain, domain * beta + b, 'red', label='LAD')
    plt.legend()
    plt.title('Simdata')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()