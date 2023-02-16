# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
<Name> Trevor Wai
<Class> Section 2
<Date> 2/1/23
"""

from scipy import optimize as opt
import numpy as np

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=100):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    conv = False
    #Set the Initial minimizer approximation as the interval midpoint
    x0 = (a + b) / 2
    phi = (1 + np.sqrt(5)) / 2
    #Iterate only maxiter times at most
    for i in range(1, maxiter + 1):
        c = (b-a) / phi
        new_a = b - c
        new_b = a + c
        #Get new boundaries for the search interval
        if f(new_a) <= f(new_b):
            b = new_b
        else:
            a = new_a
        #Set the minimizer approximation as the interval midpoint
        x1 = (a + b) / 2
        if abs(x1 - x0) < tol:
            conv = True
            #Stop iterating if the apprximation stops changing enough
            break
            
        x0 = x1

    return x1, conv, i


# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=100):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Initialize
    x = x0
    #Iterate at most maxiter times
    for i in range(maxiter):
        x -= (df(x) / d2f(x))
        #Stops iterating if the approximation stops changing enough
        if abs(x0 - x) < tol:
            return x, True, i
        
        x0 = x

    return x, False, maxiter


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=100):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Initialize
    df_x1 = df(x1)
    df_x0 = df(x0)
    #Iterate Maxiter times
    for i in range(1, maxiter + 1):
        x =  (x0*df_x1 - x1*df_x0) / (df_x1 - df_x0)

        #Stop once the difference is small
        if abs(x - x1) < tol:
            return x, True, i
        
        df_x0 = df_x1
        df_x1 = df(x)
        x0 = x1
        x1 = x
    return x, False, maxiter
        


# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    #Compute these values once
    Dfp = Df(x).T @ p
    fx = f(x)

    
    while (f(x + alpha * p) > fx + (c * alpha * Dfp)):
        alpha = rho * alpha
    
    return alpha
