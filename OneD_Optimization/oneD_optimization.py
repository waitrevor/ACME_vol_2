# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
<Name> Trevor Wai
<Class> Section 2
<Date> 2/1/23
"""
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
    x0 = (a + b) / 2
    phi = (1 + np.sqrt(5)) / 2
    for i in range(1, maxiter + 1):
        c = (b-a) / phi
        new_a = b - c
        new_b = a + c
        if f(new_a) <= f(new_b):
            b = new_b
        else:
            a = new_a
        x1 = (a + b) / 2
        if abs(x0 - x1) < tol:
            break
            
        x0 = x1

    return x1


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
    x = x0
    for i in range(maxiter):
        x -= (df(x) / d2f(x))

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
    dx1 = df(x1)
    dx0 = df(x0)
    x = x1 - ((x1*dx1 - x0*dx0) / (dx1 - dx0))
    
    for i in range(maxiter):
        dx = df(x)
        xtemp = x
        x -= (x * dx - x1 * dx1) / (dx - dx1)
        if abs(xtemp - x) < tol:
            return x, True, i
        x1 = xtemp
        dx1 = dx
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
    Dfp = Df(x).T @ p
    fx = f(x)

    while (f(x + alpha * p) > fx + (c * alpha * Dfp)):
        alpha = rho * alpha
    
    return alpha
