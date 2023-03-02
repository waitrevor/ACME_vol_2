# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
<Name> Trevor Wai
<Class> Section 1
<Date> 2/27/23
"""

import scipy.optimize as opt
import scipy.linalg as la
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    
    for i in range(maxiter):
        #Implements the exact method of steepest descent
        g = lambda a: f(x0 - a * Df(x0))
        a = opt.minimize_scalar(g).x
        #Approximates minimizers
        x0 = x0 - a * Df(x0)
        #Tests for convergence
        if np.max(np.abs(Df(x0))) < tol:
            return x0, True, i + 1
        
    return x0, False, maxiter


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Intializes
    conv = False
    n = Q.shape[0]
    r0 = Q @ x0 - b
    d = -r0
    k = 0
    while la.norm(r0) >= tol and k < n:
        a = (r0 @ r0) / (d @ Q @ d)
        #Caclulates the approximation for minimizer
        x0 = x0 + a * d
        #Steepest descent directions
        r = r0 + a * Q @ d
        beta = (r @ r) / (r0 @ r0)
        d = -r + beta * d
        k += 1
        r0 = r
    #Tests for convergence
    if la.norm(r0) >= tol:
        conv = True   
    return x0, conv, k


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Initialize variables
    conv = False
    r0 = -df(x0)
    d = r0
    g = lambda a: f(x0 + a * d)
    a = opt.minimize_scalar(g).x
    x = x0 + a * d
    k = 1
    
    while la.norm(r0) >= tol and k < maxiter:
        #Calculates the gradient
        r = -df(x)
        beta = (r@r) / (r0@r0)
        d = r + beta * d
        #Modified scalar for line search
        g = lambda a: f(x + a * d)
        a = opt.minimize_scalar(g).x
        #Calculates minimizer
        x = x + a * d
        k += 1
        r0 = r
    #Test for convergence
    if k < maxiter:
        conv = True
    return x, conv, k



# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    data = np.loadtxt(filename)
    #Stores B
    b = np.copy(data[:,0])
    #Edits Data to make the matrix A
    data[:,0] = 1
    A = data
    #Equation in least square form
    Q = A.T @ A
    b = A.T @ b
    return conjugate_gradient(Q, b, x0)[0]


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        #Minimize beta
        likely = lambda b:np.sum(np.log(1 + np.exp(-b[0] - b[1] * x)) + (1 - y) * (b[0] + b[1] * x))
        self.b0, self.b1 = opt.fmin_cg(likely, guess, full_output=False)

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        #Calculate the probability of an unlabeled predictor variable having an outcome of 1
        return 1/(1+np.exp(-(self.b0 + self.b1 * x)))


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    #Creates object and loads data
    model = LogisticRegression1D()
    data = np.load(filename)
    #Logistical Regression from the class
    model.fit(data[:,0], data[:,1], guess)

    #Plots data
    domain = np.linspace(30, 100, 1000)
    plt.scatter(31, model.predict(31), c='green', label="P(Damage at Launch)")
    plt.plot(domain, model.predict(domain), c='orange')
    plt.scatter(data[:,0], data[:,1], label='Previous Damage')
    plt.title('Probability of O-Ring Damage')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Probability there was O-ring damage
    return model.predict(31)

