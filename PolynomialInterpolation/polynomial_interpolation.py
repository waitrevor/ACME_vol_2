# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Name> Trevor Wai
<Class> Section 2
<Date> 1/18/23
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy.interpolate import BarycentricInterpolator
from numpy.fft import fft

# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    n = len(xint)
    m = len(points)
    #Subroutine that will evaluate each of the n Lagrange Basis functions at every point in domain
    dem = np.array([np.prod(np.delete([xint[j] - xint[k] for k in range(n)], j)) for j in range(n)])
    num = np.array([np.prod(np.delete(points - xint.reshape((n,1)),j,0),axis=0) for j in range(n)])
    L = num / dem.reshape((n,1))
    #Final array
    p = np.sum(L * yint.reshape(n,1), axis=0)
    return p


# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        #Initialzes Weights and x and y interpolating points
        self.xint = xint
        self.yint = yint
        #Calculates the Barycentric weights
        weights = np.array([])
        for j in xint:
            w = 1
            for k in xint:
                if j != k:
                    w *= j - k
            weights = np.append(weights, 1/w)
        self.weights = weights

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        val = 0
        num = 0
        dem = 0

        #Calculates the interpolating polynomial at points
        for i in range(len(self.xint)):
            num += ((self.weights[i] / (points - self.xint[i] + 0.000000001)) * self.yint[i]) 
            dem += (self.weights[i] / (points - self.xint[i] + 0.000000001))

        val = num/dem

        return val

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        #Updates existing Barycentric weights
        for x in xint:
            weights = np.array([np.prod(1 / (x - self.xint + 0.000000001))])
            self.weights = self.weights / (x - self.xint + 0.000000001)
        #Saves new weights, x points and y points
        self.xint = np.append(self.xint, xint)
        self.yint = np.append(self.yint, yint)
        self.weights = np.append(self.weights, weights)


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    domain = np.linspace(-1, 1, 400)
    f = lambda x: 1/(1+25 * x**2) 
    n = [2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8]
    L = []
    C_L = []
    for i in n:
        #Interpolates over evenly spaced points
        pts = np.linspace(-1, 1, i)
        poly = BarycentricInterpolator(pts)
        poly.set_yi(f(pts))
        L.append(la.norm(f(domain) - poly(domain), ord=np.inf))
        #Interpolates over the Chebyshev polynomials
        cheby = np.cos(np.arange(i+1) * np.pi / i)
        cheb_poly = BarycentricInterpolator(cheby)
        cheb_poly.set_yi(f(cheby))
        C_L.append(la.norm(f(domain) - cheb_poly(domain), ord=np.inf))

    plt.loglog(n, L, label='evenly spaced')
    plt.loglog(n, C_L, label='Chebysev')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    #Finds the Chebyshev polynomial
    y = np.cos((np.pi * np.arange(2*n)) / n)
    samples = f(y)
    #Interpolates
    coeffs = np.real(fft(samples))[:n+1] / n
    coeffs[0] = coeffs[0] / 2
    coeffs[n] = coeffs[n] / 2

    return coeffs


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    #Loads the information
    data = np.load('airdata.npy')

    #Interpolates the air quality data using Barycentril Lagrange Interpolation
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    poly = Barycentric(domain[temp2], data[temp2])

    #Plot of original Data
    plt.subplot(211)
    plt.plot(domain, data)
    #Plot of Interpolating polynomial
    plt.subplot(212)
    plt.plot(domain, poly(domain))
    plt.show()


