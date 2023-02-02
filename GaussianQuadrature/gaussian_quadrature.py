# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
<Name> Trevor Wai
<Class> Section 1
<Date> 1/30/23
"""
from scipy import linalg as la
from scipy.stats import norm
from scipy.integrate import quad
import numpy as np
from matplotlib import pyplot as plt

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        #Raises ValueError if polytime is not acceptable
        if polytype not in ['legendre', 'chebyshev']:
            raise ValueError('Polytype is not legendre or chebyshev')
        
        #Stores objects
        self.n = n
        self.polytype = polytype
        
        #Stores w inverse
        if self.polytype == 'legendre':
            self.w_inv = lambda x: 1
        elif self.polytype == 'chebyshev':
            self.w_inv = lambda x: np.sqrt(1 - x**2)
        
        #Stores the points and weights from problem 3
        self.points, self.weights = self.points_weights(n)

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        #Creates Alpha
        alpha = np.zeros(n)

        #Creates Beta if polytype is legendre
        if self.polytype == 'legendre':
            beta = np.array([k**2 / (4*k**2 - 1) for k in range(1,n)])

        #Creates Beta if polytype is chebyshev
        elif self.polytype == 'chebyshev':
            for k in range(1, n):
                if k == 1:
                    beta = np.array([1/2])
                else:
                    beta = np.append(beta, 1/4)
        beta = np.sqrt(beta)

        #Creates the jacobian J
        J = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)

        #Finds the corresponding eigen values and eigen vectors
        eigen_vals, eigen_vect = la.eigh(J)

        #Finds the weights
        weights = eigen_vect[0]**2
        if self.polytype == 'legendre':
            weights *= 2
        else:
            weights *= np.pi

        return np.real(eigen_vals), weights           


    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        #Appriximates the integral
        return np.dot(self.weights, np.multiply(f(self.points), self.w_inv(self.points)))

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        #Approximate the integral of a function on the interval [a,b]
        h = lambda x: f(((b-a) / 2 * x) + ((a + b) / 2))
        return ((b - a) / 2) * self.basic(h)
    
    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        #Change the bounds of integration
        h = lambda x, y: f((b1 - a1) / 2 * x + ((a1 + b1) / 2), (b2 - a2) / 2 * y + ((a2 + b2) / 2))
        #Divide by w(x)w(y)
        g = lambda x, y: h(x, y) * self.w_inv(x) * self.w_inv(y)
        #Calculate double sum
        fin = np.sum([self.weights[i] * self.weights[j] * g(self.points[i], self.points[j]) for j in range(self.n) for i in range(self.n)])
        return ((b1 - a1) * (b2 - a2) / 4 ) * fin


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn't depend on n).
    """
    
    exact = norm.cdf(2) - norm.cdf(-3) # Integrate f from -3 to 2.
    leg_err = []
    cheb_err = []
    #Plot Horizontal Line
    domain = np.linspace(5,50,10)
    zero = np.ones(len(domain)) * abs(exact - quad(norm.pdf, -3, 2)[0])
    #legendre and cheby error calculator
    for n in range(5, 51, 5):
        #Legendre
        gaus = GaussianQuadrature(n)
        leg = gaus.integrate(norm.pdf, -3, 2)
        leg_err.append(abs(exact - leg))
        #Chebyshev
        gaus_cheb = GaussianQuadrature(n, 'chebyshev')
        cheb = gaus_cheb.integrate(norm.pdf, -3, 2)
        cheb_err.append(abs(exact - cheb))

    
    plt.semilogy(domain, leg_err, label='legendre error')
    plt.semilogy(domain, cheb_err, label='chebyshev error')
    plt.semilogy(domain, zero, label='exact')
    plt.xlabel('n')
    plt.ylabel('error')
    plt.legend()
    plt.tight_layout()
    plt.show()