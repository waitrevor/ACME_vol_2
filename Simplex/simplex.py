"""Volume 2: Simplex

<Name> Trevor Wai
<Date> 3/8/23
<Class> Section 1
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        m,n = A.shape
        x = np.zeros(n)
        #Raises a value error if the problem is infeasible at the origin
        comparison = A @ x > b.T
        if comparison.any():
            raise ValueError('The problem is infeasible at the origin.')
        
        #Saves the dictionary
        self.dictionary = self._generatedictionary(c, A, b)

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        m,n = A.shape
        #Creates Abar
        Abar = np.concatenate((A, np.eye(m)), axis=1)
        #Creates cbar
        cbar = np.append(c, np.zeros(m))
        
        stacked = np.vstack((cbar, -Abar))
        #Creates the first column
        bbar = np.vstack(np.append(np.array([0]), b))

        return np.concatenate((bbar, stacked), axis=1)



    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        #Return the column index of the next pivot column
        mask = self.dictionary[0, 1:] < 0
        return np.argmax(mask) + 1
        

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        #Raises an error if the problem is unbounded
        if (self.dictionary[1:, index] >= 0).all():
            raise ValueError('The problem is unbounded and has no solution.')
        m,n = self.dictionary.shape
        #Gets the row index of the next pivot row
        ratio = [-self.dictionary[i, 0] / self.dictionary[i, index] if self.dictionary[i, index] < 0 else np.inf for i in range(1, m)]
        return np.argmin(ratio) + 1
        

    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        #Pivot element index
        col = self._pivot_col()
        row = self._pivot_row(col)
        m,n = self.dictionary.shape

        #Reduces the column to negative elementary vector
        self.dictionary[row,:] = self.dictionary[row,:] / -self.dictionary[row][col]
        for i in range(m):
            if i != row:
                self.dictionary[i] = self.dictionary[i] + (self.dictionary[row] * self.dictionary[i][col])

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        #Solves the linear optimization problem
        while (self.dictionary[0][1:] < 0).any():
            self.pivot()

        independent = {}
        dependent = {}
        m,n = self.dictionary.shape
        #Findes the independent afnd dependent variables and their values
        for i in range(1, n):
            #Dependent
            if self.dictionary[0][i] == 0:
                index = np.argmin(self.dictionary[:,i])
                dependent[i-1] = self.dictionary[index][0]
            #Independent
            else:
                independent[i-1] = 0
        return (self.dictionary[0][0], dependent, independent)

# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    data = np.load(filename)
    n = data['A'].shape[1]
    #Makes the b vector
    b = np.hstack((data['m'],data['d']))
    #Makes the A matrix
    A = np.vstack((data['A'], np.eye(n)))
    solver = SimplexSolver(-data['p'], A, b)
    
    dep = solver.solve()[1]

    #Gets the number of units that should be prodced for each product
    answer = []
    i = 0
    while len(answer) < n:
        if i in dep.keys(): answer.append(dep[i])
        i += 1
    
    
    return np.array(answer)
