# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Name> Trevor Wai
<Class> Section 2
<Date> 9/8/22
"""

import numpy as np

def prob1():
    """ Define the matrices A and B as arrays. Return the matrix product AB. """
    #Define matrix A
    A = np.array([[3, -1, 4], [1, 5, -9]])

    #Define matrix B
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])

    #Computs product of A and B
    ans = A @ B

    #Returns answer
    return ans


def prob2():
    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
    #Define matrix A
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])

    #Compute the answer
    ans = -(A @ A @ A) + 9 * (A @ A) - 15 * (A)

    #Returns the answer
    return ans


def prob3():
    """ Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    #Creates a 7x7 matrix full of ones
    A = np.ones((7,7))

    #Finds the upper triangle of matrix A
    A = np.triu(A)

    #Creates a 7x7 matrix full of -1 and defines B as the lower triangle of that matrix
    B = np.tril(np.full((7,7), -1))

    #Creates a 7x7 matrix full of 5 and defines C as the upper triangle but shifted up one diagonal
    C = np.triu(np.full((7,7), 5), 1)

    #Adds matrices C and B
    B = B + C

    #Returns the matrix product
    return (A @ B @ A).astype(np.int64)


def prob4(A):
    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    #Creates a copy of matrix A
    B = np.copy(A)

    #Finds all the elements of B less than zero
    mask = B < 0

    #Changes all the elements less than zero to 0
    B[mask] = 0

    #Returns the new matrix
    return B


def prob5():
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    #Defines matrix A
    A = np.array([[0, 2, 4], [1, 3, 5]])

    #Defines Matrix B
    B = np.tril(np.full((3,3), 3))

    #Defines Matrix C
    C = np.diag([-2, -2, -2])

    #Defines Matrix I
    I = np.eye(3,3)

    #Creates row 1
    row1 = np.hstack((np.zeros((3, 3)), np.transpose(A), I))

    #Creates row 2
    row2 = np.hstack((A, np.zeros((2,2)), np.zeros((2, 3))))

    #Creates row 3
    row3 = np.hstack((B, np.zeros((3,2)), C))

    #Stacks row 1, row 2, and row 3 creating the block matrix
    block = np.vstack((row1, row2, row3))

    #Returns the block matrix
    return block


def prob6(A):
    """ Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    #Finds a matrix with each row as the sum of the rows of A
    B = np.reshape(A.sum(axis=1), (-1,1))

    #Divids A by B
    C = A / B

    #Returns the answer
    return C


def prob7():
    """ Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """
    #Imports the given array stored in grid.npy
    grid = np.load("grid.npy")

    #Finds the max product of the rows
    ho = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:])
    #Finds the max product of the columns
    vert = np.max(grid[:-3,:] * grid[1:-2,:] * grid[2:-1,:] * grid[3:,:])
    #Finds the max product of diagonal going from top left to bottom right
    diag1 = np.max(grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1,2:-1] * grid[3:,3:])
    #Finds the max product of diagonal going from top right to bottom left
    diag2 = np.max(grid[:-3,3:] * grid[1:-2,2:-1] * grid[2:-1,1:-2] * grid[3:,:-3])
    #Finds the max of the four products
    Max_ = max(ho, vert, diag1, diag2)
    #returns the answer
    return Max_


#testing

print(prob3())