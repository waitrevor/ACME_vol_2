# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name> Trevor Wai
<Class> Section 1
<Date> 3/22/23
"""

import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Objective
    x = cp.Variable(3, nonneg=True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    #Constraints
    G = np.array([[1, 2, 0], [0, 1, -4], [-2, -10, -3]])
    h = np.array([3, 1, -12])
    constraints = [G @ x <= h]

    #Problem
    problem = cp.Problem(objective, constraints)
    min_val = problem.solve()

    return x.value, min_val

# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Objective
    m, n = A.shape
    x = cp.Variable(n)
    c = cp.norm(x, 1)
    objective = cp.Minimize(c)
    #Constraints
    constraints = [A @ x == b]

    #Problem
    problem = cp.Problem(objective, constraints)
    min_val = problem.solve()

    return x.value, min_val



# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Objective
    x = cp.Variable(6, nonneg=True)
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c.T @ x)

    #Constraints
    A = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
    b = np.array([7, 2, 4, 5, 8])
    constraints = [A @ x == b]

    #Problem
    problem = cp.Problem(objective, constraints)
    min_val = problem.solve()

    return x.value, min_val

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Objective
    Q = np.array([[3,2,1], [2,4,2], [1,2,3]])
    r = np.array([3, 0, 1])
    x = cp.Variable(3)
    #Prolblem with constraints
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + r.T @ x))

    min_val = prob.solve()
    return x.value, min_val



# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Objective
    m, n = A.shape
    x = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(cp.norm(A @ x - b, 2))
    
    #Constraint
    constraint = [sum(x) == 1]

    #Problem
    problem = cp.Problem(objective, constraint)
    min_val = problem.solve()

    return x.value, min_val




# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Reads in Data
    data = np.load('food.npy', allow_pickle=True)

    #Objective
    m,n = data.shape
    x = cp.Variable(m, nonneg=True)
    objective = cp.Minimize(data[:,0] @ x)

    #Constraint
    s = data[:,1]
    c, f, s_hat, c_hat, f_hat, p_hat = np.array([data[:,i] * s for i in range(2, n)])
    G = np.array([c, f, s_hat])
    P = np.array([c_hat, f_hat, p_hat])
    h = np.array([2000, 65, 50])
    q = np.array([1000, 25, 46])
    constraints = [G @ x <= h, P @ x >= q]

    #Problem
    problem = cp.Problem(objective, constraints)

    min_value = problem.solve()
    return x.value, min_value
