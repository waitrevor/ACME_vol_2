# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Name> Trevor Wai
<Class> Section 2
<Date> 4/13/23
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    Vt = 0

    for i in reversed(range(1, N+1)):
        #Calculate expected value
        Vt1 = max(Vt * (i-1) / i + 1/N, Vt)
        #Stops when Finds optimal value
        if Vt == Vt1:
            break
        Vt = Vt1

    return Vt1, i


# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    optimal = np.array([calc_stopping(3)])

    #Find the optimal stoping percentages and max probability
    for i in range(4, M+1):
        optimal = np.concatenate((optimal, np.array([calc_stopping(i)])), axis=0)

    domain = range(3, M+1)
    #Graph the optimal stopping percentage of candidates and max probability
    plt.plot(domain, optimal[:,0], label='Max Probability')
    plt.plot(domain, optimal[:,1] / np.arange(3, M+1), label='Optimal Stopping Percentage')

    plt.title('Optimal Stopping Percentages of Candidates and Max Probability')
    plt.xlabel('N')
    plt.ylabel('Percent')

    plt.legend()
    plt.tight_layout()
    plt.show()

    #Optimal Stopping Percentage
    return calc_stopping(M)[1]/M
    


# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    #Create the consumption matrix for the given parameters
    w = u(np.arange(N+1) / N)
    return sparse.diags(w, -np.arange(N+1), shape=(N+1, N+1)).toarray()


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    C = get_consumption(N,u)
    w = np.arange(N+1) / N
    mask = (C > 0) + np.eye(N+1)
    #Value Matrix
    A = np.zeros((N+1, T+1))
    A[:,-1] = u(w)
    #Policy Matrix
    P = np.zeros((N+1, T+1))
    P[:,-1] = w

    #Create Value and Policy Matrices
    for i in range(2, T+2):
        CV = (B * A[:,-i+1] + C) * mask
        A[:,-i] = np.max(CV, axis=1)
        P[:,-i] = w - w[np.argmax(CV, axis=1)]

    return A, P



# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    P = eat_cake(T, N, B, u)[1]
    T = []
    i = N

    #Find the most optimal path to take
    for j in range(N):
        T.append(P[i,j])
        i -= round(P[i,j] * N)

    return np.array(T)
