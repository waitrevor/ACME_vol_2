# markov_chains.py
"""Volume 2: Markov Chains.
<Name> Trevor Wai
<Class> Section 2
<Date> 11/2/22
"""

import numpy as np
from scipy import linalg as la



class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        if not np.allclose(A.sum(axis=0), np.ones(A.shape[1])):
            raise ValueError('A is not square or not column stochastic')

        self.A = A

        m,n = A.shape

        if states is None:
            self.states = [str(i) for i in range(n)]
        else:
            self.states = states

        self.dictionary = dict(zip(self.states, [i for i in range(n)]))

        

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        col = self.dictionary[state]
        trans = np.argmax(np.random.multinomial(1, self.A[:,col]))
        label = self.states[trans]
        return label 

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        L = []
        currState = start

        for i in range(N):
            L.append(currState)
            currState = self.transition(currState)
        
        return L

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        currState = start
        L = []
        while currState != stop:
            L.append(currState)
            currState = self.transition(currState)

        L.append(stop)
        return L
        

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        m,n = self.A.shape
        y = np.random.random(n)
        x = y / sum(y) 
        for itr in range(maxiter):
            x_0 = x
            x = self.A @ x
            if sum(abs(x_0 - x)) < tol:
                return x

        raise ValueError("There is no convergence within maxiter iterations")


class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
            L = [line.strip().split() for line in lines]
            word_set = set()
            for sentence in L:
                word_set.update(sentence)
            words_list = ['$tart'] + list(word_set) + ['$top']
            mat = np.zeros((len(words_list),len(words_list)))
            for quote in L:
                quote = ['$tart'] + quote + ['$top']
                for i in range(len(quote) - 1):
                    
                    mat[words_list.index(quote[i+1])][words_list.index(quote[i])] += 1

        mat[len(words_list) - 1][len(words_list) - 1] = 1

        mat = mat / np.sum(mat, axis=0)

        MarkovChain.__init__(self, mat, words_list)

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        return ' '.join(self.path('$tart', '$top')[1:-1])
