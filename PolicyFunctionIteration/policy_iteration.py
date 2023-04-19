# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Name> Trevor Wai
<Class> Section 1
<Date> 4/18/23
"""

import numpy as np
import gym
from gym import wrappers

# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]



# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    #Initialize
    v = np.zeros(nS)
    n = 0
    #Loops
    while n < maxiter:
        v_0 = np.copy(v)
        #Compute v
        for s in range(nS):
            v[s] = max([sum([p*(r + beta*v_0[s1]) for p, s1, r, _ in P[s][a]]) for a in range(nA)])
        #Break Case
        if np.linalg.norm(v- v_0) < tol:
            break
        n += 1

    return v, n+1

# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    #Initialize
    policy = np.zeros(nS, dtype=int)
    #Compute Policy
    for s in range(nS):
        policy[s] = np.argmax([sum([p*(r + beta * v[s1]) for p, s1, r, _ in P[s][a]]) for a in range(nA)])
    return policy

# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    #Initialize
    v = np.zeros(nS)
    #Loops
    while True:
        v_0 = np.copy(v)
        #Compute v
        for s in range(nS):
            v[s] = sum([p*(r + beta*v_0[s1]) for p, s1, r, _ in P[s][policy[s]]])
        #Break Case
        if np.linalg.norm(v - v_0) < tol:
            break
    return v

# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    #Initialize
    policy = np.zeros(nS, dtype=int)
    v = np.zeros(nS)
    n = 0
    #Loops
    while n < maxiter:
        v = compute_policy_v(P, nS, nA, policy, beta, tol)
        policy_0 = np.copy(policy)
        #Compute Policy
        for s in range(nS):
            policy[s] = np.argmax([sum([p*(r + beta*v[s1]) for p, s1, r, _ in P[s][a]]) for a in range(nA)])
        #Break Case
        if np.array_equal(policy, policy_0):
            break
        n += 1
    return v, policy, n

# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    if basic_case:
        env = gym.make('FroxenLake-v1')
        nS = 16
        nA = 4
    else:
        env = gym.make('FrozenLake8x8-v1')
        nS = 64
        nA = 4
    P = env.unwrapped.P
    vi_value_func, _ = value_iteration(P, nS, nA)
    vi_policy =extract_policy(P, nS, nA, vi_value_func)
    vi_total_rewards = run_simulation(env, vi_policy, render)

    pi_value_func, p_policy = policy_iteration(P, nS, nA)
    pi_total_rewards, pi_policy, _ = policy_iteration(P, nS, nA)
    
    return vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards
    

# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    M = 1000
    total_reward = 0
    for _ in range(M):
        s = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            s, r, done, _ = env.step(policy[s])
            total_reward += r * beta
            beta *= beta
    return total_reward / M
