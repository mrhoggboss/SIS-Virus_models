import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Parameters
N = 20  # Number of nodes
h = 0.1  # Discretization step size
W = 2  # Upperbound on network edge weights
ZERO_BOUND = 1e-8  # The bound for which we consider the value zero
iterations = 1000
tolerance = 1e-8  # Convergence tolerance

def run_simulation(B, delta):
    # Initial infection levels - completely random
    x = np.random.uniform(0, 1, N)

    # Store infection levels for plotting and convergence measurement
    x_history = [x.copy()]
    convergence_rate = []

    # Simulation loop
    convergence_flag = True
    for t in range(iterations):
        x_next = x + h * ((np.eye(N) - np.diag(x)) @ B - np.diag(delta)) @ x
        x_next = np.clip(x_next, 0, 1)  # Ensure infection levels are between 0 and 1
        
        if convergence_flag:
            # Measure the difference (residual) between consecutive states
            residual = np.linalg.norm(x_next - x)
            convergence_rate.append(residual)
        
        # Check if the residual is below the tolerance level
        if residual < tolerance:
            convergence_flag = False
        
        x = x_next
        x_history.append(x.copy())

    # Convert history to arrays for easier plotting
    x_history = np.array(x_history)

    # Validate Theorem 1 and Proposition 2
    spectral_radius = np.max(np.abs(np.linalg.eigvals(np.eye(N) + h * (B - np.diag(delta)))))

    print('Spectral radius is '+str(spectral_radius))
    print('Equilibrium is '+str(x_history[-1]))

    if spectral_radius <= 1:
        print("Theorem 1: The healthy state is asymptotically stable.")
    else:
        print("Proposition 2: The system has two equilibria, including a non-zero endemic state.")
    
    return spectral_radius, x_history, convergence_rate

def plot_simulation(x_history, convergence_rate):
    # Plot the infection levels over time
    plt.figure(figsize=(12, 6))
    for i in range(N):
        plt.plot(x_history[:, i], label=f'Node {i+1}')

    plt.xlabel('Time step')
    plt.ylabel('Infection level')
    plt.title('Infection levels over time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Plot the convergence rate
    plt.figure(figsize=(12, 6))
    plt.plot(convergence_rate)
    plt.xlabel('Time step')
    plt.ylabel('Residual (L2 norm)')
    plt.title('Convergence rate over time')
    plt.yscale('log')
    plt.show()

# ------------------------------------------------------------------------------------------------------------
# Experiment 1: completely random data
def random_exp():
    A = np.random.uniform(0, W, (N, N))  # Each weight greater than 0. Note that the network must be connected

    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    beta = []
    for i in range(N):
        # Must follow assumption 3
        beta.append(np.random.uniform(0, 10 / (np.sum(A[i]))))
    beta = np.array(beta)
    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')

    delta = np.random.uniform(0, 10, N)

    # Just to double check
    print('Delta is '+str(delta))
    print('Beta is '+ str(beta))

    return B, delta

# ------------------------------------------------------------------------------------------------------------
# Experiment 2: same healing rate beta but different deltas
def same_beta():
    A = np.random.uniform(0, W, (N, N))  # Each weight greater than 0. Note that this must be connected

    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    lower_bound = float('inf')
    for i in range(N):
        # Healing rate beta must follow assumption 3
        lower_bound = min(10 / (np.sum(A[i])), lower_bound)
    beta = np.random.uniform(0, lower_bound)
    beta = list(np.full((1, N), beta)[0])
    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')

    # N random deltas
    delta = np.random.uniform(0, 10, N)

    # Just to double check
    print('Delta is '+str(delta))
    print('Beta is '+ str(beta))

    return B, delta

# ------------------------------------------------------------------------------------------------------------
# Experiment 3: same infection rate delta but different healing rate betas
def same_delta():
    A = np.random.uniform(0, W, (N, N))  # Each weight greater than 0. Note that this must be connected

    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    delta = np.random.uniform(0, 10)
    delta = list(np.full((1, N), delta)[0])
    
    # N random betas
    lower_bound = float('inf')
    for i in range(N):
        # Healing rate beta must follow assumption 3
        lower_bound = min(10 / (np.sum(A[i])), lower_bound)
    beta = np.random.uniform(0, lower_bound, N)
    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')
    
    # Just to double check
    print('Delta is '+str(delta))
    print('Beta is '+ str(beta))

    return B, delta

# ------------------------------------------------------------------------------------------------------------
def run_and_save(num_of_exp: int, generation_func, output_file):
    spectral_radii, equilibria, validation, anomalies, convergence_rates = list(), list(), list(), list(), list()
    for _ in range(num_of_exp):
        B, delta = generation_func()
        spectral_radius, x_history, convergence_rate = run_simulation(B, delta)
        equilibrium = x_history[-1]
        spectral_radii.append(spectral_radius)
        equilibria.append(equilibrium)
        convergence_rates.append(convergence_rate)
        if spectral_radius <= 1:
            flag = True
            for x in equilibrium:
                if abs(x) > ZERO_BOUND:
                    flag = False
            validation.append(flag)
        else:
            flag = False
            for x in equilibrium:
                if abs(x) > 1e-8:
                    flag = True
            validation.append(flag)
        if not flag:
            anomalies.append(tuple((spectral_radius, x_history)))
    df = pd.DataFrame({
        'spectral_radius': spectral_radii,
        'equilibrium': equilibria,
        'validation': validation,
        'convergence_rate': [len(cr) for cr in convergence_rates]  # Store the length of the convergence rate (time to converge)
    })

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_file, index=False)
    print(f'Simulation results saved to {output_file}')

    return anomalies

# # Run and save simulations
# num_of_exp = 1000
# print(run_and_save(num_of_exp, random_exp, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/roc_random_param_1000.csv'))
# print(run_and_save(num_of_exp, same_beta, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/roc_same_beta_1000.csv'))
# print(run_and_save(num_of_exp, same_delta, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/roc_same_delta_1000.csv'))

# ------------------------------------------------------------------------------------------------------------
# Experiment 4: Assuming DFE, strong virus and weak healing abilities (so that spectral radius is close to 1)
def strong_virus():
    A = np.random.uniform(0, W, (N, N))  # Each weight greater than 0. Note that the network must be connected

    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    beta = []
    for i in range(N):
        # Must follow assumption 3
        beta.append(np.random.uniform(3 / (np.sum(A[i])), 6 / (np.sum(A[i]))))
    beta = np.array(beta)
    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')

    delta = np.random.uniform(4, 5, N)

    # Just to double check
    print('Delta is '+str(delta))
    print('Beta is '+ str(beta))

    return B, delta

# ------------------------------------------------------------------------------------------------------------
# Experiment 5: Assuming DFE, weak virus and strong healing abilities (so that spectral radius is close to 0)
def weak_virus():
    A = np.random.uniform(0, W, (N, N))  # Each weight greater than 0. Note that the network must be connected

    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    beta = []
    for i in range(N):
        # Must follow assumption 3
        beta.append(np.random.uniform(0, 2 / (np.sum(A[i]))))
    beta = np.array(beta)
    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')

    delta = np.random.uniform(8, 10, N)

    # Just to double check
    print('Delta is '+str(delta))
    print('Beta is '+ str(beta))

    return B, delta

# ------------------------------------------------------------------------------------------------------------
# Experiment 6: Assuming DFE, random beta and delta
def DFE_random():
    A = np.random.uniform(0, W, (N, N))  # Each weight greater than 0. Note that the network must be connected

    bound = np.random.randint(1, 10)
    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    beta = []
    for i in range(N):
        # Must follow assumption 3
        beta.append(np.random.uniform((bound-1) / (np.sum(A[i])), bound / (np.sum(A[i]))))
    beta = np.array(beta)
    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')

    delta = np.random.uniform(bound-1, 10, N)

    # Just to double check
    print('Delta is '+str(delta))
    print('Beta is '+ str(beta))

    return B, delta

num_of_exp = 1000
print(run_and_save(num_of_exp, DFE_random, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/roc_DFErandom_1000.csv'))
