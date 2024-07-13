import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy

# SOME NOTES:
# beta is infection rate - wrt each pair of nodes
# delta is healing rate - wrt each node

# TODO
# - Look at the generation functions. How to effectively converge towards healing and infections rates which satisfy the conditions? 

# Questions:
# - What to do with diagonal entries in A?
# - manually restrict x up to 1 at every iteration?

# seed
# np.random.seed(42)

# Parameters
N = 20  # Number of nodes
h = 0.1  # Discretization step size
W = 2 # upperbound on network edge weights
ZERO_BOUND = 1e-8 # the bound for which we consider the value zero
iterations = 1000

def run_simulation(B, delta):
    '''
    both B and delta are lists of two np.ndarrays, each representing the corresponding beta and delta values
    '''

    # Initial infection levels - completely random
    # assumption 1 - between 0 and 1
    x = np.random.uniform(0, 1, (2, N))

    # Store infection levels for plotting
    x1_history = [x[0].copy()]
    x2_history = [x[1].copy()]

    # Simulation loop
    for _ in range(iterations):
        sum_of_x = copy.deepcopy(np.diag(x[0]) + np.diag(x[1]))
        x[0] = x[0] + h * ((np.eye(N) - (sum_of_x)) @ B[0] - np.diag(delta[0])) @ x[0]
        x[1] = x[1] + h * ((np.eye(N) - (sum_of_x)) @ B[1] - np.diag(delta[1])) @ x[1]
        x = np.clip(x, 0, 1)  # Ensure infection levels are between 0 and 1
        x1_history.append(x[0].copy())
        x2_history.append(x[1].copy())

    # # Convert history to arrays for easier plotting
    # x1_history = np.array(x1_history)
    # x2_history = np.array(x2_history)
    
    # Validate Theorem 1 and Proposition 2
    # Check for stability conditions
    spectral_radius_1 = np.max(np.abs(np.linalg.eigvals(np.eye(N) + h * (B[0] - np.diag(delta[0])))))
    spectral_radius_2 = np.max(np.abs(np.linalg.eigvals(np.eye(N) + h * (B[1] - np.diag(delta[1])))))

    print('spectral radius 1 is '+str(spectral_radius_1))
    print('spectral radius 2 is '+str(spectral_radius_2))
    print('x1 equilibrium is '+str(x1_history[-1]))
    print('x2 equilibrium is '+str(x2_history[-1]))

    if spectral_radius_1 < 1 and spectral_radius_2 < 1:
        label = 2
        print("Theorem 2: The healthy state is asymptotically stable.")
    elif spectral_radius_1 > 1 and spectral_radius_2 > 1:
        det_radius_1 = np.max(np.abs(np.linalg.eigvals(np.eye(N) - h * np.diag(delta[1]) + (np.eye(N) - np.diag(x1_history[-1])) @ B[1])))
        det_radius_2 = np.max(np.abs(np.linalg.eigvals(np.eye(N) - h * np.diag(delta[0]) + (np.eye(N) - np.diag(x2_history[-1])) @ B[0])))
        if det_radius_1 < 1 and det_radius_2 < 1:
            label = 5 
            # 1) both (x_1, 0) and (0, x_2) are stable
            # 2) also exists (x_1hat, x_2hat) which is unstable
        else:
            if det_radius_1 > 1 and det_radius_2 > 1:
                label = 4.1
                # 1) both (x_1, 0) and (0, x_2) are unstable
            elif det_radius_1 <= 1 and det_radius_2 > 1:
                label = 4.2
                # 1) (x_1, 0) stable 
                # 2) (0, x_2) unstable
            elif det_radius_1 > 1 and det_radius_2 <= 1:
                label = 4.3
                # 1) (x_1, 0) unstable 
                # 2) (0, x_2) stable
            else:
                label = 4.4
                # both (x_1, 0) and (0, x_2) are stable
    elif spectral_radius_1 > 1 and spectral_radius_2 <= 1:
        label = 3.1 # Theorem 3, with (x_1, 0) stable
        print("Theorem 3")
    elif spectral_radius_2 > 1 and spectral_radius_1 <= 1:
        label = 3.2 # Theorem 3, with (0, x_2) stable
        print("Theorem 3")
    
    return label, spectral_radius_1, spectral_radius_2, x1_history, x2_history

def plot_simulation(x1_history, x2_history):
    # Plot the results
    # plt.figure(figsize=(12, 6))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(N):
        ax1.plot(x1_history[i], label=f'x1_Node {i+1}')
        ax2.plot(x2_history[i], label=f'x2_Node {i+1}')
    print(len(x1_history))
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Infection level')
    # ax1.title('Infection levels of virus 1 over time')
    # ax1.legend(bbox_to_anchor=(1.05, 1))

    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Infection level')
    # ax2.title('Infection levels of virus 2 over time')
    # ax2.legend(bbox_to_anchor=(1.05, 1))

    plt.show()

# ------------------------------------------------------------------------------------------------------------
# Experiment 1: completely random data
def random_exp():

    #I NEED TO REDO THIS. FOLLOW ASSUMPTIONS 1-2, 4-6.

    A = np.random.uniform(0, W, (N, N))  # each weight greater than 0. Note that the network must be connected

    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    beta = []
    for i in range(N):
        # must follow assumption 3
        beta.append(np.random.uniform(0, 10 / (np.sum(A[i]))))
    beta = np.array(beta)
    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')

    delta = np.random.uniform(0, 10, N)

    # just to double check
    print('delta is '+str(delta))
    print('beta is '+ str(beta))

    # spectral_radius, x_history = run_simulation(B, delta)
    # plot_simulation(x_history)

    return B, delta

# B_1, delta_1 = random_exp()
# B_2, delta_2 = random_exp()
# B = [B_1, B_2]
# delta = [delta_1, delta_2]
# label, spectral_radius_1, spectral_radius_2, x1_history, x2_history = run_simulation(B, delta)
# # plot_simulation(x1_history, x2_history) 

# ------------------------------------------------------------------------------------------------------------
def run_and_save(num_of_exp: int, generation_func, output_file):
    labels, spectral_radii_1, equilibria_1, spectral_radii_2, equilibria_2, validation, anomalies = list(), list(), list(), list(), list(), list(), list()
    for _ in range(num_of_exp):
        B_1, delta_1 = generation_func()
        B_2, delta_2 = generation_func()
        B = [B_1, B_2]
        delta = [delta_1, delta_2]
        label, spectral_radius_1, spectral_radius_2, x1_history, x2_history = run_simulation(B, delta)
        labels.append(label)
        equilibrium_1 = x1_history[-1]
        equilibrium_2 = x2_history[-1]
        spectral_radii_1.append(spectral_radius_1)
        spectral_radii_2.append(spectral_radius_2)
        equilibria_1.append(equilibrium_1)
        equilibria_2.append(equilibrium_2)

        if label == 2:
            flag = True
            for x in equilibrium_1:
                if abs(x) > ZERO_BOUND:
                    flag = False
            for x in equilibrium_2:
                if abs(x) > ZERO_BOUND:
                    flag = False
            validation.append(flag)
        elif label == 5:
            validation.append(None)
        elif label == 4.1:
            validation.append(None)
        elif label == 4.2:
            validation.append(None)
        elif label == 4.3:
            validation.append(None)
        elif label == 4.4:
            validation.append(None)
        elif label == 3.1:
            # x_2 is supposed to die out and x_1 is supposed to survive
            flag_1 = True
            for x in equilibrium_1:
                flag_1 = flag_1 and (abs(x) > ZERO_BOUND) # flag_1 false if there exists a single zero
            flag_2 = True
            for x in equilibrium_2:
                if abs(x) > ZERO_BOUND:
                    flag = False # flag_2 false if there exists a non-zero element
            validation.append(flag_1 and flag_2)
        elif label == 3.2:
            # x_1 is supposed to die out and x_2 is supposed to survive
            flag_1 = True
            for x in equilibrium_1:
                if abs(x) > ZERO_BOUND:
                    flag = False # flag_1 false if there exists a non-zero element
            flag_2 = True
            for x in equilibrium_2:
                flag_2 = flag_2 and (abs(x) > ZERO_BOUND) # flag_2 false if there exists a single zero
            validation.append(flag_1 and flag_2)

    df = pd.DataFrame({
        'label': labels,
        'spectral_radius_1': spectral_radii_1,
        'spectral_radius_2': spectral_radii_2,
        'equilibrium_1': equilibria_1,
        'equilibrium_2': equilibria_2,
        'validation': validation
    })

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_file, index=False)
    print(f'Simulation results saved to {output_file}')

    return anomalies

num_of_exp = 1000
print(run_and_save(num_of_exp, random_exp, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/bivirus_random_param_1000.csv'))
# print(run_and_save(num_of_exp, same_beta, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/same_beta_1000.csv'))
# print(run_and_save(num_of_exp, same_delta, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/same_delta_1000.csv'))

# ------------------------------------------------------------------------------------------------------------
# Now, we would like to assess rate of convergence
# The idea here is that we find the residual (L2 norm of x and x_next) 
# at every step. We then plot them on an exponential scale,
# and find the best fit line to the residual plot. 
# We can then compare the gradient for the rate of convergence. 
# This is assuming the convergences are exponential, which they seem to be.
