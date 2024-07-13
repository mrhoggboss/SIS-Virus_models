import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# SOME NOTES:
# beta is infection rate - wrt each pair of nodes
# delta is healing rate - wrt each node

# TODO
# - Look at the generation functions. How to effectively converge towards healing and infections rates which satisfy the conditions? 

# Questions:
# - What to do with diagonal entries in A?
# - manually restrict x up to 1 at every iteration?

# seed
np.random.seed(342)

# Parameters
N = 20  # Number of nodes
h = 0.1  # Discretization step size
W = 2 # upperbound on network edge weights
ZERO_BOUND = 1e-8 # the bound for which we consider the value zero
iterations = 100

def run_simulation(x0, B, delta):
    # Initial infection levels - completely random
    # assumption 1 - between 0 and 1
    x = x0

    # Store infection levels for plotting
    x_avg = np.average(x)
    x_history = [x_avg]

    # Simulation loop
    for _ in range(iterations):
        x = x + h * ((np.eye(N) - np.diag(x)) @ B - np.diag(delta)) @ x
        x = np.clip(x, 0, 1)  # Ensure infection levels are between 0 and 1
        x_avg = np.average(x)
        x_history.append(x_avg)

    # Convert history to arrays for easier plotting
    x_history = np.array(x_history)

    # Validate Theorem 1 and Proposition 2
    # Check for stability conditions
    spectral_radius = np.max(np.abs(np.linalg.eigvals(np.eye(N) + h * (B - np.diag(delta)))))

    print('spectral radius is '+str(spectral_radius))
    print('equilibrium is '+str(x_history[-1]))

    if spectral_radius <= 1:
        print("Theorem 1: The healthy state is asymptotically stable.")
    else:
        print("Proposition 2: The system has two equilibria, including a non-zero endemic state.")
    
    return spectral_radius, x_history

def plot_simulation(x_histories):
    # Plot the results
    fig, axs = plt.subplots(nrows=3, ncols=3)
    # for i in range(N):
    #     plt.plot(x_history[:, i], label=f'Node {i+1}')
    for row in axs:
        for col in row:
            col.plot(x_histories.pop(0))
            col.text(iterations, x_history[-1], f"({round(x_history[-1], 4)})", fontsize=8)
            ax = plt.gca()
            ax.set_ylim([0, 1])
            plt.xlabel('Time step')
            plt.ylabel('Average Infection level across 20 Pop. nodes')
            
    # for x_history in x_histories:
    #     ax = fig.add_subplot(111)
    #     ax.plot(x_history)
    #     ax.text(iterations, x_history[-1], f"({round(x_history[-1], 4)})", fontsize=8)
    #     plt.xlabel('Time step')
    #     plt.ylabel('Average Infection level across 20 Pop. nodes')
    #     plt.title('Average Infection levels over time')

    for ax in axs.flat:
        ax.set(xlabel='Time step', ylabel='Avg. Infection level')
        ax.label_outer()
        ax.set_ylim([0, 1])
    
    # for ax in axs.flat:
        
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle('Average Infection levels across 20 pop. nodes VS Time')
    plt.show()

# ------------------------------------------------------------------------------------------------------------
# Experiment 1: completely random data
def random_exp():
    A = np.random.uniform(0, W, (N, N))  # each weight greater than 0. Note that the network must be connected

    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    beta = []
    for i in range(N):
        # must follow assumption 3
        beta.append(np.random.uniform(0, 10 / (np.sum(A[i]))))
    beta = np.array(beta)

    # beta *= 0.5

    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')

    delta = np.random.uniform(0, 10, N)

    # delta *= 1.5
    
    # just to double check
    print('delta is '+str(delta))
    print('beta is '+ str(beta))

    spectral_radius = np.max(np.abs(np.linalg.eigvals(np.eye(N) + h * (B - np.diag(delta)))))
    if spectral_radius <= 1:
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/rand_thm1/A.csv", A, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/rand_thm1/beta.csv", beta, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/rand_thm1/delta.csv", delta, delimiter=",")
    else:
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/rand_prop2/A.csv", A, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/rand_prop2/beta.csv", beta, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/rand_prop2/delta.csv", delta, delimiter=",")

    return B, delta

x_histories = []
B, delta = random_exp()
x = np.random.uniform(0, 1, N)
for num in range(1, 10):
    x.fill(num/10)
    print('x is '+str(x))
    spectral_radius, x_history = run_simulation(x, B, delta)
    x_histories.append(x_history)

plot_simulation(x_histories)

# ------------------------------------------------------------------------------------------------------------
# Experiment 2: same healing rate beta but different deltas
def same_beta():
    A = np.random.uniform(0, W, (N, N))  # each weight greater than 0. Note that this must be connected

    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    lower_bound = float('inf')
    for i in range(N):
        # healing rate beta must follow assumption 3
        lower_bound = min(10 / (np.sum(A[i])), lower_bound)
    beta = np.random.uniform(0, lower_bound)

    # for making spectral radius smaller
    # beta *= 0.7

    beta = list(np.full((1, N), beta)[0])
    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')
    
    # N random deltas
    delta = np.random.uniform(0, 10, N)

    # just to double check
    print('delta is '+str(delta))
    print('beta is '+ str(beta))
    print('A is ' + str(A))
    spectral_radius = np.max(np.abs(np.linalg.eigvals(np.eye(N) + h * (B - np.diag(delta)))))
    if spectral_radius <= 1:
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_beta_thm1/A.csv", A, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_beta_thm1/beta.csv", beta, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_beta_thm1/delta.csv", delta, delimiter=",")
    else:
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_beta_prop2/A.csv", A, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_beta_prop2/beta.csv", beta, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_beta_prop2/delta.csv", delta, delimiter=",")

    return B, delta

# x_histories = []
# B, delta = same_beta()
# x = np.random.uniform(0, 1, N)
# for num in range(1, 10):
#     x.fill(num/10)
#     print('x is '+str(x))
#     spectral_radius, x_history = run_simulation(x, B, delta)
#     x_histories.append(x_history)

# plot_simulation(x_histories)

# ------------------------------------------------------------------------------------------------------------
# Experiment 3: same infection rate delta but different healing rate betas
def same_delta():
    A = np.random.uniform(0, W, (N, N))  # each weight greater than 0. Note that this must be connected

    # Assumption 3 - each delta is bounded by 10 and the sum of any row of beta_ij cannot exceed 10
    delta = np.random.uniform(0, 10)
    delta = list(np.full((1, N), delta)[0])
    
    # N random betas
    lower_bound = float('inf')
    for i in range(N):
        # healing rate beta must follow assumption 3
        lower_bound = min(10 / (np.sum(A[i])), lower_bound)
    beta = np.random.uniform(0, lower_bound, N)
    B = np.diag(beta) @ A
    if np.count_nonzero(B) <= N:
        print('Assumption 4 could be violated! Too many zeroes in B!')
    
    # just to double check
    print('delta is '+str(delta))
    print('beta is '+ str(beta))

    spectral_radius = np.max(np.abs(np.linalg.eigvals(np.eye(N) + h * (B - np.diag(delta)))))
    if spectral_radius <= 1:
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_delta_thm1/A.csv", A, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_delta_thm1/beta.csv", beta, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_delta_thm1/delta.csv", delta, delimiter=",")
    else:
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_delta_prop2/A.csv", A, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_delta_prop2/beta.csv", beta, delimiter=",")
        np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Single-system-analysis/same_delta_prop2/delta.csv", delta, delimiter=",")

    return B, delta

# x_histories = []
# B, delta = same_beta()
# x = np.random.uniform(0, 1, N)
# for num in range(1, 10):
#     x.fill(num/10)
#     print('x is '+str(x))
#     spectral_radius, x_history = run_simulation(x, B, delta)
#     x_histories.append(x_history)

# plot_simulation(x_histories)
# ------------------------------------------------------------------------------------------------------------
# random_exp()
# same_beta()
# same_delta()

# ------------------------------------------------------------------------------------------------------------
def run_and_save(num_of_exp: int, generation_func, output_file):
    spectral_radii, equilibria, validation, anomalies = list(), list(), list(), list()
    for _ in range(num_of_exp):
        B, delta = generation_func()
        spectral_radius, x_history = run_simulation(B, delta)
        equilibrium = x_history[-1]
        spectral_radii.append(spectral_radius)
        equilibria.append(equilibrium)
        if spectral_radius <= 1:
            flag = True
            for x in equilibrium:
                if abs(x) > ZERO_BOUND:
                    flag = False # false if there exists a single non-zero component
            validation.append(flag)
        else:
            flag = True
            for x in equilibrium:
                flag = flag and (abs(x) > ZERO_BOUND)
            validation.append(flag)
        if not flag:
            anomalies.append(tuple((spectral_radius, x_history)))
    df = pd.DataFrame({
        'spectral_radius': spectral_radii,
        'equilibrium': equilibria,
        'validation': validation
    })

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_file, index=False)
    print(f'Simulation results saved to {output_file}')

    return anomalies

num_of_exp = 1000
# anomalies = run_and_save(num_of_exp, random_exp, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/random_param_1000.csv')
# anomalies = run_and_save(num_of_exp, same_beta, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/same_beta_1000.csv')
# anomalies = run_and_save(num_of_exp, same_delta, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/same_delta_1000.csv')
# print(len(anomalies))
# sr = []
# fs = []
# for anomaly in anomalies:
#     spectral_radius = anomaly[0]
#     final_state = anomaly[1][-1]
#     sr.append(spectral_radius)
    
#     sum = 0
#     for num in final_state:
#         sum += num**2
#     fs.append(sum)

# # Plot the anomalies
# plt.figure(figsize=(12, 6))
# for i in range(N):
#     plt.scatter(sr, fs)
# plt.xlabel('Spectral Radius')
# plt.ylabel('L2 Norm of final state')
# plt.title('Plot of Spectral radii VS L2 Norm of final state for anomalies')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()
# ------------------------------------------------------------------------------------------------------------
# Now, we would like to assess rate of convergence
# The idea here is that we find the residual (L2 norm of x and x_next) 
# at every step. We then plot them on an exponential scale,
# and find the best fit line to the residual plot. 
# We can then compare the gradient for the rate of convergence. 
# This is assuming the convergences are exponential, which they seem to be.