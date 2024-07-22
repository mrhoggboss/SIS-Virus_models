import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy
import sys
import random
# SOME NOTES:
# beta is infection rate - wrt each pair of nodes
# delta is healing rate - wrt each node

# TODO
# - Look at the generation functions. How to effectively converge towards healing and infections rates which satisfy the conditions? 

# Questions:
# - What to do with diagonal entries in A?
# - manually restrict x up to 1 at every iteration?

# seed = random.randrange(1, 2**31)
seed = 1669509115
rng = np.random.seed(seed)
print("Seed was:", seed)
# seed
# np.random.seed(62)

# this seed is used for rand thm2 and rand thm3.
# for rand thm2, let delta_bound be between 7 and 10.
# for rand thm3, let delta_bound be between 4 and 7, and let beta_1_bound be between 0 and beta_bound / 20
# np.random.seed(444441212)

# this seed generates a case where sr1 slightly above 1 and sr2 is slightly below 1.
# np.random.seed(44444112)

# Parameters
N = 20  # Number of nodes
h = 0.1  # Discretization step size
W = 2 # upperbound on network edge weights
ZERO_BOUND = 1e-8 # the bound for which we consider the value zero
iterations = 10000

def run_simulation(x1, x2 , B, delta):
    '''
    both B and delta are lists of two np.ndarrays, each representing the corresponding beta and delta values
    '''

    # Initial infection levels - completely random
    # assumption 1 - between 0 and 1
    x1_history = [x1.copy()]
    x2_history = [x2.copy()]

    # Store infection levels for plotting
    x1_avg = np.average(x1)
    x1_avg_history = [x1_avg]
    x2_avg = np.average(x2)
    x2_avg_history = [x2_avg]
    x = [x1, x2]
    # Simulation loop
    for _ in range(iterations):
        sum_of_x = copy.deepcopy(np.diag(x[0]) + np.diag(x[1]))
        x[0] = x[0] + h * ((np.eye(N) - (sum_of_x)) @ B[0] - np.diag(delta[0])) @ x[0]
        x[1] = x[1] + h * ((np.eye(N) - (sum_of_x)) @ B[1] - np.diag(delta[1])) @ x[1]
        x = np.clip(x, 0, 1)  # Ensure infection levels are between 0 and 1

        x1_history.append(x[0].copy())
        x2_history.append(x[1].copy())

        x1_avg = np.average(x[0])
        x1_avg_history.append(x1_avg)
        x2_avg = np.average(x[1])
        x2_avg_history.append(x2_avg)

    print('x1 equilibrium is '+str(x1_history[-1]))
    print('x2 equilibrium is '+str(x2_history[-1]))

    # determine thm tested
    spectral_radius_1 = np.max(np.abs(np.linalg.eigvals(np.eye(N) + h * (B[0] - np.diag(delta[0])))))
    spectral_radius_2 = np.max(np.abs(np.linalg.eigvals(np.eye(N) + h * (B[1] - np.diag(delta[1])))))

    print('spectral radius 1 is '+str(spectral_radius_1))
    print('spectral radius 2 is '+str(spectral_radius_2))
    
    if spectral_radius_1 < 1 and spectral_radius_2 < 1:
        label = 2
    elif spectral_radius_1 > 1 and spectral_radius_2 > 1:
        det_radius_1 = np.max(np.abs(np.linalg.eigvals(np.eye(N) - h * np.diag(delta[1]) + (np.eye(N) - np.diag(x1_history[-1])) @ B[1])))
        det_radius_2 = np.max(np.abs(np.linalg.eigvals(np.eye(N) - h * np.diag(delta[0]) + (np.eye(N) - np.diag(x2_history[-1])) @ B[0])))
        print('det radius 1 is '+str(det_radius_1))
        print('det radius 2 is '+str(det_radius_2))
        if det_radius_1 < 1 and det_radius_2 < 1:
            label = 5 
            # 1) both (x_1, 0) and (0, x_2) are stable
            # 2) also exists (x_1hat, x_2hat) which is unstable
        elif det_radius_1 > 1 and det_radius_2 > 1:
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
    elif spectral_radius_1 > 1 and spectral_radius_2 <= 1:
        label = 3.1 # Theorem 3, with (x_1, 0) stable

    elif spectral_radius_2 > 1 and spectral_radius_1 <= 1:
        label = 3.2 # Theorem 3, with (0, x_2) stable
 
    print(label)

    return label, spectral_radius_1, spectral_radius_2, x1_avg_history, x2_avg_history, det_radius_1, det_radius_2

# def plot_simulation(x1_history, x2_history):
#     # Plot the results
#     # plt.figure(figsize=(12, 6))
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     for i in range(N):
#         ax1.plot(x1_history[i], label=f'x1_Node {i+1}')
#         ax2.plot(x2_history[i], label=f'x2_Node {i+1}')
#     print(len(x1_history))
#     ax1.set_xlabel('Time step')
#     ax1.set_ylabel('Infection level')
#     # ax1.title('Infection levels of virus 1 over time')
#     # ax1.legend(bbox_to_anchor=(1.05, 1))

#     ax2.set_xlabel('Time step')
#     ax2.set_ylabel('Infection level')
#     # ax2.title('Infection levels of virus 2 over time')
#     # ax2.legend(bbox_to_anchor=(1.05, 1))

#     plt.show()

def plot_simulation(x1_histories, x2_histories):
    # Plot the results  
    fig, axs = plt.subplots(nrows=3, ncols=3)
    # for i in range(N):
    #     plt.plot(x_history[:, i], label=f'Node {i+1}')
    for row in axs:
        for col in row:
            x1_history = x1_histories.pop(0)
            x2_history = x2_histories.pop(0)
            col.plot(x1_history, 'b')
            col.plot(x2_history, 'r')
            if abs(x1_history[-1] - x2_history[-1]) > 0.1:
                col.text(iterations, x1_history[-1], f"({round(x1_history[-1], 2)})", fontsize=8, color = 'b')
                col.text(iterations, x2_history[-1], f"({round(x2_history[-1], 2)})", fontsize=8, color = 'r')
            elif x1_history[-1] > x2_history[-1]:
                col.text(iterations, x1_history[-1] + 0.05, f"({round(x1_history[-1], 2)})", fontsize=8, color = 'b')
                col.text(iterations, x2_history[-1] - 0.05, f"({round(x2_history[-1], 2)})", fontsize=8, color = 'r')
            elif x1_history[-1] <= x2_history[-1]:
                col.text(iterations, x1_history[-1] - 0.05, f"({round(x1_history[-1], 2)})", fontsize=8, color = 'b')
                col.text(iterations, x2_history[-1] + 0.05, f"({round(x2_history[-1], 2)})", fontsize=8, color = 'r')
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

    # this makes sure B^l is non zero and irreducible, satisfying assumptions 4 and 5
    A1, A2 = np.random.uniform(0, W, (N, N)), np.random.uniform(0, W, (N, N))
    # A = np.random.uniform(0, W, (N, N)) # each weight greater than 0. Note that the network must be connected

    delta_1 = []
    delta_2 = []
    beta_1 = []
    beta_2 = []
    for i in range(N):
        # sum_a6 = np.random.uniform(0, 10)
        sum_a6 = 8
        delta_bound = np.random.uniform(0, sum_a6)
        beta_bound = sum_a6 - delta_bound

        delta_1.append(np.random.uniform(0, delta_bound / 5))
        delta_2.append(np.random.uniform(0, delta_bound / 8))

        beta_1_bound = np.random.uniform(0, beta_bound / 6)
        beta_1.append(np.random.uniform(0, beta_1_bound / (np.sum(A1[i]))))
        beta_2.append(np.random.uniform(0, (beta_bound - beta_1_bound) / (np.sum(A2[i]))))

        # beta_1.append(np.random.uniform(0, beta_bound / 40))
        # beta_2.append(np.random.uniform(0, beta_bound / 40))
    
    delta_1 = np.array(delta_1)
    delta_2 = np.array(delta_2)
    beta_1 = np.array(beta_1)
    beta_2 = np.array(beta_2)

    B1 = np.diag(beta_1) @ A1
    B2 = np.diag(beta_2) @ A2

    # just to double check
    print('delta1 is '+str(delta_1))
    print('delta2 is '+str(delta_2))
    print('beta1 is '+ str(beta_1))
    print('beta2 is ' + str(beta_2))
    
    for i in range(N):
        s = delta_1[i]
        for j in range(N):
            s += B1[i, j] + B2[i, j]
        if s > 10:
            print('assumption 6 violated') 

    for i in range(N):
        s = delta_2[i]
        for j in range(N):
            s += B1[i, j] + B2[i, j]
        if s > 10:
            print('assumption 6 violated')

    return [A1, A2], [B1, B2], [beta_1, beta_2], [delta_1, delta_2]


A, B, beta, delta = random_exp()
# # init conditions
# x1x2sum = np.random.uniform(0, 1, N)
# x1 = np.random.uniform(0, x1x2sum, N)
# x2 = x1x2sum - x1

# running the simulation
x1_avg_histories = []
x2_avg_histories = []
x1 = np.random.uniform(0, 1, N)
x2 = np.random.uniform(0, 1, N)
for num1 in [0.25, 0.50, 0.75]:
    num2bound = 1 - num1
    num2list = [num2bound * (i + 1) / 4 for i in range(3)]
    for num2 in num2list:
        x1.fill(num1)
        x2.fill(num2)
        print('x1 is '+str(x1))
        print('x2 is ' + str(x2))
        label, spectral_radius_1, spectral_radius_2, x1_avg_history, x2_avg_history, det_radius_1, det_radius_2 = run_simulation(x1, x2, B, delta)
        x1_avg_histories.append(x1_avg_history)
        x2_avg_histories.append(x2_avg_history)

plot_simulation(x1_avg_histories, x2_avg_histories) 

# if label == 2:
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm2/sr.csv", np.array([spectral_radius_1, spectral_radius_2]), delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm2/A1.csv", A[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm2/A2.csv", A[1], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm2/beta_1.csv", beta[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm2/beta_2.csv", beta[1], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm2/delta_1.csv", delta[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm2/delta_2.csv", delta[1], delimiter=",")
# elif label == 3.1 or label == 3.2:
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm3/sr.csv", np.array([spectral_radius_1, spectral_radius_2]), delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm3/A1.csv", A[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm3/A2.csv", A[1], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm3/beta_1.csv", beta[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm3/beta_2.csv", beta[1], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm3/delta_1.csv", delta[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm3/delta_2.csv", delta[1], delimiter=",")
# elif label == 4.3:
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_stable/sr.csv", np.array([spectral_radius_1, spectral_radius_2]), delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_stable/detr.csv", np.array([det_radius_1, det_radius_2]), delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_stable/A1.csv", A[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_stable/A2.csv", A[1], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_stable/beta_1.csv", beta[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_stable/beta_2.csv", beta[1], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_stable/delta_1.csv", delta[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_stable/delta_2.csv", delta[1], delimiter=",")
# elif label == 4.1:
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_unstable/sr.csv", np.array([spectral_radius_1, spectral_radius_2]), delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_unstable/A1.csv", A[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_unstable/A2.csv", A[1], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_unstable/beta_1.csv", beta[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_unstable/beta_2.csv", beta[1], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_unstable/delta_1.csv", delta[0], delimiter=",")
#     np.savetxt("c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/Double-systems-analysis/rand_thm4_unstable/delta_2.csv", delta[1], delimiter=",")

# # ------------------------------------------------------------------------------------------------------------
# def run_and_save(num_of_exp: int, generation_func, output_file):
#     labels, spectral_radii_1, equilibria_1, spectral_radii_2, equilibria_2, validation, anomalies = list(), list(), list(), list(), list(), list(), list()
#     for _ in range(num_of_exp):
#         B_1, delta_1 = generation_func()
#         B_2, delta_2 = generation_func()
#         B = [B_1, B_2]
#         delta = [delta_1, delta_2]
#         spectral_radius_1, spectral_radius_2, x1_history, x2_history = run_simulation(B, delta)
        
#         equilibrium_1 = x1_history[-1]
#         equilibrium_2 = x2_history[-1]
#         spectral_radii_1.append(spectral_radius_1)
#         spectral_radii_2.append(spectral_radius_2)
#         equilibria_1.append(equilibrium_1)
#         equilibria_2.append(equilibrium_2)

#         if label == 2:
#             flag = True
#             for x in equilibrium_1:
#                 if abs(x) > ZERO_BOUND:
#                     flag = False
#             for x in equilibrium_2:
#                 if abs(x) > ZERO_BOUND:
#                     flag = False
#             validation.append(flag)
#         elif label == 5:
#             validation.append(None)
#         elif label == 4.1:
#             validation.append(None)
#         elif label == 4.2:
#             validation.append(None)
#         elif label == 4.3:
#             validation.append(None)
#         elif label == 4.4:
#             validation.append(None)
#         elif label == 3.1:
#             # x_2 is supposed to die out and x_1 is supposed to survive
#             flag_1 = True
#             for x in equilibrium_1:
#                 flag_1 = flag_1 and (abs(x) > ZERO_BOUND) # flag_1 false if there exists a single zero
#             flag_2 = True
#             for x in equilibrium_2:
#                 if abs(x) > ZERO_BOUND:
#                     flag = False # flag_2 false if there exists a non-zero element
#             validation.append(flag_1 and flag_2)
#         elif label == 3.2:
#             # x_1 is supposed to die out and x_2 is supposed to survive
#             flag_1 = True
#             for x in equilibrium_1:
#                 if abs(x) > ZERO_BOUND:
#                     flag = False # flag_1 false if there exists a non-zero element
#             flag_2 = True
#             for x in equilibrium_2:
#                 flag_2 = flag_2 and (abs(x) > ZERO_BOUND) # flag_2 false if there exists a single zero
#             validation.append(flag_1 and flag_2)

#     df = pd.DataFrame({
#         'label': labels,
#         'spectral_radius_1': spectral_radii_1,
#         'spectral_radius_2': spectral_radii_2,
#         'equilibrium_1': equilibria_1,
#         'equilibrium_2': equilibria_2,
#         'validation': validation
#     })

#     output_dir = os.path.dirname(output_file)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     df.to_csv(output_file, index=False)
#     print(f'Simulation results saved to {output_file}')

#     return anomalies

# num_of_exp = 1000
# # print(run_and_save(num_of_exp, random_exp, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/bivirus_random_param_1000.csv'))
# # print(run_and_save(num_of_exp, same_beta, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/same_beta_1000.csv'))
# # print(run_and_save(num_of_exp, same_delta, 'c:/Users/bloge/OneDrive/Documents/Rice/Research/Virus Simulation/same_delta_1000.csv'))

# ------------------------------------------------------------------------------------------------------------
# Now, we would like to assess rate of convergence
# The idea here is that we find the residual (L2 norm of x and x_next) 
# at every step. We then plot them on an exponential scale,
# and find the best fit line to the residual plot. 
# We can then compare the gradient for the rate of convergence. 
# This is assuming the convergences are exponential, which they seem to be.
