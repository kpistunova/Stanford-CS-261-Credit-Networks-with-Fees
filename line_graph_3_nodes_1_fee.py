import networkx as nx
import numpy as np
import matplotlib
from scipy.optimize import curve_fit
from pyswarm import pso

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transaction_simulator import simulate_transactions_fees, create_random_graph
import time
from scipy.optimize import curve_fit
from scipy.special import lambertw
def simulate_network_network_size_variation(node, capacity_range, transaction_amount, fee, epsilon, window_size, num_runs, avg_degree, checkpointing = False, checkpoint_interval = 20):
    """
    Simulates a credit network with varying capacities and transaction fees, computes the success rate of transactions,
    and optionally saves checkpoints of the simulation results.

    Parameters:
    num_nodes (int): The number of nodes in the credit network graph.
    capacity_range (iterable): A range or sequence of capacities to be tested in the simulation.
    transaction_amount (float): The amount involved in each transaction.
    fee_range (iterable): A range or sequence of transaction fees to be tested.
    epsilon (float): The convergence threshold for the success rate to determine the steady state.
    window_size (int): The number of transactions processed in each iteration.
    num_runs (int): The number of simulation runs for each combination of capacity and fee.
    avg_degree (float): The average out-degree (number of outgoing edges) for nodes in the graph.
    checkpointing (bool): Whether to save checkpoints of the results at intervals.
    checkpoint_interval (int): The interval (in terms of runs) at which to save checkpoints.

    Returns:
    pandas.DataFrame: A DataFrame containing the results of the simulation with columns for capacities,
                      runs, success rates, and fees.

    Note:
    - The function creates a directed graph for each combination of capacity and fee, and for each run,
      simulating transactions to calculate the success rate.
    - Checkpoints are saved as pickle files if checkpointing is enabled.
    """
    results = {
        'run': [],
        'success_rate': [],
        'capacity': [],
        'avg_path_length': []  # New field for average path length
    }
    total_execution_time = 0
    for run in range(num_runs):
        start_time = time.time()
        for capacity in capacity_range:
            # if run == 0:
            #     visualize = True
            # else:
            #     visualize = False
            G = create_random_graph(node, avg_degree, capacity, 'line')
            pos = nx.spring_layout(G)
            success_rate, avg_path_length = simulate_transactions_fees(G, capacity, node, epsilon, fee,
                                                                       transaction_amount, window_size, pos, visualize=False,
                                                                       visualize_initial=0
                                                                       )
            # print(f'Completed run {run}/{num_runs}, degree {degree}, fee {fee}')

            results['run'].append(run)
            results['success_rate'].append(success_rate)
            results['capacity'].append(capacity)
            results['avg_path_length'].append(avg_path_length)


            if checkpointing == True and run % checkpoint_interval == 0:
                checkpoint_df = pd.DataFrame(results)
                checkpoint_filename = f'checkpoint_capacity_fixed_{capacity}_run_{run}.pkl'
                checkpoint_df.to_pickle(checkpoint_filename)
                # print(f'Saved checkpoint to {checkpoint_filename}')
        print(f'Completed run {run}/{num_runs}')
        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
        remaining_fees = num_runs - (run + 1)
        estimated_remaining_time = remaining_fees * (total_execution_time / (run + 1))
        print(f"Processed capacity {run} in time {execution_time} seconds")
        print(f"Estimated remaining time: {estimated_remaining_time / 60} minutes\n")
    return pd.DataFrame(results)

def plot_results_network_size_variation(df, name, size = (8 / 1.2, 6 / 1.2)):
    """
    Plots the results of the network simulation, showing the relationship between edge capacity, fees, and
    transaction success rate.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the simulation results with columns for capacities,
                           success rates, and fees.

    Note:
    - The function generates two plots: a line plot showing success rates against capacities for different fees,
      and a heatmap showing the success rate for each combination of fee and capacity.
    - The plots are saved as image files.
    """
    cmap = sns.cubehelix_palette(as_cmap=True)
    bg_color = plt.gcf().get_facecolor()

    sns.set_theme()
    fig, ax = plt.subplots(figsize=size, dpi=300)
    # sns.lineplot(data=df_filtered, x='nodes', y='success_rate', hue='fee', marker='o', alpha = 0.9, ci='sd', legend='full')
    sns.lineplot(data=df, x='capacity', y='success_rate', marker='o', ci='sd', legend='full')

    plt.xlabel('Capacity', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    plt.ylim([-0.01, 1.1])
    # plt.xlim(left=-10)

    plt.tight_layout()
    fig.savefig(f'{name}_capacity.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=size, dpi=300)
    sns.lineplot(data=df, x='capacity', y='avg_path_length', marker='o', ci='sd', legend='full')
    # Calculate the y-ticks based on the data range and set them to only show one decimal place
    # min_y, max_y = df['avg_path_length'].min(), df['avg_path_length'].max()
    min_y = 1.0
    max_y = 1.328782086291644
    y_ticks = np.arange(np.floor(min_y * 10) / 10, np.ceil(max_y * 10) / 10, 0.1)
    plt.yticks(y_ticks, [f'{tick:.1f}' for tick in y_ticks])
    # Improve the legibility of the plot
    plt.ylabel('Average path length', fontsize=16)
    plt.xlabel('Capacity', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylim(y_ticks[0], y_ticks[-1])

    #
    # ax.xaxis.labelpad = 15
    # ax.yaxis.labelpad = 15
    # Adjust legend
    plt.ylim(top=1.32)
    # Set the limits appropriately

    # Save the figure with tight layout
    plt.tight_layout()
    fig.savefig(f'{name}_path_lenght.png', dpi=300, bbox_inches='tight')
    # Display the plot
    plt.show()



# Configuration
num_nodes = 3
capacity_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 3000, 3500, 4000, 5000]
transaction_amount = 1
fee_range = 1
epsilon = 0.002
num_runs = 100
window_size = 1000
avg_degree = 10
# df = pd.read_pickle('3_node_line_len_vs_fee_capacity_denser.pkl')

# Simulation
df = simulate_network_network_size_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing=False, checkpoint_interval = num_runs)
df.to_pickle('3_node_line_len_vs_fee_capacity_denser.pkl')
plot_results_network_size_variation(df, '3_node_line_denser')
mean_df = df.groupby('capacity')['success_rate'].agg(['mean', 'std']).reset_index()
print('Done!')


# Define the piecewise function with three different behaviors
def piecewise_function(x, a11, b11, c11, a33, x11):
    # Create an empty array to store function values
    a11, b11, c11, a33, x11 = map(float, [a11, b11, c11, a33, x11])
    y = np.empty(x.shape)
    expr = -np.exp(a11 / b11 - a33 / b11 - 1) * (c11 + x11)

    x00 = ((-c11-x11)/(lambertw(expr)) - c11).real
    # Print or log x00 value
    # print("x00 value:", x00)
    if x00 >= 50 or x00 < 10:
        # print('x00 is too large')
        return np.inf
    # Logarithmic for 1 - 20
    cond1 = (x < x00)
    y[cond1] = a11 + b11 * np.log(x[cond1] + c11)
    # Linear for 20 - 2000
    cond2 = (x >= x00) & (x < x11)
    linear_start = a11 + b11 * np.log(x00 + c11)
    linear_slope = (a33 - linear_start) / (x11 - x00)
    y[cond2] = linear_start + linear_slope * (x[cond2] - x00)
    # Constant for 2000 - end
    cond3 = (x >= x11)
    y[cond3] = a33
    return y

# Define your piecewise_function and calculate_x00 here
def calculate_x00(a11, b11, c11, a33, x11):
    # This function calculates x00 based on the parameters
    a11, b11, c11, a33, x11 = map(float, [a11, b11, c11, a33, x11])
    expr = -np.exp(a11 / b11 - a33 / b11 - 1) * (c11 + x11)
    x00 = ((-c11-x11) / (lambertw(expr)) - c11).real
    return x00
def calculate_rmse(y_observed, y_predicted):
    return np.sqrt(np.mean((y_observed - y_predicted)**2))
# max_retries = 5000000
# attempt = 0
# successful_fit = False
# error_threshold = 0.01
# while attempt < max_retries and not successful_fit:
#     if attempt % 10000 == 0:
#         print(f'attempt {attempt}')
#     # Randomize initial guesses within a reasonable range
#     initial_guess = [random.uniform(0, 1), random.uniform(0.01, 0.1), random.uniform(-1, 0),
#                      random.uniform(0, 2), random.uniform(1000, 3000)]
#     try:
#         popt, pcov = curve_fit(piecewise_function, mean_df['capacity'], mean_df['mean'], p0=initial_guess)
#         x00 = calculate_x00(*popt)
#         predicted = piecewise_function(mean_df['capacity'], *popt)
#         rmse_log = calculate_rmse(mean_df['mean'][:7], predicted[:7])
#         rmse_lin = calculate_rmse(mean_df['mean'][7:21], predicted[7:21])
#         rmse_const = calculate_rmse(mean_df['mean'][21:], predicted[21:])
#         rmse = calculate_rmse(mean_df['mean'], predicted)
#         if rmse_log < error_threshold and rmse < error_threshold and rmse_lin < error_threshold and rmse_const < error_threshold:
#             successful_fit = True
#             print(rmse)
#         else:
#             attempt += 1
#     except RuntimeError:
#         attempt += 1
#
# if successful_fit:
#     print("Fit successful:", popt)
# else:
#     print("Failed to fit after", max_retries, "attempts")
def objective_function(params):
    global iteration, start_time
    current_time = time.time()
    elapsed_time = current_time - start_time
    # if iteration % 1000 == 0:
    #     print(f"Iteration {iteration}: Elapsed Time = {elapsed_time:.2f} seconds")
    iteration += 1  # Increment the iteration
    a11, b11, c11, a33, x11 = params

    try:
        popt, pcov = curve_fit(piecewise_function, mean_df['capacity'], mean_df['mean'], p0=[a11, b11, c11, a33, x11])
        x00 = calculate_x00(*popt)
        predicted = piecewise_function(mean_df['capacity'], *popt)
        rmse = calculate_rmse(mean_df['mean'], predicted)

        if predicted is not np.inf:
            # Create boolean masks based on capacity values
            mask_log = mean_df['capacity'] <= 20
            mask_lin = (mean_df['capacity'] > 20) & (mean_df['capacity'] <= 2000)
            mask_const = mean_df['capacity'] > 2000

            # Use the masks to calculate RMSE for each segment
            rmse_log = calculate_rmse(mean_df['mean'][mask_log], predicted[mask_log])
            rmse_lin = calculate_rmse(mean_df['mean'][mask_lin], predicted[mask_lin])
            rmse_const = calculate_rmse(mean_df['mean'][mask_const], predicted[mask_const])
            if rmse_log > 0.019 or rmse_const > 0.0025 or rmse_lin > 0.007:
                return np.inf
        return rmse
    except RuntimeError:
        return np.inf

def objective_function_wrapper(params):
    global iteration, start_time
    result = objective_function(params, iteration, start_time)
    iteration += 1
    return result


# Set the bounds for the parameters in your function
lb = [0.4, 0.01, -1.1, 0.85, 1000]
ub = [0.6, 0.1, -0.8, 0.94, 3000]

global iteration, start_time
iteration = 0
start_time = time.time()

# Use Particle Swarm Optimization
xopt, fopt = pso(objective_function, lb, ub, debug = True, swarmsize=200)
# Initial guess for the parameters
# initial_guess = [ 0.55131752,  0.04371398, -0.99338461,  0.90358792, 2000]  # Params for constant part and transition points
# popt, pcov = curve_fit(piecewise_function, mean_df['capacity'], mean_df['mean'], p0=initial_guess)
# #
x_values = np.linspace(min(mean_df['capacity']), max(mean_df['capacity']), 1000000)
y_fitted = piecewise_function(x_values, *xopt)
plt.figure(figsize=(10, 6))
plt.scatter(mean_df['capacity'], mean_df['mean'], label='Data')
plt.plot(x_values, y_fitted, label='Fitted function', color='red')
plt.xlabel('Capacity')
plt.ylabel('Success Rate')
plt.xlim([-0.01, 25])
plt.legend()
plt.show()