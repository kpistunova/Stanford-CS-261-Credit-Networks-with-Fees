import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transaction_simulator import simulate_transactions_fees, create_random_graph
import time

def simulate_network_network_size_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing = False, checkpoint_interval = 20):
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
        'nodes': [],
        'run': [],
        'success_rate': [],
        'fee': [],
        'capacity': [],
        'avg_path_length': []  # New field for average path length
    }
    total_execution_time = 0
    for fee in fee_range:
        start_time = time.time()
        for capacity in capacity_range:
            for node in num_nodes:
                for run in range(num_runs):
                    G = create_random_graph(node, avg_degree, capacity, 'line')
                    pos = nx.spring_layout(G)
                    success_rate, avg_path_length = simulate_transactions_fees(G, 1, node, epsilon, fee,
                                                                               transaction_amount, window_size, pos,
                                                                               visualize=True)
                    # print(f'Completed run {run}/{num_runs}, degree {degree}, fee {fee}')

                    results['nodes'].append(node)
                    results['run'].append(run)
                    results['success_rate'].append(success_rate)
                    results['fee'].append(fee)
                    results['capacity'].append(capacity)
                    results['avg_path_length'].append(avg_path_length)
                    if run % checkpoint_interval == 0:
                        print(f'Completed run {run}/{num_runs}, node {node}, capacity {capacity}, fee {fee}')

                    if checkpointing == True and run % checkpoint_interval == 0:
                        checkpoint_df = pd.DataFrame(results)
                        checkpoint_filename = f'checkpoint_capacity_fixed_{capacity}_fee_{fee}_run_{run}_node_{node}.pkl'
                        checkpoint_df.to_pickle(checkpoint_filename)
                        # print(f'Saved checkpoint to {checkpoint_filename}')
            end_time = time.time()
            execution_time = end_time - start_time
            total_execution_time += execution_time
            remaining_fees = len(fee_range) - (fee_range.index(fee) + 1)
            estimated_remaining_time = remaining_fees * (total_execution_time / (fee_range.index(fee) + 1))
            print(f"Processed fee {fee} in time {execution_time} seconds")
            print(f"Estimated remaining time: {estimated_remaining_time / 60} minutes\n")
    return pd.DataFrame(results)

def plot_results_network_size_variation(df, capacity):
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
    df_filtered = df[df['capacity'] == capacity]

    cmap = sns.cubehelix_palette(as_cmap=True)
    bg_color = plt.gcf().get_facecolor()

    sns.set_theme()
    fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    sns.lineplot(data=df_filtered, x='nodes', y='success_rate', hue='fee', marker='o', ci='sd', legend='full')

    plt.xlabel('Node Number', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.annotate(f'Capacity: {capacity}',
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 horizontalalignment='right', verticalalignment='bottom',
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))
    # plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
    # plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim([0, 1.1])
    plt.xlim(left=0)

    plt.tight_layout()
    fig.savefig(f'network_size_var_capacity_{capacity}.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    # Use transparency to alleviate overplotting
    sns.lineplot(data=df_filtered, x='avg_path_length', y='success_rate', hue='nodes', style='fee',
                 palette='coolwarm', markers=True, dashes=False, alpha=0.7, ax=ax)
    # Improve the legibility of the plot
    plt.xlabel('Average path length', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Adjust the position and background color of the annotation
    plt.annotate(f'Capacity: {capacity}',
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 horizontalalignment='right', verticalalignment='bottom',
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))

    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(title='Legend', loc='best', ncol=2, fontsize='x-small', title_fontsize='small')
    # Set the limits appropriately
    plt.ylim([0.0, 1.1])
    plt.xlim(left=0.9)
    # Save the figure with tight layout
    plt.tight_layout()
    fig.savefig(f'network_size_var_capacity_path_lenght_{capacity}.png', dpi=300, bbox_inches='tight')
    # Display the plot
    plt.show()
    # Heatmap
    # pivot_table = df.pivot_table(values='success_rate', index='fee', columns='capacity', aggfunc='mean')
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, vmin=0, cbar_kws={'label': 'Success Rate'}, square=True, legend=False)
    #
    # plt.title('Success Rate by Fee and Capacity')
    # plt.xlabel('Edge Capacity')
    # plt.ylabel('Fee')
    # plt.savefig('heatmap_capacity_vs_fees_vm', dpi=300, bbox_inches='tight')
    # plt.show()


# Configuration
num_nodes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
capacity_range = [2, 3, 4, 5, 8, 10, 15, 20, 30]
transaction_amount = 1
fee_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
epsilon = 0.002
num_runs = 20
avg_degree = 10
window_size = 10000

# df = pd.read_pickle('network_size_capacity_all.pkl')

# Simulation
df = simulate_network_network_size_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing=True)
df.to_pickle('line_len_vs_fee_capacity.pkl')
#
# # Plotting
for capacity in df['capacity'].unique():
    plot_results_network_size_variation(df, capacity)