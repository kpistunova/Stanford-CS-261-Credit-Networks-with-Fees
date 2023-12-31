import networkx as nx
import numpy as np
import matplotlib
import random

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
        'degree': [],
        'run': [],
        'success_rate': [],
        'fee': [],
        'capacity': [],
        'avg_path_length': []  # New field for average path length
    }
    total_execution_time = 0
    for run in range(num_runs):
        print(f'started run {run}')
        start_time = time.time()
        for fee in fee_range:
            print(f'Started run {run}/{num_runs},fee {fee}')
            for capacity in capacity_range:
                for degree in avg_degree:
                    G = create_random_graph(num_nodes, degree, capacity, 'er')
                    pos = nx.spring_layout(G)
                    success_rate, avg_path_length = simulate_transactions_fees(G, capacity, degree, epsilon, fee,
                                                                               transaction_amount, window_size, pos, visualize=False
                                                                               )
                    # print(f'Completed run {run}/{num_runs}, capacity {capacity}, fee {fee}, node {node}')

                    results['degree'].append(degree)
                    results['run'].append(run)
                    results['success_rate'].append(success_rate)
                    results['fee'].append(fee)
                    results['capacity'].append(capacity)
                    results['avg_path_length'].append(avg_path_length)
                    # if run % checkpoint_interval == 0:
                    #     print(f'Completed run {run}/{num_runs}, node {node}, capacity {capacity}, fee {fee}')

                    if checkpointing == True and run % checkpoint_interval == 0:
                        checkpoint_df = pd.DataFrame(results)
                        checkpoint_filename = f'checkpoint_random_graph_{capacity}_fee_{fee}_run_{run}_node_{degree}.pkl'
                        checkpoint_df.to_pickle(checkpoint_filename)
                        # print(f'Saved checkpoint to {checkpoint_filename}')
        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
        remaining_fees = num_runs - (run + 1)
        estimated_remaining_time = remaining_fees * (total_execution_time / (run + 1))
        print(f"Processed fee {run} in time {execution_time} seconds")
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
    sns.lineplot(data=df_filtered, x='nodes', y='success_rate', hue='fee', marker='o', alpha = 0.9, ci='sd', legend='full')

    plt.xlabel('Node Number', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.annotate(f'Capacity: {capacity}',
                 xy=(0.5, 1), xycoords='axes fraction',
                 xytext=(0, -10), textcoords='offset points',  # Offset by 10 points from the top
                 horizontalalignment='center', verticalalignment='top',
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))
    # plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
    # plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim([-0.01, 1.1])
    plt.xlim(left=-0.01)

    plt.tight_layout()
    fig.savefig(f'cycle_success_vs_fee_capacity_{capacity}.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    # Use transparency to alleviate overplotting
    sns.lineplot(data=df_filtered, x='avg_path_length', y='success_rate', hue='nodes', style='fee',
                 palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(), markers=True, dashes=False, alpha=0.9, ax=ax)
    # Improve the legibility of the plot
    plt.xlabel('Average path length', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Adjust the position and background color of the annotation
    plt.annotate(f'Capacity: {capacity}',
                 xy=(0.5, 1), xycoords='axes fraction',
                 xytext=(0, -10), textcoords='offset points',  # Offset by 10 points from the top
                 horizontalalignment='center', verticalalignment='top',
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(title='Legend', loc='best', ncol=2, fontsize='x-small', title_fontsize='small')
    # Set the limits appropriately
    plt.ylim([-0.03, 1.1])
    plt.xlim(left=0.95)
    # Save the figure with tight layout
    plt.tight_layout()
    fig.savefig(f'cycle_len_vs_fee_capacity_{capacity}.png', dpi=300, bbox_inches='tight')
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


#


# Configuration
num_nodes = 200
capacity_range = [1, 3 , 5, 7, 9, 10,   50,  70, 100, 500]
transaction_amount = 1
fee_range = [0.0, 0.4, 0.8,  1.0]
epsilon = 0.002
num_runs = 5
avg_degree = [20, 30, 40, 50, 60, 70, 80, 90]
window_size = 1000

# df = pd.read_pickle('random_er_graph_node_vs_fee_all_capacity_after_fix_correct.pkl')
# df_filtered = df[df['fee'] != 0.0]

# for capacity in df_filtered['capacity'].unique():
#     plot_results_network_size_variation(df_filtered, capacity)
# Simulation
df = simulate_network_network_size_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing=False, checkpoint_interval=num_runs)
df.to_pickle('random_er_graph_node_degree_vs_fee_all_capacity_after_fix_correct.pkl')

sns.set_theme()
fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
sns.lineplot(data=df, x='nodes', y='success_rate', hue='capacity', style = 'fee', marker ='o', alpha=0.9, ci='sd', legend=None, palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(), markers=True)

plt.xlabel('Number of Nodes', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
sns.lineplot(data=df, x='capacity', y='success_rate', hue='nodes', style = 'fee', marker='o', alpha=0.9, ci='sd', legend=None, palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(), markers=True)

plt.xlabel('Capacity', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
print('done')