import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nx_cugraph as nxcg
from transaction_simulator import simulate_transactions_fees, create_random_graph
import time

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
def simulate_network_node_degree_fee_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing = False, checkpoint_interval = 5):
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
        'avg_degree': [],
        'run': [],
        'success_rate': [],
        'fee': [],
        'avg_path_length': [],
        'capacity': [],
    }
    total_execution_time = 0
    for fee in fee_range:
        start_time = time.time()
        for capacity in capacity_range:
            for degree in avg_degree:
                for run in range(num_runs):
                    G = create_random_graph(num_nodes, degree, capacity)
                    pos = nx.spring_layout(G)
                    success_rate, avg_path_length = simulate_transactions_fees(G, capacity, num_nodes, epsilon, fee,
                                                                               transaction_amount, window_size, pos)
                    # print(f'Completed run {run}/{num_runs}, capacity {capacity}, fee {fee}')
                    results['avg_degree'].append(degree)
                    results['run'].append(run)
                    results['success_rate'].append(success_rate)
                    results['fee'].append(fee)
                    results['avg_path_length'].append(avg_path_length)
                    results['capacity'].append(capacity)
                    if checkpointing == True and run % 10000 == 0:
                        checkpoint_df = pd.DataFrame(results)
                        checkpoint_filename = f'checkpoint_avg_degree_fixed_{degree}_fee_{fee}_run_{run}_capacity_{capacity}.pkl'
                        checkpoint_df.to_pickle(checkpoint_filename)
                        print(f'Saved checkpoint to {checkpoint_filename}')
        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
        remaining_fees = len(fee_range) - (fee_range.index(fee) + 1)
        estimated_remaining_time = remaining_fees * (total_execution_time / (fee_range.index(fee) + 1))
        print(f"Processed fee {fee} in time {execution_time} seconds")
        print(f"Estimated remaining time: {estimated_remaining_time / 60} minutes\n")
    return pd.DataFrame(results)

def plot_results_node_degree_fee_variation(df, capacity):
    """
    Plots the results of the network simulation.
    """
    df_filtered = df[df['capacity'] == capacity]
    cmap = sns.cubehelix_palette(as_cmap=True)
    sns.set_theme()
    bg_color = plt.gcf().get_facecolor()
    fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    sns.lineplot(data=df_filtered, x='avg_degree', y='success_rate', hue='fee', marker='o', ci='sd', legend='full')

    plt.xlabel('Average Node Degree', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Adjust the position and background color of the annotation
    plt.annotate(f'Capacity: {capacity}',
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 horizontalalignment='right', verticalalignment='bottom',
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))

    # plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
    # plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim([0.0, 1.1])
    plt.xlim(left=1)

    plt.tight_layout()
    fig.savefig(f'node_degree_vs_fees_capacity_{capacity}.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    # Use transparency to alleviate overplotting
    sns.lineplot(data=df_filtered, x='avg_path_length', y='success_rate', hue='avg_degree', style='fee',
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
    fig.savefig(f'node_degree_vs_fees_capacity_path_lenght_{capacity}.png', dpi=300)
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
num_nodes = 200
capacity_range = [3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 30]
transaction_amount = 1
fee_range = [0, 0.3, 0.5, 0.8, 1]
# fee_range = np.round(np.arange(0.0, 1.1, 0.1), 2)
epsilon = 0.002
num_runs = 20
avg_degree = [10, 12, 15, 17, 20, 30, 40, 50, 60, 70, 80, 90, 99]
window_size = 1000

df = pd.read_pickle('node_degree_capacity_all_variation_with_length.pkl')
for capacity in df['capacity'].unique():
    plot_results_node_degree_fee_variation(df, capacity)
# Simulation
df = simulate_network_node_degree_fee_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing=True)
df.to_pickle('node_degree_capacity_all_variation_with_length.pkl')
#
# # Plotting
plot_results_node_degree_fee_variation(df)

#------path_lenght_vs_node_degre_analysis_for_fee_0.3


# Filter the DataFrame to include only the desired fees
selected_fees = [0.3]
df_filtered = df[df['fee'].isin(selected_fees)]

# Now create the plot with the filtered DataFrame
sns.set_theme()
fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)

# Define line styles for the fees
line_styles = {0: (2, 2), 0.3: (1, 0)}  # (solid line for 0.3, dashed line for 0)

# Use the filtered DataFrame for plotting, with line styles based on fees
sns.lineplot(data=df_filtered, x='avg_degree', y='avg_path_length', hue='capacity', style='fee',
             palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(),
             dashes=[line_styles[fee] for fee in df_filtered['fee'].unique()],
             markers=True, alpha=0.9, ci=95, ax=ax, legend='full')
# Plot the average path length for fee 0 as a dashed line
sns.lineplot(data=mean_path_length_fee_0, x='avg_degree', y='avg_path_length',
             label='Fee 0 Average', markers=True, linestyle='--', color='red', ax=ax)

plt.xlabel('Average Node Degree', fontsize=16)
plt.ylabel('Average Path Length', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adjust line width for better visibility
for line in ax.lines:
    line.set_linewidth(2)

# Adjust the legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(title='Legend', loc='best', ncol=2, fontsize='x-small', title_fontsize='small')

plt.tight_layout()
fig.savefig('average_path_length_for_selected_fees.png', dpi=300)
plt.show()


#-------------------------- Effect of varying graph density-----------------------
#
# success_rates = []
# degree_range = [2, 5, 10, 20]
#
# results = {
#     'avg_degree': [],
#     'run': [],
#     'success_rate': [],
#     'fee': [],
# }
# for fee in fee_range:
#     for avg_degree in degree_range:
#         for run in range(num_runs):
#             G = create_random_graph(num_nodes, avg_degree, fixed_capacity)
#             pos = nx.spring_layout(G)
#             success_rate = simulate_transactions_fees(G, num_nodes, epsilon, fee, transaction_amount, pos)
#             print(f'Completed run {run}/{num_runs} for nodes {num_nodes}, avg_degree {avg_degree}, fee {fee}')
#             results['avg_degree'].append(avg_degree)
#             results['run'].append(run)
#             results['success_rate'].append(success_rate)
#             results['fee'].append(fee)  # Keep track of the fee for this simulation
#
#
#
#
# # df_fees_0 = pd.DataFrame(results)
# df_fees_2 = pd.DataFrame(results)
# # stats_df_fees = df_fees.groupby('avg_degree')['success_rate'].agg(['mean', 'std']).reset_index()
# sns.set_theme()
# # sns.lineplot(data=stats_df_fees, x='avg_degree', y='mean', marker = 'o')
# # plt.fill_between(stats_df_fees['avg_degree'], stats_df_fees['mean'] - stats_df_fees['std'], stats_df_fees['mean'] + stats_df_fees['std'], alpha=0.3)
# fig = plt.figure(figsize=(8/1.2, 6/1.2), dpi=300)
# sns.lineplot(data=df_fees_2, x='avg_degree', y='success_rate', hue='fee', marker='o', ci='sd', linewidth=2.5, markersize=8)
#
#
# plt.xlabel('Average Degree', fontsize=14)
# plt.ylabel('Success Rate', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
# plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
#
# plt.tight_layout()
# fig.savefig('c_1_t_1_an_nodes.png', dpi=300, bbox_inches='tight')
# plt.show()
#
#
#
# print('------------------')
# print('Finished!')