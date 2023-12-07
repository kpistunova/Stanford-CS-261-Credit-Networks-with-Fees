import networkx as nx
import numpy as np
import matplotlib
import matplotlib.lines as mlines

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
                    # if run == 1:
                    #     visualize = True
                    # else:
                    #     visualize = False
                    G = create_random_graph(node, avg_degree, capacity, 'line')
                    pos = nx.spring_layout(G)
                    success_rate, avg_path_length = simulate_transactions_fees(G, capacity, node, epsilon, fee,
                                                                               transaction_amount, window_size, pos, visualize=False
                                                                               )
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
    fig.savefig(f'success_line_len_vs_fee_different_capacity_{capacity}.png', dpi=300, bbox_inches='tight')
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
    fig.savefig(f'path_lenght_line_len_vs_fee_different_capacity_{capacity}.png', dpi=300, bbox_inches='tight')
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
# num_nodes = [2, ]
capacity_range = [2, 3, 4, 5, 8, 10, 15, 20, 30]
transaction_amount = 1
fee_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
epsilon = 0.002
num_runs = 20
avg_degree = 10
window_size = 1000

df = pd.read_pickle('line_len_vs_fee_different_capacity.pkl')
df_filtered = df[df['fee'] != 0.0]

for capacity in df['capacity'].unique():
    plot_results_network_size_variation(df, capacity)
# Simulation
df = simulate_network_network_size_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing=True)
df.to_pickle('line_len_vs_fee_capacity.pkl')
#
# # Plotting
for capacity in df['capacity'].unique():
    plot_results_network_size_variation(df, capacity)

# # Plotting
# plot_results_network_size_variation(df)

# Filter the DataFrame to include only the desired fees
for f in df['fee'].unique():
    cmap = sns.cubehelix_palette(as_cmap=True)
    bg_color = plt.gcf().get_facecolor()
    selected_fees = [f]
    df_filtered = df[df['fee'].isin(selected_fees)]
    palette = sns.color_palette('coolwarm', n_colors=len(df_filtered['capacity'].unique()))

    # Now create the plot with the filtered DataFrame
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)

    # Define line styles for the fees
    line_styles = {f: 'solid'}  # (solid line for 0.3, dashed line for 0)
    marker_styles = {f: 'o'}
    linewidth= {f: 1.5}
    for fee in df_filtered['fee'].unique():
        subset = df_filtered[df_filtered['fee'] == fee]
        alpha_value = 0.8
        ci =  'sd'
        sns.lineplot(data=subset, x='nodes', y='avg_path_length', hue='capacity',
                     palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(),
                     linestyle=line_styles[fee], marker=marker_styles[fee], linewidth=linewidth[fee],
                     alpha=alpha_value, ci=ci, ax=ax, markersize=6, legend = 'auto')
        plt.annotate(f'Fee: {fee}',
                     xy=(0.5, 1), xycoords='axes fraction',
                     xytext=(0, -10), textcoords='offset points',  # Offset by 10 points from the top
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))

    plt.xlabel('Node Number', fontsize=16)
    plt.ylabel('Average Path Length', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    plt.tight_layout()
    fig.savefig(f'average_path_length_vs_node_number_after_fix_fee_{f}.png', dpi=300)
    plt.show()



    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    # Define line styles for the fees

    for fee in df_filtered['fee'].unique():
        subset = df_filtered[df_filtered['fee'] == fee]
        alpha_value =  0.8
        ci = 'sd'
        sns.lineplot(data=subset, x='nodes', y='success_rate', hue='capacity',
                     palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(),
                     linestyle=line_styles[fee], marker=marker_styles[fee], linewidth=linewidth[fee],
                     alpha=alpha_value, ci=ci, ax=ax, markersize=6, legend = 'auto')

        plt.annotate(f'Fee: {fee}',
                     xy=(0.5, 1), xycoords='axes fraction',
                     xytext=(0, -10), textcoords='offset points',  # Offset by 10 points from the top
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))

    plt.xlabel('Node Number', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    plt.tight_layout()
    fig.savefig(f'success_rate_vs_node_number_after_fix_fee_{f}.png', dpi=300)
    plt.show()


for n in df['nodes'].unique():
    selected_nodes = [n]
    df_filtered = df[df['nodes'].isin(selected_nodes)]
    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)    # Define line styles for the fees
    alpha_value = 0.8
    ci = 'sd'
    sns.lineplot(data=df_filtered, x='capacity', y='success_rate', hue='fee',
                 palette='coolwarm', marker='o', linewidth=1.5,
                 alpha=alpha_value, ci=ci, ax=ax, markersize=6, legend = 'auto')

    plt.annotate(f'Nodes: {n}',
                 xy=(0.5, 1), xycoords='axes fraction',
                 xytext=(0, -10), textcoords='offset points',  # Offset by 10 points from the top
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))

    plt.xlabel('Capacity', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([-0.03, 1.1])
    plt.tight_layout()
    fig.savefig(f'success_rate_vs_capacity_vs_fee_fixed_node_number_{n}.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)  # Define line styles for the fees
    alpha_value = 0.8
    ci = 'sd'
    sns.lineplot(data=df_filtered, x='capacity', y='avg_path_length', hue='fee',
                 palette='coolwarm', marker='o', linewidth=1.5,
                 alpha=alpha_value, ci=ci, ax=ax, markersize=6, legend='auto')

    plt.annotate(f'Nodes: {n}',
                 xy=(0.5, 1), xycoords='axes fraction',
                 xytext=(0, -10), textcoords='offset points',  # Offset by 10 points from the top
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))

    plt.xlabel('Capacity', fontsize=16)
    plt.ylabel('Average Path Length', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(bottom = 1 )

    plt.tight_layout()
    fig.savefig(f'Average Path Length_vs_capacity_vs_fee_fixed_node_number_{n}.png', dpi=300)
    plt.show()



# Define the overall figure size
figsize = (20, 15)  # Adjust this as needed

# Set a larger font scale for the plots
# sns.set_context("notebook", font_scale=1.5)

# Create two figures with shared axes, one for each metric
fig1, axs1 = plt.subplots(4, 4, figsize=figsize, dpi=300, sharex=True, sharey=True)  # For success rate
fig2, axs2 = plt.subplots(4, 4, figsize=figsize, dpi=300, sharex=True, sharey=False)  # For average path length

# Flatten the axes arrays for easy iteration
axs1 = axs1.flatten()
axs2 = axs2.flatten()
legend_handles, legend_labels = [], []
# Iterate over each node and plot in the corresponding subplot
for idx, n in enumerate(sorted(df['nodes'].unique())):
    selected_nodes = [n]
    df_filtered = df[df['nodes'].isin(selected_nodes)]

    # Success rate subplot
    ax1 = axs1[idx]
    lineplot1  = sns.lineplot(data=df_filtered, x='capacity', y='success_rate', hue='fee',
                 palette='coolwarm', marker='o', linewidth=1.5,
                 alpha=0.8, ci='sd', ax=ax1, markersize=8)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    # If this is the first plot, save the handles and labels for the legend
    if idx == 0:
        legend_handles, legend_labels = ax1.get_legend_handles_labels()
    ax1.legend().remove()
    # Annotation for node number
    ax1.annotate(f'Nodes: {n}',
                 xy=(0.5, 0.05), xycoords='axes fraction',
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=22, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))

    # Set labels only on the leftmost and bottom subplots to avoid repetition
    # if idx % 4 == 0:  # First column
    #     ax1.set_ylabel('Success Rate')
    # if idx >= 12:  # Bottom row
    #     ax1.set_xlabel('Capacity')

    # Average path length subplot
    ax2 = axs2[idx]

    lineplot2 = sns.lineplot(data=df_filtered, x='capacity', y='avg_path_length', hue='fee',
                 palette='coolwarm', marker='o', linewidth=1.5,
                 alpha=0.8, ci='sd', ax=ax2, markersize=8)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.legend().remove()
    # Annotation for node number
    ax2.annotate(f'Nodes: {n}',
                 xy=(0.5, 0.05), xycoords='axes fraction',
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=22, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))







# Set common x and y labels for the entire figure
fig1.text(0.5, 0.03, 'Capacity', ha='center', va='center', fontsize=24)
fig1.text(0.06, 0.5, 'Success Rate', ha='center', va='center', rotation='vertical', fontsize=24)

fig2.text(0.5, 0.04, 'Capacity', ha='center', va='center', fontsize=24)
fig2.text(0.02, 0.5, 'Average Path Length', ha='center', va='center', rotation='vertical', fontsize=24)

# Add legend to the last subplot (bottom right) for both figures
fig1.legend(handles=legend_handles, labels=legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_labels), fontsize=14)
fig2.legend(handles=legend_handles, labels=legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_labels), fontsize=14)



# Adjust layout to prevent overlapping subplots and to allocate space for the common x and y labels
# plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.07, hspace=0.2, wspace=0.2)
plt.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.08, hspace=0.2, wspace=0.2)  # Adjust bottom to accommodate the legend

# plt.legend()
# Save the figures
fig1.savefig('success_rate_vs_capacity_vs_fee_tiled_subplots.png', dpi=300)
fig2.savefig('avg_path_length_vs_capacity_vs_fee_tiled_subplots.png', dpi=300)

# Display the figures
plt.show()





figsize = (20/1.3, 15/1.3)  # Adjust this as needed

# Set a larger font scale for the plots
# sns.set_context("notebook", font_scale=1.5)

# Create two figures with shared axes, one for each metric
fig1, axs1 = plt.subplots(2, 2, figsize=figsize, dpi=300, sharex=True, sharey=True)  # For success rate
fig2, axs2 = plt.subplots(2, 2, figsize=figsize, dpi=300, sharex=True, sharey=False)  # For average path length

# Flatten the axes arrays for easy iteration
axs1 = axs1.flatten()
axs2 = axs2.flatten()
legend_handles, legend_labels = [], []
# Iterate over each node and plot in the corresponding subplot
for idx, f in enumerate(sorted(df['fee'].unique())):
    selected_nodes = [f]
    df_filtered = df[df['fee'].isin(selected_nodes)]

    # Success rate subplot
    ax1 = axs1[idx]
    if idx == 0:
        leg = 'auto'
    else:
        leg = None

    lineplot1  = sns.lineplot(data=df_filtered, x='nodes', y='success_rate', hue='capacity',
                 palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(), marker='o', linewidth=1.5,
                 alpha=0.8, ci='sd', ax=ax1, markersize=8, legend = None)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    # If this is the first plot, save the handles and labels for the legend

    # Annotation for node number
    ax1.annotate(f'Fee: {f}',
                 xy=(0.5, 0.95), xycoords='axes fraction',
                 horizontalalignment='center', verticalalignment='top',
                 fontsize=22, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))

    # Set labels only on the leftmost and bottom subplots to avoid repetition
    # if idx % 4 == 0:  # First column
    #     ax1.set_ylabel('Success Rate')
    # if idx >= 12:  # Bottom row
    #     ax1.set_xlabel('Capacity')

    # Average path length subplot
    ax2 = axs2[idx]

    lineplot2 = sns.lineplot(data=df_filtered, x='nodes', y='avg_path_length', hue='capacity',
                 palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(), marker='o', linewidth=1.5,
                 alpha=0.8, ci='sd', ax=ax2, markersize=8, legend = leg)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    # Annotation for node number
    ax2.annotate(f'Fee: {f}',
                 xy=(0.5, 0.95), xycoords='axes fraction',
                 horizontalalignment='center', verticalalignment='top',
                 fontsize=22, bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor=bg_color))







# Set common x and y labels for the entire figure
fig1.text(0.5, 0.03, 'Node Number', ha='center', va='center', fontsize=24)
fig1.text(0.06, 0.5, 'Success Rate', ha='center', va='center', rotation='vertical', fontsize=24)

fig2.text(0.5, 0.03, 'Node Number', ha='center', va='center', fontsize=24)
fig2.text(0.01, 0.5, 'Average Path Length', ha='center', va='center', rotation='vertical', fontsize=24)

# # Add legend to the last subplot (bottom right) for both figures
# fig1.legend(handles=legend_handles, labels=legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_labels), fontsize=14)
# fig2.legend(handles=legend_handles, labels=legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_labels), fontsize=14)
#
#

# Adjust layout to prevent overlapping subplots and to allocate space for the common x and y labels
# plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.07, hspace=0.2, wspace=0.2)
plt.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.08, hspace=0.2, wspace=0.2)  # Adjust bottom to accommodate the legend

# plt.legend()
# Save the figures
fig1.savefig('success_rate_vs_node_number_vs_fee_tiled_subplots.png', dpi=300)
fig2.savefig('avg_path_length_vs_node_number_vs_fee_tiled_subplots.png', dpi=300)

# Display the figures
plt.show()


df['scale'] = df['capacity'] / (df['nodes'] ** 2)

sns.set_theme()
fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
sns.lineplot(data=df, x='scale', y='success_rate', hue='fee', marker='o', alpha=0.9, ci='sd', legend=None)

plt.xlabel(r'$\frac{c}{n}$', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.title('Cycle graph, fee = 0', fontsize=14)
plt.ylim([-0.01, 1.1])
plt.xlim([-0.01, 1.1])
plt.tight_layout()
fig.savefig(f'cycle_success_graph_c_div_by_n.png', dpi=300, bbox_inches='tight')
plt.show()
