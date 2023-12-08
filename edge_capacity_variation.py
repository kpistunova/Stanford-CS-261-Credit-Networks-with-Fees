import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import nx_cugraph as nxcg # Kate, I had to temporarily comment this out becuase I don't have GPUs -Russell
from transaction_simulator import *
import time
import uuid
from datetime import datetime

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def simulate_network_capacity_fee_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing = False, checkpoint_interval = 20):
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
        'capacity': [],
        'run': [],
        'success_rate': [],
        'fee': [],
        'avg_path_length': []  # New field for average path length
    }
    total_execution_time = 0
    for fee in fee_range:
        start_time = time.time()
        for capacity in capacity_range:
            for run in range(num_runs):
                G = create_random_graph(num_nodes, avg_degree, capacity)
                pos = nx.spring_layout(G)
                success_rate, avg_path_length = simulate_transactions_fees(G, capacity, num_nodes, epsilon, fee,
                                                                           transaction_amount, window_size, pos)
                # print(f'Completed run {run}/{num_runs}, capacity {capacity}, fee {fee}')

                results['capacity'].append(capacity)
                results['run'].append(run)
                results['success_rate'].append(success_rate)
                results['fee'].append(fee)
                results['avg_path_length'].append(avg_path_length)


                if checkpointing == True and run % checkpoint_interval == 0:
                    print(f'Completed run {run}/{num_runs}, capacity {capacity}, fee {fee}')
                    # checkpoint_df = pd.DataFrame(results)
                    # checkpoint_filename = f'checkpoint_capacity_fixed_{capacity}_fee_{fee}_run_{run}.pkl'
                    # checkpoint_df.to_pickle(checkpoint_filename)
                    # print(f'Saved checkpoint to {checkpoint_filename}')
        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
        remaining_fees = len(fee_range) - (fee_range.index(fee) + 1)
        estimated_remaining_time = remaining_fees * (total_execution_time / (fee_range.index(fee) + 1))
        print(f"Processed fee {fee} in time {execution_time} seconds")
        print(f"Estimated remaining time: {estimated_remaining_time/60} minutes\n")
    return pd.DataFrame(results)

def simulate_network_capacity_fee_variation_random_transaction_amounts(num_nodes, capacity_range, transaction_interval, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing = False, checkpoint_interval = 20):
    """
    Simulates a credit network with varying capacities and random transaction fees, computes the success rate of transactions,
    and optionally saves checkpoints of the simulation results.

    Parameters:
    num_nodes (int): The number of nodes in the credit network graph.
    capacity_range (iterable): A range or sequence of capacities to be tested in the simulation.
    transaction_interval (tuple of float): The random interval for random transaction amounts. NOTE THIS IS DIFFERENT FROM THE ORIGINAL simulate_network_capacity_fee_variation
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
        'capacity': [],
        'run': [],
        'success_rate': [],
        'fee': [],
        'avg_path_length': []  # New field for average path length
    }
    total_execution_time = 0
    for fee in fee_range:
        start_time = time.time()
        for capacity in capacity_range:
            for run in range(num_runs):
                G = create_random_graph(num_nodes, avg_degree, capacity)
                pos = nx.spring_layout(G)

                success_rate, avg_path_length = simulate_transactions_fees_random_transaction_amounts(G, capacity, num_nodes, epsilon, fee,
                                                                           transaction_interval, window_size, pos)
                append_results(results, fee, capacity, run, success_rate, avg_path_length)
                
                if checkpointing == True and run % checkpoint_interval == 0:
                    print(f'Completed run {run}/{num_runs}, capacity {capacity}, fee {fee}')

        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
        print_fee_execution_time(fee, execution_time)
        print_estimated_remaining_time(total_execution_time, fee_range, fee)
        
    return pd.DataFrame(results)

def simulate_network_capacity_fee_variation_random_transaction_amounts_percentage_fees(num_nodes, capacity_range, transaction_interval, percentage_fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing = False, checkpoint_interval = 20):
    """
    Simulates a credit network with varying capacities and random transaction fees, computes the success rate of transactions,
    and optionally saves checkpoints of the simulation results.

    Parameters:
    num_nodes (int): The number of nodes in the credit network graph.
    capacity_range (iterable): A range or sequence of capacities to be tested in the simulation.
    transaction_interval (tuple of float): The random interval for random transaction amounts. NOTE THIS IS DIFFERENT FROM THE ORIGINAL simulate_network_capacity_fee_variation
    percentage_fee_range (iterable): A sequence of transaction fees to be tested. These fees are percentages of the transaction, so all element values should be between 0 and 1.  NOTE THIS IS DIFFERENT FROM THE ORIGINAL simulate_network_capacity_fee_variation
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
        'capacity': [],
        'run': [],
        'success_rate': [],
        'percentage_fee': [],
        'avg_path_length': []  # New field for average path length
    }
    total_execution_time = 0
    for percentage_fee in percentage_fee_range:
        start_time = time.time()
        for capacity in capacity_range:
            for run in range(num_runs):
                G = create_random_graph(num_nodes, avg_degree, capacity)
                pos = nx.spring_layout(G)

                success_rate, avg_path_length = simulate_transactions_fees_random_transaction_amounts_percentage_fees(G, capacity, num_nodes, epsilon, percentage_fee,
                                                                           transaction_interval, window_size, pos)
                append_results_percentage_fee(results, percentage_fee, capacity, run, success_rate, avg_path_length)
                
                if checkpointing == True and run % checkpoint_interval == 0:
                    print(f'Completed run {run}/{num_runs}, capacity {capacity}, percentage_fee {percentage_fee}')

        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
        print_fee_execution_time(percentage_fee, execution_time) # lazy, just used the same function
        print_estimated_remaining_time(total_execution_time, percentage_fee_range, percentage_fee) # lazy, just used the same function
        
    return pd.DataFrame(results)

def append_results(results, fee, capacity, run, success_rate, avg_path_length):
    results['fee'].append(fee)
    results['capacity'].append(capacity)
    results['run'].append(run)
    results['success_rate'].append(success_rate)
    results['avg_path_length'].append(avg_path_length)

def append_results_percentage_fee(results, percentage_fee, capacity, run, success_rate, avg_path_length):
    results['percentage_fee'].append(percentage_fee)
    results['capacity'].append(capacity)
    results['run'].append(run)
    results['success_rate'].append(success_rate)
    results['avg_path_length'].append(avg_path_length)

def print_fee_execution_time(fee, execution_time):
    print(f"Processed fee {fee} in time {execution_time} seconds")

def print_estimated_remaining_time(total_execution_time, fee_range, fee):
    remaining_fees = len(fee_range) - (fee_range.index(fee) + 1)
    estimated_remaining_time = remaining_fees * (total_execution_time / (fee_range.index(fee) + 1))
    print(f"Estimated remaining time: {estimated_remaining_time/60} minutes\n")

def generate_filename_timestamp_suffix():
    # Generate a UUID and truncate it to the segment before the first dash
    truncated_uuid = str(uuid.uuid4()).split('-')[0]

    # Format the date with dashes between year, month, and day
    formatted_date = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    return f"_{truncated_uuid}_{formatted_date}"

def plot_results_capacity_fee_variation(df, filename_suffix=""):
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
    sns.set_theme()
    fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    sns.lineplot(data=df, x='capacity', y='success_rate', hue='fee', marker='o', ci='sd', legend='full')

    plt.xlabel('Edge Capacity', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
    # plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim([0, 1.1])
    plt.xlim(left = 0)

    plt.tight_layout()
    fig.savefig(f'capacity_vs_fees_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    # Use transparency to alleviate overplotting
    sns.lineplot(data=df, x='avg_path_length', y='success_rate', hue='capacity', style='fee',
                 palette='coolwarm', markers=True, dashes=False, alpha=0.7, ax=ax)
    # Improve the legibility of the plot
    plt.xlabel('Average path length', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(title='Legend', loc='best', fontsize='x-small', title_fontsize='small')
    # Set the limits appropriately
    plt.ylim([0.3, 1.1])
    plt.xlim([1.5, 3])
    # Save the figure with tight layout
    plt.tight_layout()
    fig.savefig(f'improved_plot_smol_capacity_vs_fees_{filename_suffix}.png', dpi=300)
    # Display the plot
    plt.show()

    # Heatmap
    pivot_table = df.pivot_table(values='success_rate', index='fee', columns='capacity', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, vmin=0, cbar_kws={'label': 'Success Rate'}, square=True)

    plt.title('Success Rate by Fee and Capacity')
    plt.xlabel('Edge Capacity')
    plt.ylabel('Fee')
    plt.savefig(f'heatmap_capacity_vs_fees_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results_capacity_fee_variation_random_transactions_percentage_fees(df, filename_suffix=""):
    """
    Plots the results of the network simulation, showing the relationship between edge capacity, fees, and
    transaction success rate.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the simulation results with columns for capacities,
                           success rates, and percentage fees.

    Note:
    - The function generates two plots: a line plot showing success rates against capacities for different fees,
      and a heatmap showing the success rate for each combination of fee and capacity.
    - The plots are saved as image files.
    """
    cmap = sns.cubehelix_palette(as_cmap=True)
    sns.set_theme()
    fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    sns.lineplot(data=df, x='capacity', y='success_rate', hue='percentage_fee', marker='o', ci='sd', legend='full')

    plt.xlabel('Edge Capacity', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1.1])
    plt.xlim(left = 0)

    plt.tight_layout()
    fig.savefig(f'capacity_vs_percentage_fees_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    # Use transparency to alleviate overplotting
    sns.lineplot(data=df, x='avg_path_length', y='success_rate', hue='capacity', style='percentage_fee',
                 palette='coolwarm', markers=True, dashes=False, alpha=0.7, ax=ax)
    # Improve the legibility of the plot
    plt.xlabel('Average path length', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(title='Legend', loc='best', fontsize='x-small', title_fontsize='small')
    # Set the limits appropriately
    plt.ylim([0.3, 1.1])
    plt.xlim([1.5, 3])
    # Save the figure with tight layout
    plt.tight_layout()
    fig.savefig(f'improved_plot_smol_capacity_vs_percentage_fees_{filename_suffix}.png', dpi=300)
    # Display the plot
    plt.show()

    # Heatmap
    pivot_table = df.pivot_table(values='success_rate', index='percentage_fee', columns='capacity', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, vmin=0, cbar_kws={'label': 'Success Rate'}, square=True)

    plt.title('Success Rate by Percentage Fee and Capacity')
    plt.xlabel('Edge Capacity')
    plt.ylabel('Percentage Fee')
    plt.savefig(f'heatmap_capacity_vs_percentage_fees_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

def identify_outliers(df, column,  multiplier=0.8 ):
    """
    Identifies outliers in a specified column of a DataFrame based on the Interquartile Range (IQR) method.
    This is used primarily for fixed capacity and varied fee analysis, specifically to look for patterns of
    fees that consistently result in higher success probability, such as 0.125, 0.25, 0.5, 0.75, 1 for unit
    transaction and fixed edge capacity = 7. Refer to the slides linked below for futher details.
    https://docs.google.com/presentation/d/1bhEiso-Q2sYQxN6JX1MQkgIfFOvmj-Qwik0GOIxnM1g/edit#slide=id.g29e7a19d305_0_18

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to analyze.
    column (str): The name of the column in the DataFrame to check for outliers.
    multiplier (float): The multiplier for the IQR to adjust the sensitivity of the outlier detection.

    Returns:
    pandas.DataFrame: A DataFrame containing only the rows that are considered outliers in the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier  * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def russell_run_random_transactions_baseline():
    """ The idea for these forthcoming experiments is that we're going to vary transaction amount IN ADDITION TO capacities and fees.
    So this is the baseline configuration for me to compare against.

    i.e. 'Run an experiment where each transaction amount is always 1, and the fees are always a fixed constant'
    """
    name = 'russell_run_random_transactions_baseline'
    print(f"Running experiment {name}")

    # Config for the baseline of the varying transaction amount experiment
    num_nodes = 100
    capacity_range = np.arange(1.0, 16, 1)
    capacity_range = np.append(capacity_range, 20)
    transaction_amount = 1
    fee_range = list(np.round(np.arange(0.0, 1.01, 0.05), 2))
    epsilon = 0.002
    num_runs = 10
    avg_degree = 10
    window_size = 500

    # Simulation
    df = simulate_network_capacity_fee_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing=True)
    df.to_pickle(f'{name}.pkl')
    plot_results_capacity_fee_variation(df, name + generate_filename_timestamp_suffix())

def russell_run_random_transactions_1_2():
    """ Run an experiment where each transaction amount is a random value between [1, 2), and the fees are always a fixed constant
    """
    name = 'russell_run_random_transactions_1_2'
    print(f"Running experiment {name}")

    # Config
    num_nodes = 100
    capacity_range = np.arange(1.0, 16, 1)
    capacity_range = np.append(capacity_range, 20)
    transaction_interval = (1, 2)
    fee_range = list(np.round(np.arange(0.0, 1.01, 0.05), 2))
    epsilon = 0.002
    num_runs = 10
    avg_degree = 10
    window_size = 500

    # Simulation
    df = simulate_network_capacity_fee_variation_random_transaction_amounts(num_nodes, capacity_range, transaction_interval, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing=True)
    df.to_pickle(f'{name}.pkl')
    plot_results_capacity_fee_variation(df, name + generate_filename_timestamp_suffix())

def russell_run_random_transactions_baseline_percentage_fees():
    """ Run an experiment where each transaction amount is always 1, and the fee is a percentage charged at each edge
    """
    name = 'russell_run_random_transactions_baseline_percentage_fees'
    print(f"Running experiment {name}")

    # Config
    num_nodes = 100
    capacity_range = np.arange(1.0, 16, 1)
    capacity_range = np.append(capacity_range, 20)
    transaction_interval = (1, 1)
    fee_percentage_range = list(np.round(np.arange(0.0, 1.01, 0.05), 2))
    epsilon = 0.002
    num_runs = 10
    avg_degree = 10
    window_size = 500

    # Simulation
    df = simulate_network_capacity_fee_variation_random_transaction_amounts_percentage_fees(num_nodes, capacity_range, transaction_interval, fee_percentage_range, epsilon, window_size, num_runs, avg_degree, checkpointing=True)
    df.to_pickle(f'{name}.pkl')
    plot_results_capacity_fee_variation_random_transactions_percentage_fees(df, name + generate_filename_timestamp_suffix()) # use this one for now even though the function name is for random_transactions_percentage_fees

def russell_run_random_transactions_1_2_percentage_fees():
    """ Run an experiment where each transaction amount is a random value between [1, 2), and the fee is a percentage charged at each edge
    """
    name = 'russell_run_random_transactions_1_2_percentage_fees'
    print(f"Running experiment {name}")

    # Config
    num_nodes = 100
    capacity_range = np.arange(1.0, 16, 1)
    capacity_range = np.append(capacity_range, 20)
    transaction_interval = (1, 2)
    fee_percentage_range = list(np.round(np.arange(0.0, 1.01, 0.05), 2))
    epsilon = 0.002
    num_runs = 10
    avg_degree = 10
    window_size = 500

    # Simulation
    df = simulate_network_capacity_fee_variation_random_transaction_amounts_percentage_fees(num_nodes, capacity_range, transaction_interval, fee_percentage_range, epsilon, window_size, num_runs, avg_degree, checkpointing=True)
    df.to_pickle(f'{name}.pkl')
    plot_results_capacity_fee_variation_random_transactions_percentage_fees(df, name + generate_filename_timestamp_suffix())

def kate_run_typical_edge_capacity_variation_experiment():
    """ I've just moved the other "main" code over --Russell
    """
    # Configuration
    num_nodes = 100
    capacity_range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,30]
    # capacity_range = [25, 30, 50, 100, 300]
    transaction_amount = 1
    # fee_range = [2.2, 2.5, 2.7, 3, 4, 5, 6, 7, 8]
    fee_range = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    epsilon = 0.002
    num_runs = 20
    avg_degree = 20
    window_size = 1000
    df = pd.read_pickle('capacity_vs_fees_path_lenght_avg_degree_20.pkl') # Not found in Github  -Russell

    # Simulation
    df = simulate_network_capacity_fee_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing=True)
    df.to_pickle('capacity_vs_fees_path_lenght_avg_degree_20.pkl')
    plot_results_capacity_fee_variation(df)
    #
    # # Plotting
    #----code-fee-variation-plotting-with-fixed-capacity-----------


    # mean_success_rates = df.groupby('fee')['success_rate'].mean().reset_index()
    # top_fees = mean_success_rates.sort_values(by='success_rate', ascending=False).head(7)
    #
    #
    # sns.set_theme()
    # fig, ax = plt.subplots(figsize=(8/ 1.2, 6 / 1.2), dpi=300)
    # lineplot = sns.lineplot(data=df, x='fee', y='success_rate', hue='fee', marker='o', ci='sd',  err_style = 'bars', legend=False, ax=ax)
    # for line in ax.lines[1:]:
    #     line.set_alpha(0.7)
    # plt.xlabel('Fee', fontsize=14)
    # plt.ylabel('Success Rate', fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # # plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
    # # plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
    # plt.ylim(top=1.01)
    # plt.ylim(bottom=0)
    # plt.xlim(left=-0.01)
    # for _, row in top_fees.iterrows():
    #     ax.annotate(f"{row['fee']}",  # Change here to match the format of fees
    #                 xy=(row['fee'], row['success_rate']),
    #                 xytext=(25, 5),  # Adjust to position the text to the top right
    #                 textcoords='offset points',
    #                 ha='right',
    #                 va='bottom')
    # plt.tight_layout()
    # fig.savefig('capacity_fixed_zoom_in_no_errors', dpi=300, bbox_inches='tight')
    # plt.show()


    print('------------------')
    print('Finished!')


    #----code-for-clustering-analysis-----------

    # # Standardize the data
    # scaler = StandardScaler()
    # mean_success_rates_scaled = scaler.fit_transform(mean_success_rates[['fee', 'success_rate']])
    #
    # # Apply DBSCAN
    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # clusters = dbscan.fit_predict(mean_success_rates_scaled)
    #
    # # Adding cluster information to the mean_success_rates dataframe
    # mean_success_rates['cluster'] = clusters
    # mean_success_rates = df[df['cluster'] != -1].groupby('cluster')['success_rate'].mean()
    #
    # # plotting
    # fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    # sns.scatterplot(data=df, x='fee', y='cluster', hue='fee', marker='o', legend=False, ax=ax)
    # # Adding horizontal lines for the average success rates and annotating them
    # for cluster in mean_success_rates.index:
    #     y_coordinate = cluster
    #     ax.axhline(y=y_coordinate, color='gray', linestyle='--', linewidth=1)
    #     ax.text(0.5, y_coordinate - 0.07, f' Avg. Success Rate: {mean_success_rates[cluster]:.2f} ',
    #             transform=ax.get_yaxis_transform(),
    #             ha='center', va='top', color='gray', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    # # Annotate the outlier points with their fee values
    # for index, row in outliers.iterrows():
    #     ax.annotate(f'{row.fee:.3f}', xy=(row.fee, row.cluster), xytext=(0,10), textcoords='offset points', ha='center')
    #
    # plt.xlabel('Fee', fontsize=14)
    # plt.ylabel('Cluster', fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.ylim(top=1.05)
    # plt.ylim(bottom=-1.05)
    # plt.xlim([-0.05, 1.05])
    # plt.tight_layout()
    # fig.savefig('capacity_vs_fees_mean_capacity_cluster.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    print("Hello world!", flush=True)
    russell_run_random_transactions_baseline()
    russell_run_random_transactions_1_2()
    russell_run_random_transactions_baseline_percentage_fees()
    russell_run_random_transactions_1_2_percentage_fees()
    print('------------------', flush=True)
    print('Finished!', flush=True)
