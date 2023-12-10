import networkx as nx
import random
from transaction_simulator import simulate_transactions_fees, create_random_graph, update_graph_capacity_fees
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import pickle

def simulate_random_walk(G, num_steps, transaction_amount, fee):
    state_frequencies = {}
    for step in range(num_steps):
        if step % 1000 ==0:
            print(f"starting step {step}/{num_steps}")
        # Randomly select source (s) and sink (t) nodes
        s, t = random.sample(G.nodes, 2)

        # Attempt to route the transaction
        try:
            path = nx.shortest_path(G, source=s, target=t)
            transaction_succeeded = update_graph_capacity_fees(G, path, transaction_amount, fee)
            if transaction_succeeded:
                # Convert the current network state to a hashable format (e.g., tuple)
                current_state = tuple(sorted((u, v, round(G[u][v]['capacity'], 2)) for u, v in G.edges()))
                state_frequencies[current_state] = state_frequencies.get(current_state, 0) + 1
        except nx.NetworkXNoPath:
            pass

    # Normalize frequencies to get probabilities
    total = sum(state_frequencies.values())
    stationary_distribution = {state: freq / total for state, freq in state_frequencies.items()}

    return stationary_distribution

# Parameters


# def plot_stationary_distribution(stationary_distribution):
#     # Extract states and their probabilities
#     states = list(stationary_distribution.keys())
#     probabilities = list(stationary_distribution.values())
#
#     # Convert state tuples to string labels for better readability
#     state_labels = [f'State {i+1}' for i in range(len(states))]
#
#     # Plotting
#     plt.figure(figsize=(12, 6))
#     plt.bar(state_labels, probabilities, color='skyblue')
#     plt.xlabel('States')
#     plt.ylabel('Probability')
#     plt.title('Stationary Distribution of Network States')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#
#     plt.show()

# After calculating the stationary_distribution in your main script, call the plotting function:


def plot_network_states(G, selected_states, state_probabilities):
    fig, axes = plt.subplots(nrows=1, ncols=len(selected_states), figsize=(20, 5))
    pos = nx.spring_layout(G)
    if len(selected_states) == 1:  # Adjust if there's only one state to plot
        axes = [axes]

    for ax, state in zip(axes, selected_states):
        # Create a graph for the current state
        H = nx.DiGraph()

        for (u, v, capacity) in state:
            H.add_edge(u, v, capacity=capacity)

        # Position nodes using the specified layout
        pos = pos

        # Draw the network
        nx.draw(H, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=700, font_size=10)

        # Draw edge labels (capacities)
        edge_labels = nx.get_edge_attributes(H, 'capacity')

        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, ax=ax)

        # Title with state info and probability
        probability = state_probabilities[state]
        state_label = ', '.join([f'({u},{v},{c})' for u, v, c in state])
        ax.set_title(f"State: {state_label}\nProbability: {probability:.4f}")

    plt.tight_layout()
    plt.show()


current_date = datetime.now().strftime('%Y-%m-%d')
base_directory = 'data'
new_directory_path = os.path.join(base_directory, current_date)
# Check if the directory does not exist
if not os.path.exists(new_directory_path):
    # If it doesn't exist, create a new directory
    os.makedirs(new_directory_path)
transaction_amount = 1
# fees = [0.0]
# fee_range = np.round(np.arange(0.0, 1.1, 0.1), 2)
epsilon = 0.002
num_runs = 1
avg_degree = 10
window_size = 10000
# Configuration
num_nodes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 100]
# num_nodes = [2, ]
capacity_range = [2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50, 100, 200, 10000]
transaction_amount = 1
fees = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# capacity_range = [1, 2, 3]
# capacity_range = np.concatenate((np.arange(1.0, 11.5, 0.5), [12, 15, 20, 50, 100]))

# transaction_amount = 1
# fee_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
results = {
    'fee': [],
    'success_rate': [],
    'run': [],
    'avg_path_length': [],
    'node': [],
    'capacity': []
}
filename = 'line_random_walk.pkl'
file_path = os.path.join(new_directory_path, filename)
# df = pd.read_pickle(file_path)
total_execution_time = 0

# # Create the network and simulate the random walk
# G = create_random_graph(num_nodes, avg_degree, fixed_total_capacity, type='line')
# stationary_distribution = simulate_random_walk(G, num_steps, transaction_amount, fee)
# print("Approximate Stationary Distribution:")
# for state, probability in sorted(stationary_distribution.items(), key=lambda x: -x[1]):
#     print(f"State: {state}, Probability: {probability:.4f}")

for run in range(num_runs):
    for node in num_nodes:
        print(f'started node {node}')
        start_time = time.time()
        for fee in fees:
            for capacity in capacity_range:
                G = create_random_graph(node, avg_degree, capacity, 'line')
                # Create a reference copy of the graph
                pos = nx.spring_layout(G)
                G_reference = G.copy()
                # stationary_distribution = simulate_random_walk(G, 10000, transaction_amount, fee)
                # print("Approximate Stationary Distribution:")
                # for state, probability in sorted(stationary_distribution.items(), key=lambda x: -x[1]):
                #     print(f"State: {state}, Probability: {probability:.4f}")
                # top_states = sorted(stationary_distribution, key=stationary_distribution.get, reverse=True)

                # Plot the network states
                # plot_network_states(G, top_states, stationary_distribution, pos)

                success_rate, avg_path_length , stationary_distribution= simulate_transactions_fees(G, capacity, node, epsilon, fee,
                                                                           transaction_amount, window_size, pos,
                                                                           visualize=False, save = False, show=False
                                                                           )
                # top_states = sorted(stationary_distribution, key=stationary_distribution.get, reverse=True)
                #
                # # Plot the network states
                # plot_network_states(G, top_states, stationary_distribution)
                print(f'Completed run {run}/{num_runs}, capacity {capacity}, fee {fee}, node {node}')
                # Save the stationary distribution to a pickle file
                pickle_filename = f'stationary_distribution_run_{run}_node_{node}_fee_{fee}_capacity_{capacity}.pkl'
                pickle_filepath = os.path.join(new_directory_path, pickle_filename)
                with open(pickle_filepath, 'wb') as handle:
                    pickle.dump(stationary_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)

                results['fee'].append(fee)
                results['success_rate'].append(success_rate)
                results['run'].append(run)
                results['avg_path_length'].append(avg_path_length)
                results['node'].append(node)
                results['capacity'].append(capacity)

        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
        remaining_fees = len(num_nodes) - (num_nodes.index(node) + 1)
        estimated_remaining_time = remaining_fees * (total_execution_time / (num_nodes.index(node) + 1))
        print(f"Processed node {node} in time {execution_time} seconds")
        print(f"Estimated remaining time: {estimated_remaining_time / 60} minutes\n")
df = pd.DataFrame(results)

df.to_pickle(file_path)

# df['scale'] = 1 - (1/df['capacity'])
df['scale'] = df['capacity'] / (df['capacity'] + 1)

sns.set_theme()
fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
sns.lineplot(data=df, x='scale', y='success_rate', hue='node', style='fee', marker='o', alpha=0.9, ci='sd',
             legend='full')

plt.xlabel(r'$\frac{c}{c+1}$', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.title('n: 2 -> 1000, c: 2 -> 15', fontsize=14)
# plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
# plt.ylim([-0.01, 1.1])
# plt.xlim([-0.01, 1.1])

plt.tight_layout()
# fig.savefig(f'success_line_graph_vs_c_div_by_n_squared.png', dpi=300, bbox_inches='tight')
plt.show()

top_states = sorted(stationary_distribution, key=stationary_distribution.get, reverse=True)

# Plot the network states
plot_network_states(G, top_states, stationary_distribution)
print("Approximate Stationary Distribution:")
