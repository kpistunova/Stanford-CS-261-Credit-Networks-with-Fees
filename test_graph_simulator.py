import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_random_graph(num_nodes, avg_degree, fixed_total_capacity):
    num_edges = int(avg_degree * num_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))  # Ensure all nodes are added

    while G.number_of_edges() < num_edges :  # Multiplying by 2 since each edge is one-way in DiGraph
        u, v = random.sample(G.nodes, 2)  # Select from actual nodes

        # Add the edge only if the reverse edge does not exist
        if not G.has_edge(v, u):
            G.add_edge(u, v, capacity=fixed_total_capacity)

    return G

def update_graph_capacity_fees(G, path, transaction_amount, fee):
    fees = [(len(path) - i - 2) * fee for i in range(len(path) - 1)]

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        required_capacity = transaction_amount + fees[i]
        if G[u][v]['capacity'] < required_capacity:
            return False  # Transaction failed due to insufficient capacity

        G[u][v]['capacity'] -= required_capacity
        if G[u][v]['capacity'] == 0:
            G.remove_edge(u, v)

        if G.has_edge(v, u):
            G[v][u]['capacity'] += required_capacity
        else:
            G.add_edge(v, u, capacity=required_capacity)

    return True


def simulate_transactions_fees(G, num_nodes, epsilon, fee, transaction_amount, pos=None, snapshot_interval=100):
    total_transactions = 0
    visualize_initial = 7
    successful_transactions = 0
    window_size = 500
    prev_success_rate = -1
    # edge_usage_frequency = defaultdict(int)
    # edge_capacity_over_time = defaultdict(lambda: defaultdict(list))  # Tracks capacity over time for each edge

    while True:
        for _ in range(window_size):
            s, t = random.sample(range(num_nodes), 2)
            try:
                path = nx.shortest_path(G, s, t)
                # Direct capacity check
                if min([G[u][v]['capacity'] for u, v in zip(path, path[1:])]) > 0:
                    transaction_succeeded = update_graph_capacity_fees(G, path, transaction_amount, fee)
                    if transaction_succeeded:
                        successful_transactions += 1

            except nx.NetworkXNoPath:
                pass

            total_transactions += 1

        current_success_rate = successful_transactions / total_transactions
        if prev_success_rate != -1 and abs(current_success_rate - prev_success_rate) < epsilon:
            break
        prev_success_rate = current_success_rate

    return current_success_rate
#-----------------------------------------feeeeeeeeeees

num_nodes = 100
# avg_degree = 20
capacity_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20]

fixed_capacity = 2 # Set the fixed capacity for each edge
transaction_amount = 1
fee_range = [0, 0.000001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# fee_range = [0.01]

epsilon = 0.002
num_runs = 5



success_rates = []
avg_degree = 10
results = {
    'capacity': [],
    'run': [],
    'success_rate': [],
    'fee': [],
}

all_capacity_over_time = []

checkpoint_interval = 100
for fee_index, fee in enumerate(fee_range):
    for capacity_index, capacity in enumerate(capacity_range):
        for run in range(num_runs):
            G = create_random_graph(num_nodes, avg_degree, capacity)
            pos = nx.spring_layout(G)
            success_rate = simulate_transactions_fees(G, num_nodes, epsilon, fee, transaction_amount, pos)
            print(f'Completed run {run}/{num_runs}, capacity {capacity}, fee {fee}')
            # Append each capacity measurement to the list with corresponding fee and capacity

            results['capacity'].append(capacity)
            results['run'].append(run)
            results['success_rate'].append(success_rate)
            results['fee'].append(fee)

            # Checkpointing: save the DataFrame every n iterations
            # if run % checkpoint_interval == 0 or (run == num_runs - 1 and capacity_index == len(capacity_range) - 1 and fee_index == len(fee_range) - 1):
            #     checkpoint_df = pd.DataFrame(results)
            #     checkpoint_filename = f'checkpoint_capacity_{capacity}_fee_{fee}_run_{run}.pkl'
            #     checkpoint_df.to_pickle(checkpoint_filename)
            #     print(f'Saved checkpoint to {checkpoint_filename}')



df_capacity_1_to_20_fee_0_to_0p5_denser = pd.DataFrame(results)
df_capacity_1_to_20_fee_0_to_0p5_denser.to_pickle('df_capacity_1_to_20_fee_0_to_0p5_denser.pkl')

# After all simulations, create a DataFrame for plotting

df = pd.DataFrame(results)
# df['fee'] = df['fee'].map('{:.4f}'.format)

sns.set_theme()



# sns.lineplot(data=stats_df_fees, x='avg_degree', y='mean', marker = 'o')
# plt.fill_between(stats_df_fees['avg_degree'], stats_df_fees['mean'] - stats_df_fees['std'], stats_df_fees['mean'] + stats_df_fees['std'], alpha=0.3)
fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
sns.lineplot(data=df, x='capacity', y='success_rate', hue='fee', marker='o', ci='sd', linewidth=2.5, markersize=8,  legend="full")

plt.xlabel('Edge capacity', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim([0, 1.1])
plt.xlim([0, 21])

plt.tight_layout()
fig.savefig('df_capacity_1_to_20_fee_0_to_0p5_denser.png', dpi=300, bbox_inches='tight')
plt.show()




print('------------------')
print('Finished!')
print('------------------')
print('Finished!')


#-------------------------- Effect of varying graph density-----------------------

success_rates = []
degree_range = [2, 5, 10, 20]

results = {
    'avg_degree': [],
    'run': [],
    'success_rate': [],
    'fee': [],
}
for fee in fee_range:
    for avg_degree in degree_range:
        for run in range(num_runs):
            G = create_random_graph(num_nodes, avg_degree, fixed_capacity)
            pos = nx.spring_layout(G)
            success_rate = simulate_transactions_fees(G, num_nodes, epsilon, fee, transaction_amount, pos)
            print(f'Completed run {run}/{num_runs} for nodes {num_nodes}, avg_degree {avg_degree}, fee {fee}')
            results['avg_degree'].append(avg_degree)
            results['run'].append(run)
            results['success_rate'].append(success_rate)
            results['fee'].append(fee)  # Keep track of the fee for this simulation




# df_fees_0 = pd.DataFrame(results)
df_fees_2 = pd.DataFrame(results)
# stats_df_fees = df_fees.groupby('avg_degree')['success_rate'].agg(['mean', 'std']).reset_index()
sns.set_theme()
# sns.lineplot(data=stats_df_fees, x='avg_degree', y='mean', marker = 'o')
# plt.fill_between(stats_df_fees['avg_degree'], stats_df_fees['mean'] - stats_df_fees['std'], stats_df_fees['mean'] + stats_df_fees['std'], alpha=0.3)
fig = plt.figure(figsize=(8/1.2, 6/1.2), dpi=300)
sns.lineplot(data=df_fees_2, x='avg_degree', y='success_rate', hue='fee', marker='o', ci='sd', linewidth=2.5, markersize=8)


plt.xlabel('Average Degree', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
fig.savefig('c_1_t_1_an_nodes.png', dpi=300, bbox_inches='tight')
plt.show()



print('------------------')
print('Finished!')



def simulate_transactions(G, num_nodes, epsilon, pos, visualize_initial=4, visualize_every_n=1000):
    total_transactions = 0
    successful_transactions = 0
    window_size = 1000
    prev_success_rate = -1

    while True:
        for _ in range(window_size):
            s, t = random.sample(range(num_nodes), 2)
            try:
                path = nx.shortest_path(G, s, t, weight='capacity')
                if min([G[u][v]['capacity'] for u, v in zip(path, path[1:])]) > 0:
                    # debug = total_transactions <= visualize_initial or total_transactions % visualize_every_n == 0
                    update_graph_capacity(G, path)
                    successful_transactions += 1
            except nx.NetworkXNoPath:
                pass
            total_transactions += 1
            # Adjust the visualization logic
            # if total_transactions <= visualize_initial or total_transactions % visualize_every_n == 0:
            #     visualize_graph(G, total_transactions, pos)

        current_success_rate = successful_transactions / total_transactions
        if prev_success_rate != -1 and abs(current_success_rate - prev_success_rate) < epsilon:
            break
        prev_success_rate = current_success_rate

    return current_success_rate

def visualize_graph(G, transaction_number, pos=None):
    if pos is None:
        pos = nx.spring_layout(G)

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)

    # Draw edge labels for capacities
    edge_labels = nx.get_edge_attributes(G, 'capacity')
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)

    # # Draw curved edges to distinguish between forward and reverse edges
    # for u, v, data in G.edges(data=True):
    #     rad = 0.1  # Radius for curve, adjust as necessary
    #     if G.has_edge(v, u):  # Check for reverse edge
    #         rad = -0.1  # Curve in the opposite direction for reverse edge
    #     nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={rad}', ax=ax)
    ax.set_title(f'Graph after {transaction_number} transactions', fontsize=14)
    plt.title(f'Graph after {transaction_number} transactions', fontsize=14)
    plt.tight_layout()
    plt.show()

def update_graph_capacity(G, path, debug=False, iteration=0):
    if debug:
        print(f"Iteration {iteration}: Updating capacities for path: {path}")
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        # Decrease the capacity of the forward edge
        G[u][v]['capacity'] -= 1
        if debug:
            print(f"Iteration {iteration}: Decreased capacity of edge ({u}, {v}) to {G[u][v]['capacity']}")
        if G[u][v]['capacity'] == 0:
            G.remove_edge(u, v)
            if debug:
                print(f"Iteration {iteration}: Removed edge ({u}, {v}) due to zero capacity")

        # Update or create the reverse edge
        if G.has_edge(v, u):
            G[v][u]['capacity'] += 1
            if debug:
                print(
                    f"Iteration {iteration}: Increased capacity of reverse edge ({v}, {u}) to {G[v][u]['capacity']}")
        else:
            G.add_edge(v, u, capacity=1)
            if debug:
                print(f"Iteration {iteration}: Created reverse edge ({v}, {u}) with capacity 1")
