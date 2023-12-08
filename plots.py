import networkx as nx
from transaction_simulator import simulate_transactions_fees, create_random_graph
import time
import pandas as pd
transaction_amount = 1
# fees = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0]
fees = [0.1]
# fee_range = np.round(np.arange(0.0, 1.1, 0.1), 2)
epsilon = 0.002
num_runs = 1
avg_degree = 10
window_size = 1000
num_nodes = [2, 5, 10, 100]
# # num_nodes = [2, ]
capacity_range = [2, 5, 10, 50, 100]
# transaction_amount = 1
# fee_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
results = {
    'fee': [],
    'success_rate': [],
    'run': [],
    'avg_path_length': [],
    'node': [],
    'capacity':[]
}


for node in num_nodes:
    print(f'started node {node}')
    for fee in fees:
        for capacity in capacity_range:
            for run in range(num_runs):
                G = create_random_graph(node, avg_degree, capacity, 'star')
                # Create a reference copy of the graph
                G_reference = G.copy()
                pos = nx.spring_layout(G)
                # pos = nx.circular_layout(G)
                # pos = nx.circular_layout(G)
                success_rate, avg_path_length = simulate_transactions_fees(G, capacity , node, epsilon, fee, transaction_amount,
                                                                       window_size, pos, visualize=True, visualize_initial = -1, show = False, save = True, G_reference = G_reference, type = 'star' )
                results['fee'].append(fee)
                results['success_rate'].append(success_rate)
                results['run'].append(run)
                results['avg_path_length'].append(avg_path_length)
                results['node'].append(node)
                results['capacity'].append(capacity)

df = pd.DataFrame(results)
