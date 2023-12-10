import networkx as nx
from transaction_simulator import simulate_transactions_fees, create_random_graph
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

import numpy as np
current_date = datetime.now().strftime('%Y-%m-%d')
base_directory = 'data'
new_directory_path = os.path.join(base_directory, current_date)
# Check if the directory does not exist
if not os.path.exists(new_directory_path):
    # If it doesn't exist, create a new directory
    os.makedirs(new_directory_path)
transaction_amount = 1
# fees = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0]
fees = [0.0, 0.1]
# fee_range = np.round(np.arange(0.0, 1.1, 0.1), 2)
epsilon = 0.002
num_runs = 5
avg_degree = 10
window_size = 1000
num_nodes = [2, 5, 10, 50, 100]
# # num_nodes = [2, ]
capacity_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 50, 100, 500]
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
    'capacity':[]
}
filename = 'star_graph_tests3.pkl'
file_path = os.path.join(new_directory_path, filename)
# df = pd.read_pickle(file_path)
total_execution_time = 0
for run in range(num_runs):
    print(f'started run {run}')
    start_time = time.time()
    for node in num_nodes:
        for fee in fees:
            for capacity in capacity_range:
                G = create_random_graph(node, avg_degree, capacity, 'star')
                # Create a reference copy of the graph
                G_reference = G.copy()

                results['fee'].append(fee)
                results['success_rate'].append(success_rate)
                results['run'].append(run)
                results['avg_path_length'].append(avg_path_length)
                results['node'].append(node)
                results['capacity'].append(capacity)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time
    remaining_fees = num_runs - (run + 1)
    estimated_remaining_time = remaining_fees * (total_execution_time / (run + 1))
    print(f"Processed fee {run} in time {execution_time} seconds")
    print(f"Estimated remaining time: {estimated_remaining_time / 60} minutes\n")
df = pd.DataFrame(results)

df.to_pickle(file_path)

# df['scale'] = 1 - (1/df['capacity'])
df['scale'] = df['capacity']/(df['capacity']+1)

sns.set_theme()
fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
sns.lineplot(data=df, x='scale', y='success_rate', hue='node', style = 'fee', marker='o', alpha=0.9, ci='sd', legend='full')

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