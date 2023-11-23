import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from transaction_simulator import simulate_transactions_fees, create_random_graph

num_nodes = 100
capacity_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20]
window_size = 500
transaction_amount = 1
fee_range = [0.4, 0.45, 0.5, 0.55, 0.6]
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

checkpoint_interval = 100
checkpointing = False
for fee_index, fee in enumerate(fee_range):
    for capacity_index, capacity in enumerate(capacity_range):
        for run in range(num_runs):
            G = create_random_graph(num_nodes, avg_degree, capacity)
            pos = nx.spring_layout(G)
            success_rate = simulate_transactions_fees(G, num_nodes, epsilon, fee, transaction_amount, window_size, pos)
            print(f'Completed run {run}/{num_runs}, capacity {capacity}, fee {fee}')
            # Append each capacity measurement to the list with corresponding fee and capacity

            results['capacity'].append(capacity)
            results['run'].append(run)
            results['success_rate'].append(success_rate)
            results['fee'].append(fee)

            # Checkpointing: save the DataFrame every n iterations
            if checkpointing == True and run % checkpoint_interval == 0:
                checkpoint_df = pd.DataFrame(results)
                checkpoint_filename = f'checkpoint_capacity_{capacity}_fee_{fee}_run_{run}.pkl'
                checkpoint_df.to_pickle(checkpoint_filename)
                print(f'Saved checkpoint to {checkpoint_filename}')


df = pd.DataFrame(results)
# df.to_pickle('df_capacity_1_to_20_fee_0_to_0p5_denser.pkl')

sns.set_theme()


fig = plt.figure(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
sns.lineplot(data=df, x='capacity', y='success_rate', hue='fee', marker='o', ci='sd', legend="full")

plt.xlabel('Edge capacity', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim([0, 1.1])
plt.xlim([0, 21])

plt.tight_layout()
fig.savefig('capacity_near_0p5.png', dpi=300, bbox_inches='tight')
plt.show()



pivot_table = df.pivot_table(values='success_rate', index='fee', columns='capacity', aggfunc='mean')

cmap=sns.cubehelix_palette(as_cmap=True)

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, vmin=0, bar_kws={'label': 'Success Rate'})

# Customizing the plot
plt.title('Success Rate by Fee and Capacity')
plt.xlabel('Edge Capacity')
plt.ylabel('Fee')

# Save and show the figure
plt.savefig('heatmap_capacity_by_fee_near_0p5.png', dpi=300, bbox_inches='tight')
plt.show()

print('------------------')
print('Finished!')