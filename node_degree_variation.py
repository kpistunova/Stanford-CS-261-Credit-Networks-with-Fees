import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from transaction_simulator import simulate_transactions_fees, create_random_graph


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