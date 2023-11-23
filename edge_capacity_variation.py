import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transaction_simulator import simulate_transactions_fees, create_random_graph
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
def simulate_network(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing = False, checkpoint_interval = 5):
    print('sim started working')
    results = {
        'capacity': [],
        'run': [],
        'success_rate': [],
        'fee': [],
    }

    for fee in fee_range:
        for capacity in capacity_range:
            for run in range(num_runs):
                G = create_random_graph(num_nodes, avg_degree, capacity)
                pos = nx.spring_layout(G)
                success_rate = simulate_transactions_fees(G, num_nodes, epsilon, fee, transaction_amount, window_size, pos)
                print(f'Completed run {run}/{num_runs}, capacity {capacity}, fee {fee}')

                results['capacity'].append(capacity)
                results['run'].append(run)
                results['success_rate'].append(success_rate)
                results['fee'].append(fee)

                if checkpointing == True and run % checkpoint_interval == 0:
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_filename = f'checkpoint_capacity_fixed_{capacity}_fee_{fee}_run_{run}.pkl'
                    checkpoint_df.to_pickle(checkpoint_filename)
                    print(f'Saved checkpoint to {checkpoint_filename}')

    return pd.DataFrame(results)

def plot_results(df):
    cmap = sns.cubehelix_palette(as_cmap=True)
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
    plt.xlim(left = 0)

    plt.tight_layout()
    fig.savefig('capacity_vs_fees_more_than_2_line_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Heatmap
    pivot_table = df.pivot_table(values='success_rate', index='fee', columns='capacity', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap, vmin=0, cbar_kws={'label': 'Success Rate'}, square=True)

    plt.title('Success Rate by Fee and Capacity')
    plt.xlabel('Edge Capacity')
    plt.ylabel('Fee')
    plt.savefig('heatmap_capacity_vs_fees_more_than_2.png', dpi=300, bbox_inches='tight')
    plt.show()

# Function to identify outliers using IQR
def identify_outliers(df, column,  multiplier=0.8 ):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier  * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Configuration
num_nodes = 100
fee_range = np.linspace(1.0, 20, num=1001)
transaction_amount = 1
# fee_range = [2.2, 2.5, 2.7, 3, 4, 5, 6, 7, 8]
fee_range = np.linspace(0.0, 1.0, num=1001)
epsilon = 0.002
num_runs = 20
avg_degree = 10
window_size = 1000

# Simulation
df = simulate_network(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree, checkpointing=True)
df.to_pickle('capacity_fixed.pkl')


# Plotting
plot_results(df)
mean_success_rates = df.groupby('fee')['success_rate'].mean().reset_index()
top_fees = mean_success_rates.sort_values(by='success_rate', ascending=False).head(7)


sns.set_theme()
fig, ax = plt.subplots(figsize=(8/ 1.2, 6 / 1.2), dpi=300)
lineplot = sns.lineplot(data=df, x='fee', y='success_rate', hue='fee', marker='o', ci='sd',  err_style = 'bars', legend=False, ax=ax)
for line in ax.lines[1:]:
    line.set_alpha(0.7)
plt.xlabel('Fee', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.title('Total capacity = 1, transaction = 1, nodes = 200', fontsize=14)
# plt.legend(title='Fee', title_fontsize='13', fontsize='12', loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim(top=1.01)
plt.ylim(bottom=0)
plt.xlim(left=-0.01)
for _, row in top_fees.iterrows():
    ax.annotate(f"{row['fee']}",  # Change here to match the format of fees
                xy=(row['fee'], row['success_rate']),
                xytext=(25, 5),  # Adjust to position the text to the top right
                textcoords='offset points',
                ha='right',
                va='bottom')
plt.tight_layout()
fig.savefig('capacity_fixed_zoom_in_no_errors', dpi=300, bbox_inches='tight')
plt.show()


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
