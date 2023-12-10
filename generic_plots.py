import networkx as nx
from transaction_simulator import simulate_transactions_fees, create_random_graph
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import os
from matplotlib.lines import Line2D
import time
import numpy as np
current_date = datetime.now().strftime('%Y-%m-%d')
base_directory = 'data'
new_directory_path = os.path.join(base_directory, current_date)
# file_path = os.path.join(new_directory_path, filename)
# Check if the directory does not exist
if not os.path.exists(new_directory_path):
    # If it doesn't exist, create a new directory
    os.makedirs(new_directory_path)
name = 'random_er_graph_node_vs_fee_all_capacity_after_fix_correct.pkl'
graph_type = 'Random'
df = pd.read_pickle(name)
unique_capacities = np.sort(df['capacity'].unique())
unique_nodes = np.sort(df['nodes'].unique())

#------fixed---fees--dif---capacity

for fee in df['fee'].unique():
    if fee != 0.0:  # Skip fee of 0.0 as we always include it
        # Select 5 evenly spaced indices from the sorted unique capacities
        selected_indices = np.linspace(0, len(unique_capacities) - 1, 5, dtype=int)
        selected_capacities = unique_capacities[selected_indices]
        selected_fees = [fee, 0.0]
        #x is node number
        df_filtered = df[(df['fee'].isin([fee, 0.0])) & (df['capacity'].isin(selected_capacities))]
        bg_color = plt.gcf().get_facecolor()
        palette = sns.color_palette('coolwarm', n_colors=len(df_filtered['capacity'].unique()))

        sns.set_theme()
        fig, ax = plt.subplots(figsize=(8 , 6 / 1.2), dpi=300)

        # Use 'fee' as the style variable
        sns.lineplot(data=df_filtered, x='nodes', y='avg_path_length', hue='capacity', style='fee', dashes=True, markers = True,
                     palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(), ci='sd', alpha = 0.8)

        title = graph_type + f' graph, fee = {fee}'
        plt.title(title, fontsize=16)
        plt.xlabel('Node Number', fontsize=16)
        plt.ylabel('Average Path Length', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        filename = graph_type + '_graph' + f'_average_path_length_vs_node_number_fee_{fee}.png'
        file_path = os.path.join(new_directory_path, filename)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig.savefig((file_path), dpi=300)
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6 / 1.2), dpi=300)

        # Use 'fee' as the style variable
        sns.lineplot(data=df_filtered, x='nodes', y='success_rate', hue='capacity', style='fee', dashes=True, markers = True,
                     palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(), ci='sd', alpha = 0.8)

        title = graph_type + f' graph, fee = {fee}'
        plt.title(title, fontsize=16)
        plt.xlabel('Node Number', fontsize=16)
        plt.ylabel('Success Rate', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        filename = graph_type + '_graph' + f'_success_rate_vs_node_number_fee_{fee}.png'
        file_path = os.path.join(new_directory_path, filename)
        plt.tight_layout()
        fig.savefig((file_path), dpi=300)
        plt.show()
        plt.close(fig)

for fee in df['fee'].unique():
    if fee != 0.0:
        selected_indices = np.linspace(0, len(unique_nodes) - 1, 5, dtype=int)
        selected_nodes = unique_nodes[selected_indices]
        selected_fees = [fee, 0.0]
        #x is node number
        df_filtered = df[(df['fee'].isin([fee, 0.0])) & (df['nodes'].isin(selected_nodes))]
        # x is capacity
        fig, ax = plt.subplots(figsize=(8 , 6 / 1.2), dpi=300)

        # Use 'fee' as the style variable
        sns.lineplot(data=df_filtered, x='capacity', y='avg_path_length', hue='nodes', style='fee', dashes=True, markers = True,
                     palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(), ci='sd', alpha = 0.8, legend = 'full')
        title = graph_type + f' graph, fee = {fee}'
        plt.title(title, fontsize=16)
        plt.xlabel('Capacity', fontsize=16)
        plt.ylabel('Average Path Length', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        filename = graph_type + '_graph' + f'_average_path_length_vs_capacity_fee_{fee}.png'
        file_path = os.path.join(new_directory_path, filename)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig.savefig((file_path), dpi=300)
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6 / 1.2), dpi=300)

        # Use 'fee' as the style variable
        sns.lineplot(data=df_filtered, x='capacity', y='success_rate', hue='nodes', style='fee', dashes=True, markers = True,
                     palette='coolwarm', hue_norm=matplotlib.colors.LogNorm(), ci='sd', alpha = 0.8, legend = 'full')
        title = graph_type + f' graph, fee = {fee}'
        plt.title(title, fontsize=16)
        plt.xlabel('Capacity', fontsize=16)
        plt.ylabel('Success Rate', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        filename = graph_type + '_graph' + f'_success_rate_vs_capacity_fee_{fee}.png'
        file_path = os.path.join(new_directory_path, filename)
        plt.tight_layout()
        fig.savefig((file_path), dpi=300)
        plt.show()
        plt.close(fig)

print('terue')