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


def simulate_transactions_fees(G, num_nodes, epsilon, fee, transaction_amount, window_size, pos=None, snapshot_interval=100):
    total_transactions = 0
    successful_transactions = 0
    prev_success_rate = -1

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

#
#
#
# def simulate_transactions(G, num_nodes, epsilon, pos, visualize_initial=4, visualize_every_n=1000):
#     total_transactions = 0
#     successful_transactions = 0
#     window_size = 1000
#     prev_success_rate = -1
#
#     while True:
#         for _ in range(window_size):
#             s, t = random.sample(range(num_nodes), 2)
#             try:
#                 path = nx.shortest_path(G, s, t, weight='capacity')
#                 if min([G[u][v]['capacity'] for u, v in zip(path, path[1:])]) > 0:
#                     # debug = total_transactions <= visualize_initial or total_transactions % visualize_every_n == 0
#                     update_graph_capacity(G, path)
#                     successful_transactions += 1
#             except nx.NetworkXNoPath:
#                 pass
#             total_transactions += 1
#             # Adjust the visualization logic
#             # if total_transactions <= visualize_initial or total_transactions % visualize_every_n == 0:
#             #     visualize_graph(G, total_transactions, pos)
#
#         current_success_rate = successful_transactions / total_transactions
#         if prev_success_rate != -1 and abs(current_success_rate - prev_success_rate) < epsilon:
#             break
#         prev_success_rate = current_success_rate
#
#     return current_success_rate
#
# def visualize_graph(G, transaction_number, pos=None):
#     if pos is None:
#         pos = nx.spring_layout(G)
#
#     fig, ax = plt.subplots(figsize=(12, 10))
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
#
#     # Draw edge labels for capacities
#     edge_labels = nx.get_edge_attributes(G, 'capacity')
#     nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)
#
#     # # Draw curved edges to distinguish between forward and reverse edges
#     # for u, v, data in G.edges(data=True):
#     #     rad = 0.1  # Radius for curve, adjust as necessary
#     #     if G.has_edge(v, u):  # Check for reverse edge
#     #         rad = -0.1  # Curve in the opposite direction for reverse edge
#     #     nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={rad}', ax=ax)
#     ax.set_title(f'Graph after {transaction_number} transactions', fontsize=14)
#     plt.title(f'Graph after {transaction_number} transactions', fontsize=14)
#     plt.tight_layout()
#     plt.show()
#
# def update_graph_capacity(G, path, debug=False, iteration=0):
#     if debug:
#         print(f"Iteration {iteration}: Updating capacities for path: {path}")
#     for i in range(len(path) - 1):
#         u, v = path[i], path[i + 1]
#         # Decrease the capacity of the forward edge
#         G[u][v]['capacity'] -= 1
#         if debug:
#             print(f"Iteration {iteration}: Decreased capacity of edge ({u}, {v}) to {G[u][v]['capacity']}")
#         if G[u][v]['capacity'] == 0:
#             G.remove_edge(u, v)
#             if debug:
#                 print(f"Iteration {iteration}: Removed edge ({u}, {v}) due to zero capacity")
#
#         # Update or create the reverse edge
#         if G.has_edge(v, u):
#             G[v][u]['capacity'] += 1
#             if debug:
#                 print(
#                     f"Iteration {iteration}: Increased capacity of reverse edge ({v}, {u}) to {G[v][u]['capacity']}")
#         else:
#             G.add_edge(v, u, capacity=1)
#             if debug:
#                 print(f"Iteration {iteration}: Created reverse edge ({v}, {u}) with capacity 1")
