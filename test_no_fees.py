import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_random_graph(num_nodes, avg_degree, fixed_total_capacity):
    num_edges = int(avg_degree * num_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))  # Ensure all nodes are added

    while G.number_of_edges() < num_edges * 2:  # Multiplying by 2 since each edge is one-way in DiGraph
        u, v = random.sample(G.nodes, 2)  # Select from actual nodes

        # Add the edge only if the reverse edge does not exist
        if not G.has_edge(v, u):
            G.add_edge(u, v, capacity=fixed_total_capacity)

    return G


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


def update_graph_capacity(G, path, debug=False, iteration=0):
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        # Decrease the capacity of the forward edge
        G[u][v]['capacity'] -= 1

        if G[u][v]['capacity'] == 0:
            G.remove_edge(u, v)
        # Update or create the reverse edge
        if G.has_edge(v, u):
            G[v][u]['capacity'] += 1
        else:
            G.add_edge(v, u, capacity=1)

