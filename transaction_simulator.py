import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nx_cugraph as nxcg
import cugraph
import cudf
def create_random_graph(num_nodes, avg_degree, fixed_total_capacity):
    """
    Creates a random directed graph with a specified average degree and fixed total capacity for each edge.

    Parameters:
    num_nodes (int): The number of nodes in the graph.
    avg_degree (float): The average out-degree that each node in the graph should have.
    fixed_total_capacity (int or float): The capacity assigned to each edge in the graph.

    Returns:
    networkx.DiGraph: A NetworkX directed graph with the specified number of nodes and edges, where each edge
                      has the given fixed total capacity.

    Note:
    - The function attempts to create a graph where each node has approximately the average degree specified.
      However, the actual degree may vary due to the random nature of graph generation.
    - Edges are added randomly, and the graph may not be strongly connected.
    """
    num_edges = int(avg_degree * num_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))  # Ensure all nodes are added

    while G.number_of_edges() < num_edges :
        u, v = random.sample(G.nodes, 2)  # Select from actual nodes

        # Add the edge only if the reverse edge does not exist
        if not G.has_edge(v, u):
            G.add_edge(u, v, capacity=fixed_total_capacity)

    return G

def update_graph_capacity_fees(G, path, transaction_amount, fee):
    """
    Updates the capacities and fees of edges along a given path in a graph for a specified transaction.

    Parameters:
    G (networkx.DiGraph): The graph on which the transaction is occurring.
    path (list): A list of node indices representing the path through which the transaction is routed.
    transaction_amount (int or float): The amount of the transaction.
    fee (int or float): The fee charged per edge traversed in the transaction.

    Returns:
    bool: True if the transaction is successful (i.e., all edges in the path have sufficient capacity),
          False otherwise.

    Note:
    - The function deducts the transaction amount and the cumulative fees from the capacity of each edge in the path.
    - If any edge along the path does not have enough capacity to handle the transaction amount and the fees,
      the transaction fails, and no changes are made to the graph.
    - If an edge capacity drops to zero after the transaction, the edge is removed from the graph.
    - For each edge in the path, if the reverse edge exists, its capacity is increased by the transaction amount
      and fees; otherwise, a new reverse edge is created with that capacity.
    """
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
    """
    Simulates a series of transactions in a credit network, represented as a directed graph, and computes the
    success rate of these transactions. The success rate is the ratio of successful transactions to the total number
    of attempted transactions, calculated once the system reaches a steady state.

    Parameters:
    G (networkx.DiGraph): The graph representing the credit network, where nodes are entities and edges represent
                          credit lines with a fixed total capacity.
    num_nodes (int): The total number of nodes (entities) in the graph.
    epsilon (float): The convergence threshold used to determine if the system has reached a steady state.
                     A steady state is reached when the success rate changes by less than epsilon between two
                     consecutive windows of transactions.
    fee (int or float): The fee charged per edge traversed in each transaction.
    transaction_amount (int or float): The fixed amount involved in each transaction (typically one unit).
    window_size (int): The number of transactions processed in each iteration.
    pos (dict, optional): Node positions for visualization purposes (not used in the simulation logic). Defaults to None.
    snapshot_interval (int): The interval at which to take snapshots of the simulation (not utilized in the current implementation).

    Returns:
    current_success_rate (float): The success rate of transactions at steady state, defined as the ratio of successful transactions to
           the total number of attempted transactions.

    Process:
    - Transactions are simulated by selecting a source (s) and a sink (t) node at random.
    - For each transaction, the shortest path from s to t is computed. If no path exists, the transaction is marked as failed.
    - For transactions with an available path, the function checks if each edge along the path can support the transaction
      amount plus the necessary fees. The fee for an edge depends on its position in the path and the total number of edges (L).
    - The transaction is successful if all edges in the path have sufficient capacity; otherwise, it is marked as failed.
    - After a successful transaction, the capacities of the edges along the path are updated to reflect the transaction and fees.
    - The simulation runs iteratively, processing transactions in windows of 'window_size' until the success rate stabilizes within
      the epsilon threshold, indicating a steady state.
    - At steady state, the function returns the overall success probability, calculated as the ratio of successful transactions to
      the total number of iterations.
    """
    total_transactions = 0
    successful_transactions = 0
    prev_success_rate = -1

    while True:
        for _ in range(window_size):
            s, t = random.sample(range(num_nodes), 2)
            try:
                # nxcg_G = nxcg.from_networkx(G)
                path = nx.shortest_path(G, s, t)
                # path = cugraph.shortest_path(nxcg_G, s, t)
                # sssp = cugraph.bfs(G, s)
                # path = cugraph.utilities.utils.get_traversed_path_list(sssp, t)
                # path.reverse()
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


