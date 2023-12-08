import networkx as nx
import random
from visualizing import visualize_graph

def create_random_graph(num_nodes, avg_degree, fixed_total_capacity, type = 'random'):
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
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    if type == 'random' :
        num_edges = int(avg_degree * num_nodes)
        while G.number_of_edges() < num_edges :
            u, v = random.sample(G.nodes, 2)  # Select from actual nodes
            # Add the edge only if the reverse edge does not exist
            if not G.has_edge(v, u):
                G.add_edge(u, v, capacity=fixed_total_capacity)
    elif type == 'er':
        # Calculate the probability p for the Erdos-Renyi graph
        p = avg_degree / (num_nodes - 1)

        # Create an Erdős-Rényi graph
        G = nx.erdos_renyi_graph(num_nodes, p, directed=False)

        # Convert to a directed graph and set capacities
        G = G.to_directed()
        for (u, v) in list(G.edges()):
            if random.choice([True, False]):
                # Reverse the direction of the edge
                G.remove_edge(u, v)
                G.add_edge(v, u, capacity=fixed_total_capacity)
            else:
                # Keep the edge direction and add capacity
                G[u][v]['capacity'] = fixed_total_capacity
    elif type == 'ba':
        # Calculate the number of edges each new node forms for the Barabasi-Albert graph
        d = avg_degree // 2

        # Create a Barabási-Albert graph
        G = nx.barabasi_albert_graph(num_nodes, d)

        # Convert to a directed graph and set capacities
        G = G.to_directed()
        for (u, v) in list(G.edges()):
            if random.choice([True, False]):
                # Reverse the direction of the edge
                G.remove_edge(u, v)
                G.add_edge(v, u, capacity=fixed_total_capacity)
            else:
                # Keep the edge direction and add capacity
                G[u][v]['capacity'] = fixed_total_capacity
    elif type == 'line':
        # Add edges to the graph to form a line
        for i in range(num_nodes - 1):
            # Add an edge between node i and node i+1 with the given capacity
            G.add_edge(i, i + 1, capacity=fixed_total_capacity)
            # If you want it to be bidirectional, uncomment the following line
            # G.add_edge(i + 1, i, capacity=fixed_total_capacity)
    elif type == 'cycle':
        # Add edges to the graph to form a cycle
        for i in range(num_nodes):
            G.add_edge(i, (i + 1) % num_nodes, capacity=fixed_total_capacity)
            # Connect to the next node, wrapping around to form a cycle
    elif type == 'complete':
        # Add edges to the graph to form a complete graph
        for u in range(num_nodes):
            for v in range(num_nodes):
                if u != v and not G.has_edge(v, u):
                    G.add_edge(u, v, capacity=fixed_total_capacity)
    elif type == 'star':
        # Add edges to the graph to form a star
        for i in range(1, num_nodes):
            # Connect each node to the central node (node 0)
            G.add_edge(0, i, capacity=fixed_total_capacity)
            # Uncomment the following line to make it bidirectional
            # G.add_edge(i, 0, capacity=fixed_total_capacity)
    else:
        raise ValueError("Invalid graph type. Please choose 'random', 'line', 'cycle', 'complete', or 'star'.")
    for u, v in G.edges():
        G[u][v]['direction'] = 'forward'
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
    # We are rounding because sometimes 0.1 + 0.2 = 0.30000000004 etc
    fees = [round((len(path) - i - 2) * fee, 3) for i in range(len(path) - 1)]
    required_capacities = [transaction_amount + fee for fee in fees]

    # Check if all edges have sufficient capacity first
    for i, (u, v) in enumerate(zip(path, path[1:])):
        if G[u][v]['capacity'] < required_capacities[i]:
            return False  # Transaction failed due to insufficient capacity for at least one edge

    # All edges have sufficient capacity, proceed to update
    for i, (u, v) in enumerate(zip(path, path[1:])):
        # Update capacities
        G[u][v]['capacity'] = round(G[u][v]['capacity'] - required_capacities[i], 3)
        if G[u][v]['capacity'] == 0:
            G.remove_edge(u, v)

        # Update or create the reverse edge
        if G.has_edge(v, u):
            G[v][u]['capacity'] = round(G[v][u]['capacity'] + required_capacities[i], 3)
        else:
            G.add_edge(v, u, capacity=required_capacities[i])

    return True


def simulate_transactions_fees(G, capacity, num_nodes, epsilon, fee, transaction_amount, window_size, pos=None,
                               visualize=False, visualize_initial=0, show = False, save = False, G_reference = None, type = None):
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
      :param capacity:
    """
    total_transactions = 0
    successful_transactions = 0
    prev_success_rate = -1
    total_length_of_paths = 0
    state_frequencies = {}
    if G_reference is None:
        G_reference = G.copy()
    if visualize:
        visualize_graph(G, total_transactions, successful_transactions, fee, capacity, pos, show=show, save=save, G_reference = G_reference, type = type)
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
                        total_length_of_paths += len(path) - 1
                        current_state = tuple(sorted((u, v, round(G[u][v]['capacity'], 2)) for u, v in G.edges()))
                        state_frequencies[current_state] = state_frequencies.get(current_state, 0) + 1
                        # Subtract 1 to get the number of edges
                    else:
                        if visualize and (total_transactions - successful_transactions) <= visualize_initial and show == False:
                            visualize_graph(G, total_transactions, successful_transactions, fee, capacity, pos, show=show, save=save, s=s, t=t, fail = True, G_reference = G_reference, type = type)

            except nx.NetworkXNoPath:
                if visualize and (total_transactions - successful_transactions) <= visualize_initial and show == False:
                    visualize_graph(G, total_transactions, successful_transactions, fee, capacity, pos, show=show, save=save,  s=s, t=t,
                                    no_path=True, G_reference = G_reference, type = type)
                pass

            total_transactions += 1
            if visualize and successful_transactions <= visualize_initial - 3 and successful_transactions > 0:
                visualize_graph(G, total_transactions, successful_transactions, fee, capacity, pos, show=show, save=save, s=s, t=t, G_reference = G_reference, type = type)

        current_success_rate = successful_transactions / total_transactions
        if prev_success_rate != -1 and abs(current_success_rate - prev_success_rate) < epsilon:
            break
        prev_success_rate = current_success_rate
    avg_path_length = total_length_of_paths / successful_transactions if successful_transactions > 0 else 0
    # Normalize frequencies to get probabilities
    total = sum(state_frequencies.values())
    stationary_distribution = {state: freq / total for state, freq in state_frequencies.items()}
    top_states = sorted(stationary_distribution, key=stationary_distribution.get, reverse=True)

    if visualize:
        visualize_graph(G, total_transactions, successful_transactions, fee, capacity, pos, show=show, save=save, G_reference = G_reference, type = type, state_probabilities = stationary_distribution, selected_states=top_states)
    return current_success_rate, avg_path_length, stationary_distribution
