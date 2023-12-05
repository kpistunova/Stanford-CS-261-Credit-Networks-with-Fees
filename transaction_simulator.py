import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import nx_cugraph as nxcg # Kate, I had to temporarily comment this out becuase I don't have GPUs -Russell
# import cugraph # Kate, I had to temporarily comment this out becuase I don't have GPUs -Russell
# import cudf # Kate, I had to temporarily comment this out becuase I don't have GPUs -Russell
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
            G.add_edge(i, (i + 1) % num_nodes, capacity=fixed_total_capacity)  # Connect to the next node, wrapping around to form a cycle

    else:
        raise ValueError("Invalid graph type. Please choose 'random', 'line', or 'cycle'.")

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

def update_graph_capacity_fees_percentage_fees(G, path, transaction_amount, percentage_fee):
    """
    Updates the capacities and fees of edges along a given path in a graph for a specified transaction.

    Parameters:
    G (networkx.DiGraph): The graph on which the transaction is occurring.
    path (list): A list of node indices representing the path through which the transaction is routed.
    transaction_amount (int or float): The amount of the transaction.
    percentage_fee (float): The PERCENT fee charged per edge traversed in the transaction. NOTE THIS IS DIFFERENT FROM THE ORIGINAL update_graph_capacity_fees function

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
    num_edges_in_path = len(path) - 1

    def calculate_required_capacities():
        """ There are two ways to deal with percentage fees. This choice is the one whereby each intermediate node collects its fee via surcharge rather than by taking a cut of the payload directly.
        """
        required_capacities = [transaction_amount] # initialize with the actual endgoal payment amount (the last required capacity in the path)
        for _ in range(num_edges_in_path - 1): # minus 1 because we already put in the last required capacity
            previous_required_capacity = required_capacities[0] * (1 + percentage_fee)
            required_capacities.insert(0, previous_required_capacity)
        return required_capacities

    required_capacities = calculate_required_capacities()

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
                               visualize=False, visualize_initial=0, visualize_every_n=1000):
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
    if visualize:
        visualize_graph(G, total_transactions, fee, capacity, pos)
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
                        if visualize and successful_transactions <= visualize_initial:
                            visualize_graph(G, total_transactions, fee, capacity, pos, s=s, t=t)
                        # Subtract 1 to get the number of edges


            except nx.NetworkXNoPath:
                pass

            total_transactions += 1

        current_success_rate = successful_transactions / total_transactions
        if prev_success_rate != -1 and abs(current_success_rate - prev_success_rate) < epsilon:
            break
        prev_success_rate = current_success_rate
    avg_path_length = total_length_of_paths / successful_transactions if successful_transactions > 0 else 0
    if visualize:
        visualize_graph(G, total_transactions, fee, capacity, pos, final=True)
    return current_success_rate, avg_path_length

def simulate_transactions_fees_random_transaction_amounts(G, capacity, num_nodes, epsilon, fee, transaction_interval, window_size, pos=None,
                               visualize=False, visualize_initial=0, visualize_every_n=1000):
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
    transaction_interval (tuple of float): The random interval for random transaction amounts. NOTE THIS IS DIFFERENT FROM THE ORIGINAL simulate_transactions_fees
    fee (int or float): The fee charged per edge traversed in each transaction.
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
    if visualize:
        visualize_graph(G, total_transactions, fee, capacity, pos)
    while True:
        for _ in range(window_size):
            # Select a source and sink at random
            s, t = random.sample(range(num_nodes), 2)
            # Select a random transaction amount between the transaction interval
            transaction_amount = random.uniform(*transaction_interval)
            try:
                path = nx.shortest_path(G, s, t)
                # Direct capacity check
                if min([G[u][v]['capacity'] for u, v in zip(path, path[1:])]) > 0:
                    transaction_succeeded = update_graph_capacity_fees(G, path, transaction_amount, fee) 
                    if transaction_succeeded:
                        successful_transactions += 1
                        total_length_of_paths += len(path) - 1
                        if visualize and successful_transactions <= visualize_initial:
                            visualize_graph(G, total_transactions, fee, capacity, pos, s=s, t=t)
                        # Subtract 1 to get the number of edges

            except nx.NetworkXNoPath:
                pass

            total_transactions += 1

        current_success_rate = successful_transactions / total_transactions
        if prev_success_rate != -1 and abs(current_success_rate - prev_success_rate) < epsilon:
            break
        prev_success_rate = current_success_rate
    avg_path_length = total_length_of_paths / successful_transactions if successful_transactions > 0 else 0
    if visualize:
        visualize_graph(G, total_transactions, fee, capacity, pos, final=True)
    return current_success_rate, avg_path_length

def simulate_transactions_fees_random_transaction_amounts_percentage_fees(G, capacity, num_nodes, epsilon, percentage_fee, transaction_interval, window_size, pos=None,
                               visualize=False, visualize_initial=0, visualize_every_n=1000):
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
    transaction_interval (tuple of float): The random interval for random transaction amounts. NOTE THIS IS DIFFERENT FROM THE ORIGINAL simulate_transactions_fees
    percentage_fee (float): The PERCENTAGE fee charged per edge traversed in each transaction. NOTE THIS IS DIFFERENT FROM THE ORIGINAL simulate_transactions_fees
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
    if visualize:
        visualize_graph(G, total_transactions, percentage_fee, capacity, pos)
    while True:
        for _ in range(window_size):
            # Select a source and sink at random
            s, t = random.sample(range(num_nodes), 2)
            # Select a random transaction amount between the transaction interval
            transaction_amount = random.uniform(*transaction_interval)
            try:
                path = nx.shortest_path(G, s, t)
                # Direct capacity check
                if min([G[u][v]['capacity'] for u, v in zip(path, path[1:])]) > 0:
                    transaction_succeeded = update_graph_capacity_fees_percentage_fees(G, path, transaction_amount, percentage_fee)
                    if transaction_succeeded:
                        successful_transactions += 1
                        total_length_of_paths += len(path) - 1
                        if visualize and successful_transactions <= visualize_initial:
                            visualize_graph(G, total_transactions, percentage_fee, capacity, pos, s=s, t=t)
                        # Subtract 1 to get the number of edges

            except nx.NetworkXNoPath:
                pass

            total_transactions += 1

        current_success_rate = successful_transactions / total_transactions
        if prev_success_rate != -1 and abs(current_success_rate - prev_success_rate) < epsilon:
            break
        prev_success_rate = current_success_rate
    avg_path_length = total_length_of_paths / successful_transactions if successful_transactions > 0 else 0
    if visualize:
        visualize_graph(G, total_transactions, fee, capacity, pos, final=True)
    return current_success_rate, avg_path_length

def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

def visualize_graph(G, transaction_number, fee, capacity, pos=None, final=False, s=None, t=None):
    if pos is None:
        pos = nx.spring_layout(G)

    fig, ax = plt.subplots(figsize=(8 / 1.2, 6 / 1.2), dpi=300)
    M = G.number_of_edges()

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightskyblue', edgecolors='black')

    if s is not None and t is not None:
        # Draw the source and target nodes in different colors
        nx.draw_networkx_nodes(G, pos, nodelist=[s], ax=ax, node_color='palegreen', edgecolors='black')
        nx.draw_networkx_nodes(G, pos, nodelist=[t], ax=ax, node_color='lightcoral', edgecolors='black')

    nx.draw_networkx_labels(G, pos, ax=ax)
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')

    edge_weights = nx.get_edge_attributes(G, 'capacity')
    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)
    if final:
        ax.set_title(f'Graph at steady state, after {transaction_number} transactions, f = {fee}, c = {capacity}', fontsize=14)
        plt.title(f'Graph at steady state, after {transaction_number} transactions, f = {fee}, c = {capacity}', fontsize=14)
    else:
        ax.set_title(f'Graph after {transaction_number} transactions, f = {fee}, c = {capacity}', fontsize=14)
        plt.title(f'Graph after {transaction_number} transactions, f = {fee}, c = {capacity}', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    #
    num_nodes = [5]
    capacity_range = 5
    transaction_amount = 1
    fee = 0.1
    # fee_range = np.round(np.arange(0.0, 1.1, 0.1), 2)
    epsilon = 0.002
    num_runs = 3
    avg_degree = 10
    window_size = 1000
    # num_nodes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # # num_nodes = [2, ]
    # capacity_range = [2, 3, 4, 5, 8, 10, 15, 20, 30]
    # transaction_amount = 1
    # fee_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    results = {
        'node': [],
        'success_rate': [],
        'run': []
    }
    for node in num_nodes:
        print(f'started node {node}')
        for run in range(num_runs):
            G = create_random_graph(node, avg_degree, capacity_range, 'line')
            pos = nx.spring_layout(G)
            # pos = nx.circular_layout(G)
            success_rate, avg_path_length = simulate_transactions_fees(G, capacity_range , node, epsilon, fee, transaction_amount,
                                                                window_size, pos, visualize=True, visualize_initial = 5)
            results['node'].append(node)
            results['success_rate'].append(success_rate)
            results['run'].append(run)

    result=pd.DataFrame(results)
    sns.set_theme()  # Apply the default theme
    plt.figure(figsize=(10, 6))
    plt.ylim([0.0, 1.1])
    sns.lineplot(x='node', y='success_rate', data=result, marker ='o')  # Creates a scatter plot
    plt.show()

    print(f'success rate is {success_rate}')
    print(f'Average path is {avg_path_length}')

