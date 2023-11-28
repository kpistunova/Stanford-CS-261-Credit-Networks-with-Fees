import cugraph
import cudf
import random
import networkx as nx
from cugraph import *
from cugraph.utilities import utils
def create_random_graph_gpu(num_nodes, avg_degree, fixed_total_capacity):
    edges = int(avg_degree * num_nodes)
    src = [random.randint(0, num_nodes - 1) for _ in range(edges)]
    dst = [random.randint(0, num_nodes - 1) for _ in range(edges)]
    capacity = [fixed_total_capacity for _ in range(edges)]

    # Create a DataFrame and remove self-loops and duplicate edges
    df = cudf.DataFrame({'src': src, 'dst': dst, 'capacity': capacity})
    df = df[df['src'] != df['dst']]  # Remove self-loops
    df = df.drop_duplicates()        # Remove duplicates

    G = cugraph.Graph(directed = True)
    G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='capacity')

    return G

def convert_to_networkx(G):
    # Convert to a cudf DataFrame
    df = G.view_edge_list()
    # Convert the cudf DataFrame to a pandas DataFrame
    pdf = df.to_pandas()
    # Create a networkx graph from the pandas DataFrame
    nx_graph = nx.from_pandas_edgelist(pdf, source='src', target='dst', edge_attr='capacity')
    return nx_graph

def update_graph_capacity_fees_gpu(G, path, transaction_amount, fee):
    fees = [(len(path) - i - 2) * fee for i in range(len(path) - 1)]
    df = G.view_edge_list()

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        required_capacity = transaction_amount + fees[i]

        if df.query(f"src == {u} and dst == {v}")['capacity'].iloc[0] < required_capacity:
            return False  # Transaction failed due to insufficient capacity

        mask = (df['src'] == u) & (df['dst'] == v)
        # Perform the subtraction
        new_capacity = df.loc[mask, 'capacity'] - required_capacity

        # Assign the new capacity back to the DataFrame
        df.loc[mask, 'capacity'] = new_capacity.values

        # Check for reverse edge and update
        reverse_mask = (df['src'] == v) & (df['dst'] == u)
        reverse_edge = df[reverse_mask]
        if not reverse_edge.empty:
            new_capacity_reverse = df.loc[reverse_mask, 'capacity'] + required_capacity
            df.loc[reverse_mask, 'capacity'] = new_capacity_reverse.values
        else:
            new_row = cudf.DataFrame({'src': [v], 'dst': [u], 'capacity': [required_capacity]})
            df = cudf.concat([df, new_row], ignore_index=True)

    return True


def simulate_transactions_fees_gpu(G, num_nodes, epsilon, fee, transaction_amount, window_size):
    total_transactions = 0
    successful_transactions = 0
    prev_success_rate = -1

    while True:
        for _ in range(window_size):
            s, t = random.sample(range(num_nodes), 2)
            df = cugraph.sssp(G, s)
            try:
                path = cugraph.utilities.utils.get_traversed_path_list(df, t)
                path.reverse()
                # path = cugraph.shortest_path(G, s, t)
                if path is not None and len(path) > 0:
                    if all(df.query(f"src == {u} and dst == {v}")['capacity'].iloc[0] > 0 for u, v in zip(path, path[1:])):
                        transaction_succeeded = update_graph_capacity_fees_gpu(G, path, transaction_amount, fee)
                        if transaction_succeeded:
                            successful_transactions += 1
            except Exception as e:
                pass

            total_transactions += 1

        current_success_rate = successful_transactions / total_transactions
        if prev_success_rate != -1 and abs(current_success_rate - prev_success_rate) < epsilon:
            break
        prev_success_rate = current_success_rate

    return current_success_rate
