# Credit Network Simulation with Transaction Fees

## Overview
This repository contains Python simulations for studying the dynamics of credit networks with integrated transaction fees. The goal is to understand how various factors such as edge capacities, transaction fees, graph density, and topology influence the success rate of transactions within the network.

## Repository Structure

The simulations are divided across multiple files, each pertaining to different aspects of the network:

- `transaction_simulator.py`: Contains core functions used across various simulations.
- `edge_capacity_variation.py`: Includes functions specific to simulations that vary edge capacity and transaction fees.
- Additional simulation modules (to be named and added): Future simulation studies will be encapsulated in their respective modules. 

## Modeling Transactions

In `transactions_simulator.py`, we have the following key functions:

- `create_random_graph`: Initializes a random directed graph to represent the credit network. Each node represents an entity, and each directed edge represents a credit line with a fixed total capacity and associated fee.

- `update_graph_capacity_fees`: Simulates a transaction along a path within the graph. It updates the capacities of the edges based on the transaction amount and fees, returning a boolean indicating the success or failure of the transaction.

- `simulate_transactions_fees`: Simulates multiple transactions over a credit network to calculate the steady-state success rate, defined as the ratio of successful transactions to the total number attempted once the system reaches equilibrium.

## Transaction Simulation Process

The transaction simulation involves randomly selecting source and sink nodes and attempting to route a fixed transaction amount through the shortest available path. The transaction is deemed successful if each edge along the path can accommodate the transaction amount plus the accrued fees from subsequent nodes. The simulation iterates over transactions in defined windows until the success rate stabilizes within a convergence threshold, signaling a steady state.

## Variant Simulations

In the `edge_capacity_variation.py` file, we have functions like `simulate_network_capacity_fee_variation` and `plot_results_capacity_fee_variation` that run simulations and plot results for scenarios where edge capacities and transaction fees are varied.

## Future Extensions

We plan to extend our simulation to include the variation of other parameters like:

- Average node degree (graph density)
- Total number of nodes
- Specific graph topologies such as complete graphs, balanced trees, and more

These simulations will leverage the graph generators provided by `networkx` to explore the influence of network structure on transaction success rates.

## Running the Simulations

To execute a simulation, open the appropriate Python file and set the desired parameters in the configuration section, then call the appropriate functions to run the simulation and plot the results. For example, to run the scenario where edge capacities and transaction fees are varied, in `edge_capacity_variation.py` file:

```python
# Set the simulation parameters
num_nodes = 100
capacity_range = np.arange(1.0, 20.0, 1)
transaction_amount = 1
fee_range = np.round(np.arange(0.1, 1, 0.01), 2)
epsilon = 0.002
num_runs = 20
avg_degree = 10
window_size = 1000

# Run the simulation
results_df = simulate_network_capacity_fee_variation(num_nodes, capacity_range, transaction_amount, fee_range, epsilon, window_size, num_runs, avg_degree)

# Plot the results
plot_results_capacity_fee_variation(results_df)
```

## Dependencies

To install dependencies, run the following command in your terminal:

```sh
pip install -r requirements.txt
```

Alternatively, if you prefer Poetry:
```sh
poetry install
```
Just keep in mind that `requirements.txt` is the ground source of truth, and our `pyproject.toml` may run out of date.


## Google Cloud

https://askubuntu.com/a/1261782



[//]: # ()
[//]: # ()
[//]: # (## Modeling)

[//]: # ()
[//]: # (Each edge \&#40; &#40;u_i, u_j&#41; \&#41; in the network is characterized by a credit limit \&#40; c_{ij} \&#41; and a transaction fee \&#40; f_{ij} \&#41;. The credit limit is the maximum credit that node \&#40; u_i \&#41; extends to node \&#40; u_j \&#41;, and the transaction fee is the cost charged by node \&#40; u_i \&#41; to route a payment to \&#40; u_j \&#41;.)

[//]: # ()
[//]: # (A transaction path \&#40; p \&#41; from the start node \&#40; u_s \&#41; to the end node \&#40; u_t \&#41; is successful if the payment amount \&#40; p \&#41; is within the credit limits and can cover the transaction fees imposed by the subsequent nodes along the path.)

[//]: # ()
[//]: # (![Four node credit network with fees]&#40;/mnt/data/cr2.png&#41;)

[//]: # ()
[//]: # (*Figure: Four node credit network with fees*)

[//]: # ()
[//]: # (The introduction of fees disrupts the traditional cycle reachability property of credit networks, affecting the network's liquidity and transaction success rates.)

[//]: # ()
[//]: # (### Functions)

[//]: # ()
[//]: # (- `create_random_graph&#40;num_nodes, avg_degree, fixed_total_capacity&#41;`: Initializes the credit network graph with given nodes, average degree, and fixed capacity for each edge.)

[//]: # (- `update_graph_capacity_fees&#40;G, path, transaction_amount, fee&#41;`: Processes a transaction along a path, updates edge capacities, and checks for transaction success.)

[//]: # (- `simulate_transactions_fees&#40;G, num_nodes, epsilon, fee, transaction_amount, window_size&#41;`: Simulates transactions and calculates the steady-state success probability.)

[//]: # (- `plot_results&#40;df&#41;`: Generates plots to visualize the simulation results.)

[//]: # ()
[//]: # (## Simulation Variants)

[//]: # ()
[//]: # (The simulations investigate the following variants:)

[//]: # ()
[//]: # (- `simulate_network_capacity_fee_variation`: Examines how edge capacity and transaction fees affect the transaction success rate.)

[//]: # (- Future simulation plans include varying the average node degree, fees for fixed edge capacity, total number of nodes, and exploring specific graph topologies such as complete graphs, balanced trees, barbell graphs, and more, as provided by the `networkx` package.)

[//]: # ()
[//]: # (## Topology)

[//]: # ()
[//]: # (We utilize various graph generators from `networkx` to study the influence of network topology on transaction success rates. Some of the classic graphs we plan to analyze include:)

[//]: # ()
[//]: # (- Complete graphs &#40;`nx.complete_graph`&#41;)

[//]: # (- Balanced trees &#40;`nx.balanced_tree`&#41;)

[//]: # (- Barbell graphs &#40;`nx.barbell_graph`&#41;)

[//]: # (- Circular ladder graphs &#40;`nx.circular_ladder_graph`&#41;)

[//]: # (- And more...)

[//]: # ()
[//]: # (## Simulation and Results)

[//]: # ()
[//]: # (### Implementation)

[//]: # ()
[//]: # (The simulations are implemented in Python using the `networkx` package. The key functions are:)

[//]: # ()
[//]: # (- `plot_results_capacity_fee_variation`: Plots the results of the capacity and fee variation simulation.)

[//]: # (- `simulate_network_capacity_fee_variation`: Runs the network simulation for varying capacities and fees.)

[//]: # ()
[//]: # (### Configuration)

[//]: # ()
[//]: # (The simulations can be configured with different parameters. Here's an example setup:)

[//]: # ()
[//]: # (```python)

[//]: # (# Example configuration &#40;subject to change&#41;)

[//]: # (num_nodes = 100  # Total number of nodes)

[//]: # (capacity_range = np.arange&#40;1.0, 20.0, 1&#41;  # Range of edge capacities)

[//]: # (transaction_amount = 1  # Fixed transaction amount)

[//]: # (fee_range = np.round&#40;np.arange&#40;0.1, 1, 0.01&#41;, 2&#41;  # Range of transaction fees)

[//]: # (epsilon = 0.002  # Convergence threshold)

[//]: # (num_runs = 20  # Number of simulation runs per network configuration)

[//]: # (avg_degree = 10  # Average node degree &#40;graph density&#41;)

[//]: # (window_size = 1000  # Number of transactions per simulation window)

[//]: # (```)

[//]: # (Note that the configuration will vary based on the specific simulation being conducted.)

[//]: # ()
[//]: # (### Configuration)
