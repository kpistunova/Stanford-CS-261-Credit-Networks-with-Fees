"""
Experiments:
- Vary node count
- Change the randomized amoutn interval (doesn't seem like this will do much, or like if I make it huge that just starts to say more about the transaction amount than the interval )
- Change from uniform distribution to Normal distribution, power law distribution?
- Run for different topologies
"""
from edge_capacity_variation import *

class StandardConfig:
    num_nodes = 100
    capacity_range = np.append(np.arange(1.0, 20, 2.5), 20)
    fee_range = list(np.round(np.arange(0.0, 1.01, 0.05), 2))
    epsilon = 0.002
    num_runs = 10
    avg_degree = 10
    window_size = 500


def run_random_transactions_baseline(num_nodes=StandardConfig.num_nodes):
    """ 'Run an experiment where each transaction amount is always 1, and the fees are always a fixed constant'
    """
    name = f'run_random_transactions_baseline_nodecount=({num_nodes})'
    print(f"Running experiment {name}")

    # Simulation    
    df = simulate_network_capacity_fee_variation(
        num_nodes=num_nodes, 
        capacity_range=StandardConfig.capacity_range, 
        transaction_amount=1, 
        fee_range=StandardConfig.fee_range, 
        epsilon=StandardConfig.epsilon, 
        window_size=StandardConfig.window_size, 
        num_runs=StandardConfig.num_runs, 
        avg_degree=StandardConfig.avg_degree, 
        checkpointing=True
    )

    # Results
    reference_name = name + generate_filename_timestamp_suffix()
    df.to_pickle(f'{reference_name}.pkl')
    plot_results_capacity_fee_variation(df, reference_name)

def run_random_transactions_1_2(num_nodes=StandardConfig.num_nodes):
    """ Run an experiment where each transaction amount is a random value between [1, 2), and the fees are always a fixed constant
    """
    name = f'run_random_transactions_1_2_nodecount=({num_nodes})'
    print(f"Running experiment {name}")
    
    # Simulation    
    df = simulate_network_capacity_fee_variation_random_transaction_amounts(
        num_nodes=num_nodes, 
        capacity_range=StandardConfig.capacity_range, 
        transaction_interval=(1,2), 
        fee_range=StandardConfig.fee_range, 
        epsilon=StandardConfig.epsilon, 
        window_size=StandardConfig.window_size, 
        num_runs=StandardConfig.num_runs, 
        avg_degree=StandardConfig.avg_degree, 
        checkpointing=True
    )

    # Results
    reference_name = name + generate_filename_timestamp_suffix()
    df.to_pickle(f'{reference_name}.pkl')
    plot_results_capacity_fee_variation(df, reference_name)

def run_random_transactions_0_2(num_nodes=StandardConfig.num_nodes):
    """ Run an experiment where each transaction amount is a random value between [0, 2), and the fees are always a fixed constant
    """
    name = f'run_random_transactions_0_2_nodecount=({num_nodes})'
    print(f"Running experiment {name}")
    
    # Simulation    
    df = simulate_network_capacity_fee_variation_random_transaction_amounts(
        num_nodes=num_nodes, 
        capacity_range=StandardConfig.capacity_range, 
        transaction_interval=(0,2), 
        fee_range=StandardConfig.fee_range, 
        epsilon=StandardConfig.epsilon, 
        window_size=StandardConfig.window_size, 
        num_runs=StandardConfig.num_runs, 
        avg_degree=StandardConfig.avg_degree, 
        checkpointing=True
    )

    # Results
    reference_name = name + generate_filename_timestamp_suffix()
    df.to_pickle(f'{reference_name}.pkl')
    plot_results_capacity_fee_variation(df, reference_name)

def run_random_transactions_0_1(num_nodes=StandardConfig.num_nodes):
    """ Run an experiment where each transaction amount is a random value between [0, 1), and the fees are always a fixed constant
    """
    name = f'run_random_transactions_0_1_nodecount=({num_nodes})'
    print(f"Running experiment {name}")
    
    # Simulation    
    df = simulate_network_capacity_fee_variation_random_transaction_amounts(
        num_nodes=num_nodes, 
        capacity_range=StandardConfig.capacity_range, 
        transaction_interval=(0,1), 
        fee_range=StandardConfig.fee_range, 
        epsilon=StandardConfig.epsilon, 
        window_size=StandardConfig.window_size, 
        num_runs=StandardConfig.num_runs, 
        avg_degree=StandardConfig.avg_degree, 
        checkpointing=True
    )

    # Results
    reference_name = name + generate_filename_timestamp_suffix()
    df.to_pickle(f'{reference_name}.pkl')
    plot_results_capacity_fee_variation(df, reference_name)



def run_random_transactions_0_2_normal_distribution(num_nodes=100):
    """ Run an experiment where each transaction amount is a random value between [0, 2), and the fees are always a fixed constant
    """
    name = f'run_random_transactions_0_2_nodecount=({num_nodes})'
    print(f"Running experiment {name}")
    
    # Simulation    
    df = simulate_network_capacity_fee_variation_random_transaction_amounts(
        num_nodes=num_nodes, 
        capacity_range=StandardConfig.capacity_range, 
        transaction_interval=(0,2), 
        fee_range=StandardConfig.fee_range, 
        epsilon=StandardConfig.epsilon, 
        window_size=StandardConfig.window_size, 
        num_runs=StandardConfig.num_runs, 
        avg_degree=StandardConfig.avg_degree, 
        checkpointing=True,
        distribution='normal'
    )

    # Results
    reference_name = name + generate_filename_timestamp_suffix()
    df.to_pickle(f'{reference_name}.pkl')
    plot_results_capacity_fee_variation(df, reference_name)

def run_random_transactions_0_2_lognormal_distribution(num_nodes=100):
    """ Run an experiment where each transaction amount is a random value between [0, 2), and the fees are always a fixed constant
    """
    name = f'run_random_transactions_0_2_nodecount=({num_nodes})'
    print(f"Running experiment {name}")
    
    # Simulation    
    df = simulate_network_capacity_fee_variation_random_transaction_amounts(
        num_nodes=num_nodes, 
        capacity_range=StandardConfig.capacity_range, 
        transaction_interval=(0,2), 
        fee_range=StandardConfig.fee_range, 
        epsilon=StandardConfig.epsilon, 
        window_size=StandardConfig.window_size, 
        num_runs=StandardConfig.num_runs, 
        avg_degree=StandardConfig.avg_degree, 
        checkpointing=True,
        distribution='lognormal'
    )

    # Results
    reference_name = name + generate_filename_timestamp_suffix()
    df.to_pickle(f'{reference_name}.pkl')
    plot_results_capacity_fee_variation(df, reference_name)


def bulk_run_nodecount():
    print("Executing bulk_run_nodecount")
    num_nodes_conditions = [25, 50, 100, 200, 500, 1000, 2000]
    for num_nodes in num_nodes_conditions:
        try:
            run_random_transactions_baseline(num_nodes)
        except Exception as e:
            print(e)
        try:
            run_random_transactions_0_1(num_nodes)
        except Exception as e:
            print(e)
        try:
            run_random_transactions_0_2(num_nodes)
        except Exception as e:
            print(e)
        try:
            run_random_transactions_1_2(num_nodes)
        except Exception as e:
            print(e)
        
        



if __name__ == '__main__':
    print("Hello world!", flush=True)
    # bulk_run_nodecount()
    run_random_transactions_0_2_normal_distribution()
    run_random_transactions_0_2_lognormal_distribution()
    print('------------------', flush=True)
    print('Finished!', flush=True)
