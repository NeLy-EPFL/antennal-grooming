"""
This script creates random graphs for bunch of seeds.
Then it finds the motor modules as described and calculates
the adjacency matrices for each random graph.
"""

from pathlib import Path
import itertools
import pickle
import copy

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import Figure4_graph_tools as graph_tools


if __name__ == "__main__":

    DATA_PATH = Path("../../data")
    # dictionary to store all adj matrices
    adj_matrices = []

    # load the grooming connectivity df
    connectivity_grooming = pd.read_pickle(
        DATA_PATH / "Fig4_grooming_network_sensory_merged.pkl"
    )
    motor_module_neurons = pd.read_pickle(DATA_PATH / "Fig4_MN_SN_updated.pkl")

    for random_seed in tqdm(range(0, 100, 1), desc="Making random graphs"):
        # make a random graph
        random_connectivity = graph_tools.make_a_random_graph(
            connectivity_grooming,
            motor_module_neurons["anten_mn"] + motor_module_neurons["neck_mn"],
            seed=random_seed,
        )

        random_graph = nx.from_pandas_edgelist(
            random_connectivity,
            source="pre_root_id",
            target="post_root_id",
            edge_attr=["syn_count", "syn_count_perc", "syn_count_inv"],
            create_using=nx.DiGraph(),
        )

        ########################### MOTOR MODULES ################################
        neck_ups_percentage_random = graph_tools.threshold_connection(
            random_connectivity,
            graph_tools.get_upstream(
                random_connectivity, motor_module_neurons["neck_mn"]
            ),
            motor_module_neurons["neck_mn"],
            threshold=0.05,
        )

        anten_ups_percentage_random = graph_tools.threshold_connection(
            random_connectivity,
            graph_tools.get_upstream(
                random_connectivity, motor_module_neurons["anten_mn"]
            ),
            motor_module_neurons["anten_mn"],
            threshold=0.05,
        )
        # Leg prem (predefined)
        leg_prem_module = motor_module_neurons["leg_prem"]

        leg_anten_prem = set(leg_prem_module).intersection(
            set(anten_ups_percentage_random.keys())
        )
        leg_neck_prem = set(leg_prem_module).intersection(
            set(neck_ups_percentage_random.keys())
        )
        neck_anten_prem = set(neck_ups_percentage_random.keys()).intersection(
            set(anten_ups_percentage_random.keys())
        )
        shared_prem_random = list(
            leg_anten_prem.union(leg_neck_prem).union(neck_anten_prem)
        )
        # neck module
        neck_prem_random = list(
            set(neck_ups_percentage_random.keys()).difference(shared_prem_random)
        )
        neck_module_random = motor_module_neurons["neck_mn"] + neck_prem_random
        # anten module
        anten_prem_random = list(
            set(anten_ups_percentage_random.keys()).difference(shared_prem_random)
        )
        anten_module_random = motor_module_neurons["anten_mn"] + anten_prem_random

        ########################### CENTRAL NEURONS ################################
        source_neurons = motor_module_neurons["jo_f"]
        all_prem_neurons = set(
            neck_prem_random + anten_prem_random + leg_prem_module + shared_prem_random
        )

        cutoff = 4  # based on the layer algorithm max depth
        threshold = 0.05

        graph_random = nx.from_pandas_edgelist(
            random_connectivity,
            source="pre_root_id",
            target="post_root_id",
            edge_attr=["syn_count", "syn_count_perc", "syn_count_inv"],
            create_using=nx.DiGraph(),
        )

        jof2prem_simple_paths = graph_tools.get_all_simple_paths(
            graph_random,
            source_neurons,
            list(all_prem_neurons)
            + motor_module_neurons["neck_mn"]
            + motor_module_neurons["anten_mn"],
            cutoff=cutoff,
        )
        jof2prem_simple_paths_pruned = graph_tools.threshold_prune_paths(
            graph_random,
            jof2prem_simple_paths,
            threshold=threshold,
            edge_property="syn_count_perc",
            exclude_nodes=motor_module_neurons["jo_e"]
            + motor_module_neurons["jo_c"]
            + motor_module_neurons["bm_ant"],
        )

        central_neurons_random = set(
            itertools.chain.from_iterable(jof2prem_simple_paths_pruned)
        ).difference(
            set(
                motor_module_neurons["jo_f"]
                + list(all_prem_neurons)
                + motor_module_neurons["neck_mn"]
                + motor_module_neurons["anten_mn"]
            )
        )

        ########################### ADJACENCY ################################
        neuron_groups_random = {
            "JO-F": list(source_neurons),
            "CENT.": list(central_neurons_random),
            "ANTEN PREM": list(anten_prem_random),
            "NECK PREM": list(neck_prem_random),
            "LEG PREM": list(leg_prem_module),
            "SHARED PREM": list(shared_prem_random),
            "ANTEN MN": list(motor_module_neurons["anten_mn"]),
            "NECK MN": list(motor_module_neurons["neck_mn"]),
        }

        adj_matrix_random = graph_tools.get_adj_matrix_from_conn(
            random_connectivity,
            neuron_groups_random,
        )

        adj_matrices.append(copy.deepcopy(adj_matrix_random))

    # save the adj matrices
    np.save(DATA_PATH / "EDFig9_adj_matrices_random.npy", adj_matrices)
    print(f'Random graphs are saved at {DATA_PATH}')
