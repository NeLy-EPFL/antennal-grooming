""" Graph tools. """

import copy
import itertools

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import neuprint
from neuprint import Client

import Figure4_neurons as neurons

MY_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InBnaXplbW96ZGlsQGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jTENyNEoyQWozazFqTjFGTzQtcmpIVnhZU2xGV3Y5NHptSUtndDVicVRuPXM5Ni1jP3N6PTUwP3N6PTUwIiwiZXhwIjoxODkyMTQ5OTA0fQ.1jPbQ350OHMCitI6zQy5SQIBtc6A2hdY5ErwYwqpn-A'

def signal_flow(A):
    """
    Implementation of the signal flow metric from Varshney et al 2011
    Source: A Connectome of an insect brain, Winding, Pedigo et al 2023
    """
    A = A.copy()
    # A = remove_loops(A)
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)
    L_pinv = np.linalg.pinv(L)
    z = L_pinv @ b

    return z


def get_all_shortest_paths(
    graph, source_nodes, target_nodes, edge_property="syn_count_inv"
):
    """Get all simple paths between source and target nodes"""
    all_paths = []
    for source_n, target_n in itertools.product(source_nodes, target_nodes):
        if not source_n in graph.nodes or not target_n in graph.nodes:
            continue

        if nx.has_path(graph, source_n, target_n) and source_n != target_n:
            # If there is a direct edge, remove it and find the shortest path
            edge_list = []
            while graph.has_edge(source_n, target_n):
                edge_data = graph.get_edge_data(
                    source_n, target_n
                )  # Store the edge data if needed
                edge_list.append((source_n, target_n, edge_data))
                graph.remove_edge(source_n, target_n)

            try:
                path = nx.shortest_path(
                    graph, source=source_n, target=target_n, weight=edge_property
                )
            except nx.NetworkXNoPath:
                path = []

            # Add the edge back to the graph
            for edge in edge_list:
                source_n, target_n, edge_data = edge
                graph.add_edge(source_n, target_n, **edge_data)

            all_paths.append(path)

    return all_paths


def get_all_simple_paths(graph, source_nodes, target_nodes, cutoff=4):
    """Get all simple paths between source and target nodes"""
    all_paths = []
    for source_n, target_n in itertools.product(source_nodes, target_nodes):
        if not source_n in graph.nodes or not target_n in graph.nodes:
            continue
        if nx.has_path(graph, source_n, target_n):
            all_paths.extend(
                [
                    p
                    for p in nx.all_simple_paths(
                        graph, source_n, target_n, cutoff=cutoff
                    )
                ]
            )
    return all_paths


def threshold_prune_paths(
    graph,
    paths,
    threshold=0.05,
    edge_property="syn_count_perc",
    exclude_nodes=None,
):
    """Thresholds the paths based on the threshold value"""
    thresholded_paths = [
        p
        for p in paths
        if (len(p) > 1)
        and (
            nx.path_weight(graph, path=list(p), weight=edge_property) / (len(p) - 1)
            >= threshold
        )
    ]
    # we prune the paths that include specific nodes
    if exclude_nodes is None:
        return thresholded_paths

    return [p for p in thresholded_paths if not set(p).intersection(set(exclude_nodes))]


def get_adj_matrix_from_conn(
    connectivity_df,
    neuron_groups,
):
    """Get the adjacency matrix from the connectivity dataframe between neuronal groups."""
    adjacency_matrix_perc = np.zeros((len(neuron_groups), len(neuron_groups)))
    adjacency_matrix_syn = np.zeros((len(neuron_groups), len(neuron_groups)))

    for i, (name_pre, nodes_pre) in enumerate(neuron_groups.items()):
        for j, (name_post, nodes_post) in enumerate(neuron_groups.items()):

            number_of_synapses_total = (
                connectivity_df[connectivity_df.pre_root_id.isin(nodes_pre)]
                .syn_count.abs()
                .sum()
            )
            number_of_synapses = (
                connectivity_df[
                    (connectivity_df.pre_root_id.isin(nodes_pre))
                    & (connectivity_df.post_root_id.isin(nodes_post))
                ]
                .syn_count.abs()
                .sum()
            )
            w_perc = (number_of_synapses / (number_of_synapses_total + 1e-5)) * 100

            adjacency_matrix_perc[i, j] = w_perc
            adjacency_matrix_syn[i, j] = number_of_synapses / (
                len(nodes_pre) * len(nodes_post)
            )

    return {
        "perc": adjacency_matrix_perc,
        "syn": adjacency_matrix_syn,
    }


def get_upstream(connectivity_df, neuron_ids):
    """Gets the upstream partners of a list of neurons"""
    neuron_ids = list(neuron_ids) if not isinstance(neuron_ids, list) else neuron_ids
    return connectivity_df[
        connectivity_df.post_root_id.isin(neuron_ids)
    ].pre_root_id.unique()


def threshold_connection(
    connectivity_df,
    pre_root_ids,
    post_root_ids,
    threshold_col_name="syn_count_perc",
    threshold=0.05,
    # verbose=False,
):
    """Thresholds the connections based on the threshold value"""
    pre_root_ids = (
        list(pre_root_ids) if not isinstance(pre_root_ids, list) else pre_root_ids
    )
    post_root_ids = (
        list(post_root_ids) if not isinstance(post_root_ids, list) else post_root_ids
    )
    thresholded_neurons = {}
    # for each pre root id, look at the synapse percentage
    for pre_id in pre_root_ids:
        sum_perc = connectivity_df[
            (connectivity_df.pre_root_id == pre_id)
            & (connectivity_df.post_root_id.isin(post_root_ids))
        ][threshold_col_name].sum()
        # if sum percentage is more than the threshold of the total synapses onto motor neurons
        if sum_perc >= threshold:
            thresholded_neurons[pre_id] = sum_perc
            # if verbose:
            #     print(pre_id, neurons.ALL_NEURONS_REV_JO[pre_id]["name"], sum_perc)

    return thresholded_neurons


def shuffle_postsyn_conn(adj_matrix, random_seed=0):
    """Shuffle the postsynaptic connection.
    This will ensure that a neuron will have
    the same number of synapses.
    """
    np.random.seed(random_seed)
    shuffled_adj = np.zeros_like(adj_matrix)
    for row in range(adj_matrix.shape[0]):
        shuffled_adj[row, :] = np.random.choice(
            adj_matrix[row, :], size=adj_matrix.shape[1], replace=False
        )

    return shuffled_adj


def make_a_random_graph(
    original_connectivity,
    motor_neuron_ids,
    seed=1,
):
    """Randomize the connections in a graph."""
    np.random.seed(seed)

    all_segment_ids = set(original_connectivity.pre_root_id.tolist()).union(
        set(original_connectivity.post_root_id.tolist())
    )
    all_segment_ids_wo_mns = all_segment_ids.difference(set(motor_neuron_ids))
    random_conn = copy.deepcopy(original_connectivity)

    random_conn["pre_root_id"] = np.random.choice(
        list(all_segment_ids_wo_mns), len(random_conn)
    )
    random_conn["post_root_id"] = np.random.choice(
        list(all_segment_ids), len(random_conn)
    )
    random_conn["syn_count"] = np.random.permutation(random_conn["syn_count"].values)

    random_conn["total_synapses"] = random_conn.groupby("pre_root_id")[
        "syn_count"
    ].transform("sum")
    #  Normalize syn_count by dividing it with total_synapses
    random_conn["syn_count_perc"] = (
        random_conn["syn_count"] / random_conn["total_synapses"]
    )
    random_conn["syn_count_inv"] = 1 / random_conn["syn_count"].values

    return random_conn


def get_premotor_neurons(connectivity_df, motor_neuron_ids, threshold=0.05):
    """Get the premotor neurons in the brain dataset, defined by a connectivity threshold."""
    mn_upstream = get_upstream(connectivity_df, motor_neuron_ids)
    mn_ups_percentage = threshold_connection(
        connectivity_df,
        mn_upstream,
        motor_neuron_ids,
        threshold=0.05,
    )
    return mn_ups_percentage


def get_prem_neurons_vnc():
    """Get the premotor neurons in the VNC, returns three dataframes: connectivity of neck, foreleg, and shared premotor neurons with their respective motor neurons."""

    c = Client("neuprint.janelia.org", dataset="manc:v1.2.1", token=MY_TOKEN)
    c.fetch_version()
    # Get the leg motor neurons
    neuron_df_npt1_left, _ = neuprint.fetch_neurons(
        neuprint.NeuronCriteria(outputRois=["LegNp(T1)(L)"])
    )

    neuron_df_npt1_right, _ = neuprint.fetch_neurons(
        neuprint.NeuronCriteria(outputRois=["LegNp(T1)(R)"])
    )

    # Concat the two dfs
    neuron_df_npt1 = pd.concat(
        [neuron_df_npt1_left, neuron_df_npt1_right], axis=0
    ).dropna(subset=["instance", "type"])

    # Search for the motor neurons
    t1_leg_mn_df = neuron_df_npt1[neuron_df_npt1.instance.str.contains("MNfl")]

    # Find the leg premotor neurons
    # Example: Fetch all upstream connections TO a set of neurons
    leg_mns = t1_leg_mn_df.bodyId.values
    leg_prem_df, leg_prem_conn_df = neuprint.fetch_adjacencies(
        sources=None, targets=leg_mns, min_total_weight=10
    )
    # drop if the bodyId_pre is in the leg motor neurons
    leg_prem_conn_df = leg_prem_conn_df[~leg_prem_conn_df.bodyId_pre.isin(leg_mns)]
    leg_prem_conn_clean_df = (
        leg_prem_conn_df.groupby(["bodyId_pre", "bodyId_post"]).sum().reset_index()
    )
    # Get the neck motor neurons
    neuron_df, _ = neuprint.fetch_neurons(neuprint.NeuronCriteria())
    neuron_df = neuron_df.dropna(subset=["instance", "type"])
    # search for the neck motor neurons
    neck_mn_df = neuron_df[neuron_df.instance.str.contains("MNnm")]

    # Get the neck premotor neurons
    neck_mns = neck_mn_df.bodyId.values
    neck_prem_df, neck_prem_conn_df = neuprint.fetch_adjacencies(
        sources=None, targets=neck_mns, min_total_weight=10
    )

    neck_prem_conn_df = neck_prem_conn_df[~neck_prem_conn_df.bodyId_pre.isin(neck_mns)]
    neck_prem_conn_clean_df = (
        neck_prem_conn_df.groupby(["bodyId_pre", "bodyId_post"]).sum().reset_index()
    )
    neck_prem_neurons = neck_prem_conn_clean_df.bodyId_pre.unique()
    neck_prem_df = neuron_df[neuron_df.bodyId.isin(neck_prem_neurons)]

    leg_prem_neurons = leg_prem_conn_clean_df.bodyId_pre.unique()
    leg_prem_df = neuron_df[neuron_df.bodyId.isin(leg_prem_neurons)]

    both_prem_neurons = np.intersect1d(leg_prem_neurons, neck_prem_neurons)
    common_prem_df = neuron_df[neuron_df.bodyId.isin(both_prem_neurons)]

    print(
        f"Neurons ({len(both_prem_neurons)}) projecting onto leg and neck motor neurons in the VNC are: {both_prem_neurons}"
    )

    return neck_prem_df, leg_prem_df, common_prem_df


def update_motor_modules(
    motor_module_neurons, anten_prem_set, neck_prem_set, leg_prem_set
):
    # Shared prem neurons: those at the intersection of at least two premotor neuron types
    neck_anten_leg_prem = neck_prem_set & anten_prem_set & leg_prem_set
    neck_anten_prem = neck_prem_set & anten_prem_set - neck_anten_leg_prem
    neck_leg_prem = neck_prem_set & leg_prem_set - neck_anten_leg_prem
    anten_leg_prem = anten_prem_set & leg_prem_set - neck_anten_leg_prem

    shared_prem = neck_anten_leg_prem.union(
        neck_anten_prem, neck_leg_prem, anten_leg_prem
    )
    all_prem_neurons = neck_prem_set.union(anten_prem_set, leg_prem_set)

    motor_module_neurons["leg_prem"] = list(leg_prem_set - shared_prem)
    motor_module_neurons["neck_prem"] = list(neck_prem_set - shared_prem)
    motor_module_neurons["anten_prem"] = list(anten_prem_set - shared_prem)
    motor_module_neurons["shared_prem"] = list(shared_prem)
    motor_module_neurons["all_prem"] = list(all_prem_neurons)

    return motor_module_neurons


def get_panel_m(
    grooming_network, motor_module_neurons, shared_prem, anten_prem_set, neck_prem_set
):
    # Look at how much of information to MNs come from shared vs. other prem neurons
    # total number of synapses onto neck mns
    neck_mns_upstream_synapses = grooming_network[
        grooming_network.post_root_id.isin(motor_module_neurons["neck_mn"])
    ].syn_count.sum()
    # number of synapses from shared prem onto neck mns
    neck_mns_shared_prem_input = grooming_network[
        (grooming_network.post_root_id.isin(motor_module_neurons["neck_mn"]))
        & (grooming_network.pre_root_id.isin(shared_prem))
    ].syn_count.sum()
    # number of synapses from ind. prem onto neck mns
    neck_mns_ind_input = grooming_network[
        (grooming_network.post_root_id.isin(motor_module_neurons["neck_mn"]))
        & (grooming_network.pre_root_id.isin(neck_prem_set - shared_prem))
    ].syn_count.sum()

    neck_input_percen = {
        "shared": 100 * neck_mns_shared_prem_input / neck_mns_upstream_synapses,
        "ind": 100 * neck_mns_ind_input / neck_mns_upstream_synapses,
    }

    print(
        f"""Neck motor neurons receive {neck_input_percen["shared"]:.2f}% of their input from shared premotor neurons"""
    )

    # Same for the antennal motor neurons
    # total number of synapses onto anten mns
    anten_mns_upstream_synapses = grooming_network[
        grooming_network.post_root_id.isin(motor_module_neurons["anten_mn"])
    ].syn_count.sum()
    # number of synapses from shared prem onto anten mns
    anten_mns_shared_prem_input = grooming_network[
        (grooming_network.post_root_id.isin(motor_module_neurons["anten_mn"]))
        & (grooming_network.pre_root_id.isin(shared_prem))
    ].syn_count.sum()
    # number of synapses from ind prem onto anten mns
    anten_mns_ind_input = grooming_network[
        (grooming_network.post_root_id.isin(motor_module_neurons["anten_mn"]))
        & (grooming_network.pre_root_id.isin(anten_prem_set - shared_prem))
    ].syn_count.sum()

    anten_input_percen = {
        "shared": 100 * anten_mns_shared_prem_input / anten_mns_upstream_synapses,
        "ind": 100 * anten_mns_ind_input / anten_mns_upstream_synapses,
    }

    print(
        f"""Antennal motor neurons receive {anten_input_percen["shared"]:.2f}% of their input from shared premotor neurons"""
    )
    return neck_input_percen, anten_input_percen


def identify_central_neurons(grooming_network, motor_module_neurons):
    """To identify central neurons, we look at all simple paths from the JOF to the premotor/motor neurons and take those with a connection strength more than a threshold."""

    graph_table = nx.from_pandas_edgelist(
        grooming_network,
        source="pre_root_id",
        target="post_root_id",
        edge_attr=["syn_count", "syn_count_perc"],
        create_using=nx.DiGraph(),
    )

    cutoff = 4  # based on the layer algorithm max depth
    threshold = 0.05  # average synapse count percentage

    jof2prem_simple_paths = get_all_simple_paths(
        graph_table,
        # From
        motor_module_neurons["jo_f"],
        # To
        motor_module_neurons["all_prem"]
        + motor_module_neurons["neck_mn"]
        + motor_module_neurons["anten_mn"],
        cutoff=cutoff,
    )
    # Prune those with low connection strength
    jof2prem_simple_paths_pruned = threshold_prune_paths(
        graph_table,
        jof2prem_simple_paths,
        threshold=threshold,
        edge_property="syn_count_perc",
        exclude_nodes=motor_module_neurons["jo_e"]
        + motor_module_neurons["jo_c"]
        + motor_module_neurons["bm_ant"],
    )
    # Exclude sensory and motor module neurons
    central_neurons = set(
        itertools.chain.from_iterable(jof2prem_simple_paths_pruned)
    ).difference(
        set(
            motor_module_neurons["jo_f"]
            + motor_module_neurons["all_prem"]
            + motor_module_neurons["neck_mn"]
            + motor_module_neurons["anten_mn"]
        )
    )

    return list(central_neurons)


def get_conn_between_groups(grooming_network, motor_module_neurons):
    """Calculate the connectivity percentage between neuron groups."""
    input_dictionary = {
        "sensory": motor_module_neurons["jo_f"],
        "central": motor_module_neurons["central"],
        "anten prem": motor_module_neurons["anten_prem"],
        "neck prem": motor_module_neurons["neck_prem"],
        "leg prem": motor_module_neurons["leg_prem"],
        "shared prem": motor_module_neurons["shared_prem"],
    }

    prem_mn_dictionary = {
        "central": motor_module_neurons["central"],
        "anten prem": motor_module_neurons["anten_prem"],
        "neck prem": motor_module_neurons["neck_prem"],
        "leg prem": motor_module_neurons["leg_prem"],
        "shared prem": motor_module_neurons["shared_prem"],
        "anten mn": motor_module_neurons["anten_mn"],
        "neck mn": motor_module_neurons["neck_mn"],
    }

    input_central_to_prem_array = np.zeros(
        (len(input_dictionary) + 1, len(prem_mn_dictionary))
    )
    # For each premotor neuron type, get the percentage of synapses going to each premotor neuron type
    input_central_to_prem_perc = {}
    for j, to_prem_type in enumerate(prem_mn_dictionary):
        total_synapses = grooming_network[
            grooming_network.post_root_id.isin(prem_mn_dictionary[to_prem_type])
        ].syn_count.sum()
        for i, from_prem_type in enumerate(input_dictionary):
            btw_synapses = grooming_network[
                (grooming_network.pre_root_id.isin(input_dictionary[from_prem_type]))
                & (grooming_network.post_root_id.isin(prem_mn_dictionary[to_prem_type]))
            ].syn_count.sum()

            # percentage
            if total_synapses == 0:
                perc = 0
            else:
                perc = 100 * btw_synapses / total_synapses

            # prem_to_prem_perc[(from_prem_type, to_prem_type)] = perc
            input_central_to_prem_array[i, j] = perc

    # Other
    input_central_to_prem_array[-1, :] = 100 - input_central_to_prem_array.sum(axis=0)
    return input_central_to_prem_array


def draw_graph(
    adjacency_matrix,
    neuron_group_names_pre,
    neuron_group_names_post,
    pos_custom,
    node_colors,
    title="",
    fig_name=None,
    edge_label=False,
    normalize_edge_weight=False,
    threshold=0,
    connectionstyle="arc3,rad=0.1",
    export_path=None,
):

    # Plot a graph from the adjacency matrix
    neuron_names_pre = [name.replace("_", " ") for name in neuron_group_names_pre]
    neuron_names_post = [name.replace("_", " ") for name in neuron_group_names_post]
    neuron_names = np.unique(neuron_names_pre + neuron_names_post)

    fig, ax = plt.subplots(figsize=(2.4, 2.8), dpi=300)
    edge_color = []
    edge_weight = []
    edges = []

    # create directed graph from transition dataframe
    G = nx.DiGraph()

    for neuron in neuron_names:
        G.add_node(neuron)

    for row in range(adjacency_matrix.shape[0]):
        for column in range(adjacency_matrix.shape[1]):
            if np.abs(adjacency_matrix[row, column]) > threshold:
                # print(
                #     f"{neuron_names_pre[row]} -> {neuron_names_post[column]}: {adjacency_matrix[row, column]}"
                # )
                G.add_edge(
                    neuron_names_pre[row],
                    neuron_names_post[column],
                    weight=np.abs(adjacency_matrix[row, column]),
                )

                edge_color.append(
                    "darkblue" if adjacency_matrix[row, column] < 0 else "darkred"
                )
                edges.append((neuron_names_pre[row], neuron_names_post[column]))
                edge_weight.append(adjacency_matrix[row, column])

    edge_weight = np.array(edge_weight)
    # print(edge_weight.min(), edge_weight.max())
    if normalize_edge_weight:
        edge_weight -= edge_weight.min() - 0.2
        edge_weight /= edge_weight.max() - edge_weight.min() - 0.2
        edge_weight *= 4.5

    # create node and edge labels
    node_labels = {
        node: node.replace(" ", "\n").replace("OTHER", "").replace("NEURONS", "")
        for node in G.nodes()
    }
    edge_labels = {(u, v): round(d["weight"], 1) for u, v, d in G.edges(data=True)}

    # create graph layout and draw nodes, edges, and labels
    pos = pos_custom

    node_size = 300
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=neuron_names,
        node_size=node_size,
        node_shape="s",
        node_color=[node_colors[name] for name in G.nodes()],
        alpha=1,
        edgecolors="black",
    )
    # print(G.edges())
    nx.draw_networkx_edges(
        G,
        pos,
        # node_size=node_size,
        edgelist=edges,
        width=edge_weight,
        arrows=True,
        edge_color="black",  # edge_color,
        connectionstyle=connectionstyle,
        alpha=1,
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=5)

    if edge_label:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=3,
            rotate=False,
            label_pos=0.35,
        )

    # display graph
    plt.axis("off")
    ax.margins(0.1)
    ax.set_title(title)

    if fig_name is not None and export_path is not None:
        fig.savefig(
            export_path / f"{fig_name}.png", bbox_inches="tight", facecolor="white"
        )

    # plt.show()

    return G, edge_weight


def order_signal_flow(grooming_network):
    """Takes the connectivity table of a network, and orders nodes based on their signal flow score (i.e., input to output)."""
    graph_table = nx.from_pandas_edgelist(
        grooming_network,
        source="pre_root_id",
        target="post_root_id",
        edge_attr=["syn_count", "syn_count_perc"],
        create_using=nx.DiGraph(),
    )
    # make the adj matrix
    adj_matrix_unsigned = nx.to_numpy_array(graph_table, weight=None)

    # apply the signal flow algorithm
    z = signal_flow(adj_matrix_unsigned)

    sort_inds = np.argsort(z)[::-1]
    adj_sorted_unsigned = adj_matrix_unsigned[np.ix_(sort_inds, sort_inds)]

    # get list of node names
    nodes = np.array(list(graph_table.nodes()))[sort_inds]

    # sort z too
    z_sorted = z[sort_inds]

    return adj_sorted_unsigned, nodes, z_sorted


def layer_neurons(nodes, signal_flow_score, no_layers=10):
    """Divide the neurons into layers based on their signal flow score."""
    layer_bounds = np.linspace(
        signal_flow_score.max(), signal_flow_score.min(), no_layers
    )
    layer_neurons = {}

    for layer_no in range(len(layer_bounds) - 1):
        layer_neurons[layer_no + 1] = nodes[
            np.logical_and(
                layer_bounds[layer_no] >= signal_flow_score,
                signal_flow_score >= layer_bounds[layer_no + 1],
            )
        ]

    return layer_neurons, layer_bounds


def get_connectivity_between_layers(layer_neurons, grooming_network):
    """ Computes the number of synapses between layers. """
    layers = list(layer_neurons.keys())
    # connectivity between the layers
    layer_by_layer = np.zeros((len(layers), len(layers)))
    layer_by_layer_inh = np.zeros((len(layers), len(layers)))
    layer_by_layer_exc = np.zeros((len(layers), len(layers)))

    for layer1 in sorted(layers):
        layer1_neurons = layer_neurons[layer1]
        for layer2 in sorted(layers):
            layer2_neurons = layer_neurons[layer2]
            synapses = grooming_network[
                grooming_network.pre_root_id.isin(layer1_neurons)
                & grooming_network.post_root_id.isin(layer2_neurons)
            ]

            synapses_inh = synapses[synapses.nt_type.isin(["GABA", "GLUT"])].syn_count.sum()
            synapses_exc = synapses[
                synapses.nt_type.isin(["ACH", "SER", "DA", "OCT"])
            ].syn_count.sum()

            synapses_total = synapses.syn_count.sum()
            assert synapses_total == synapses_inh + synapses_exc

            layer_by_layer[layer1 - 1, layer2 - 1] = synapses_total
            layer_by_layer_inh[layer1 - 1, layer2 - 1] = synapses_inh
            layer_by_layer_exc[layer1 - 1, layer2 - 1] = synapses_exc

    return layer_by_layer, layer_by_layer_inh, layer_by_layer_exc


def classify_central_neurons(grooming_network, neuron_groups):
    """ Classify central neurons based on their projections onto premotor neurons."""
    central_neurons = neuron_groups["central"]
    shared_prem = neuron_groups["shared_prem"]
    neck_prem = neuron_groups["neck_prem"]
    anten_prem = neuron_groups["anten_prem"]
    leg_prem = neuron_groups["leg_prem"]

    # neck upstream and central
    shared_prem_upstream = set(grooming_network[
        grooming_network.pre_root_id.isin(central_neurons) &
        grooming_network.post_root_id.isin(shared_prem)
    ].pre_root_id.unique())

    neck_prem_upstream = set(grooming_network[
        grooming_network.pre_root_id.isin(central_neurons) &
        grooming_network.post_root_id.isin(neck_prem)
    ].pre_root_id.unique()) - shared_prem_upstream

    anten_prem_upstream = set(grooming_network[
        grooming_network.pre_root_id.isin(central_neurons) &
        grooming_network.post_root_id.isin(anten_prem)
    ].pre_root_id.unique()) - shared_prem_upstream

    leg_prem_upstream = set(grooming_network[
        grooming_network.pre_root_id.isin(central_neurons) &
        grooming_network.post_root_id.isin(leg_prem)
    ].pre_root_id.unique()) - shared_prem_upstream

    central_array_projection = np.zeros((len(central_neurons), 5))

    for central_id, central_neuron in enumerate(central_neurons):
        if central_neuron in anten_prem_upstream:
            central_array_projection[central_id, 0] = 1
        if central_neuron in neck_prem_upstream:
            central_array_projection[central_id, 1] = 1
        if central_neuron in leg_prem_upstream:
            central_array_projection[central_id, 2] = 1
        if central_neuron in shared_prem_upstream:
            central_array_projection[central_id, 3] = 1
        # if not in any premotor neuron
        if (
            central_neuron not in shared_prem_upstream
            and central_neuron not in neck_prem_upstream
            and central_neuron not in anten_prem_upstream
            and central_neuron not in leg_prem_upstream
        ):
            central_array_projection[central_id, 4] = 1

    return central_array_projection