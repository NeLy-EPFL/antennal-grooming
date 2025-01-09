import Figure4_neurons as neurons
from collections import ChainMap
from typing import List, Dict, Tuple
import math
from pathlib import Path
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import trange
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import sys

warnings.simplefilter(action="ignore", category=FutureWarning)

neurons_JO = [
    "BM-Ant L",
    "JO-C L",
    "JO-E L",
    "JO-F L",
    "BM-Ant R",
    "JO-C R",
    "JO-E R",
    "JO-F R",
]

OFFSET_DICT = {
    "level_0": {
        "neuron_dict": neurons.neuron_dict["sensory_neurons"],
        "x_start": 0.9,
        "nx": 6,
        "offset": 1.8,
    },
    "level_1": {
        "neuron_dict": neurons.neuron_dict["ANTEN_MN"],
        "nx": 7,
        "offset": 1.4,
        "x_start": 0.5,
    },
    "level_2": {
        "neuron_dict": neurons.neuron_dict["WED"],
        "nx": 8,
        "offset": 1.0,
    },
    "level_3": {
        "neuron_dict": {**neurons.neuron_dict["ABN"], **neurons.neuron_dict["IN"]},
        "nx": 8,
        "offset": 0.45,
    },
    "level_4": {
        "neuron_dict": {
            **neurons.neuron_dict["ADN"],
            **neurons.neuron_dict["DN"],
            **neurons.neuron_dict["12A_DN"],
        },
        "nx": 8,
        "offset": -0.9,
    },
    "level_5": {
        "neuron_dict": neurons.neuron_dict["NECK_MN"],
        "nx": 5,
        "offset": -2.5,
        "x_start": 0.5,
    },
}

NT_TYPES = {
    "GABA": {"color": "navy", "linestyle": "-"},
    "ACH": {"color": "darkred", "linestyle": "-"},
    "GLUT": {"color": "steelblue", "linestyle": "-"},
    "OCT": {"color": "grey", "linestyle": "-"},
    "SER": {"color": "grey", "linestyle": "-"},
    "DA": {"color": "grey", "linestyle": "-"},
}

MAP_COLORS = {
    "darkred": 1,
    "navy": -1,
    "grey": 1,
    "steelblue": -1,
}


def get_node_circle_color(external_input, neuron_color_dict, index_name_list):
    """Gets color of node boundary based on the external input.
    It makes the boundary red if there is an external input.
    """
    external_input_colors = np.empty_like(external_input, dtype=np.dtype("U100"))

    for level, group_dict in neuron_color_dict.items():
        neuron_indices = [
            index_name_list.index(name.replace("_", " "))
            for name in group_dict["neuron_dict"]
            if name.replace("_", " ") in index_name_list
        ]
        external_input_colors[neuron_indices, :] = np.where(
            external_input[neuron_indices, :] > 0, "red", group_dict["color"]
        )

    return external_input_colors


def get_network_specs(
    connectivity_df: pd.DataFrame,
    neurons_of_interest: Dict = None,
    pre_label="pre_root_id",
    post_label="post_root_id",
) -> Dict:
    """From a connectivity table, obtains the edges and visual network properties.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        Connectivity table that includes `pre_root_id`, `post_root_id`,
        `nt_type`, `syn_count` as columns.
    neurons_of_interest : Dict, optional
        Nested dictionary containing segment ID as keys,
        a dictionary of color and segment name as values
        ,by default None
        It should have the following keys:
        - color: str
        - name: str


    Returns
    -------
    Dict
        Dictionary containing the network specs such as edges, colors etc.
    """

    unique_neurons = list(
        set(connectivity_df[pre_label]).union(set(connectivity_df[post_label]))
    )

    pre_root = connectivity_df[pre_label].to_list()
    post_root = connectivity_df[post_label].to_list()

    edges = []
    for pre, post in zip(pre_root, post_root):
        edges.append((pre, post))

    edge_weights = connectivity_df.syn_count.to_numpy()

    try:
        edge_type = connectivity_df.nt_type.to_numpy()
        edge_colors = [NT_TYPES[nt_name]["color"] for nt_name in edge_type]
        edge_ls = [NT_TYPES[nt_name]["linestyle"] for nt_name in edge_type]
    except AttributeError:
        print("Neurotransmitter type does not exist.")
        edge_type = ["unknown"] * len(edges)
        edge_colors = ["grey"] * len(edges)
        edge_ls = ["-"] * len(edges)

    if neurons_of_interest is not None:
        node_colors = [
            (
                neurons_of_interest[neuron_id]["color"]
                if neuron_id in neurons_of_interest
                else "lightgrey"
            )
            for neuron_id in unique_neurons
        ]
        node_labels = {
            neuron_id: (
                neurons_of_interest[neuron_id]["name"].replace("_", " ")
                if neuron_id in neurons_of_interest
                else neuron_id
            )
            for neuron_id in unique_neurons
        }
    else:
        node_colors = ["lightgrey"] * len(unique_neurons)
        node_labels = {idx: "" for idx in unique_neurons}

    return {
        "nodes": unique_neurons,
        "edges": edges,
        "edge_type": edge_type,
        "edge_colors": edge_colors,
        "edge_linestyle": edge_ls,
        "weights": edge_weights,
        "node_colors": node_colors,
        "node_labels": node_labels,
    }


def neurons_right_center(neuron_dict):
    return [neuron for neuron in neuron_dict if "_L" not in neuron]


def dist(new_point, points, r_threshold):
    """Check if new_point is within r_threshold of any point in points"""
    for point in points:
        dist = np.sqrt(np.sum(np.square(new_point - point)))
        if dist < r_threshold:
            return False
    return True


def meshgrid_positions(
    neuron_dict, offset, ascending_step=0.1, nx=6, x_start=0.2, **kwargs
):
    position_dict = {}

    nx = int(nx)
    ny = math.ceil(
        len((neuron_list_right_center := neurons_right_center(neuron_dict))) / nx
    )
    #     print(nx, ny, "\n", "-----")
    x = np.linspace(x_start, 1.6, nx)
    y = np.linspace(offset, offset - ny * 0.12, ny)

    xx, yy = np.meshgrid(x, y)

    i = 0
    for neuron in neuron_list_right_center:

        row, column = i // nx, i % nx

        offset_x = 0.1 if row % 2 == 0 else 0
        offset_y = ascending_step * column

        if "R" not in neuron and "L" not in neuron:
            position_dict[neuron.replace("_", " ")] = np.array(
                [0, yy[row, column] - 0.1 * random.randint(3, 9)]
            )
            print(f"neuron: {neuron}, position: {0, yy[row,column] - 0.1}")
        else:
            x_pos = xx[row, column] + offset_x
            y_pos = yy[row, column] + offset_y

            position_dict[neuron.replace("_", " ")] = np.array([x_pos, y_pos])
            position_dict[neuron.replace("R", "L").replace("_", " ")] = np.array(
                [-x_pos, y_pos]
            )

            #             print(f"neuron: {neuron}, position: {xx[row,column], yy[row,column]}")
            i += 1

    return position_dict


def get_node_position(node_labels, default_pos):
    """Get node positions for plotting network.
    Randomly assigns the point locations if the neuron position is
    not pre-defined in `NODE_POS_DEFAULT`.
    """
    new_pos = {}
    for seg_id, name in node_labels.items():
        if name in default_pos:
            pos = default_pos[name]
        elif name[:4] in ["BM-A", "JO-C", "JO-F", "JO-E"]:
            if name[-1] == "R":
                pos = np.array([random.uniform(0.6, 1.0), 1.95])
            else:
                pos = np.array([random.uniform(-1.0, -0.6), 1.95])
        else:
            for _ in range(30):
                pos = np.array([random.uniform(-1.6, 1.6), random.uniform(0.0, 1.15)])
                if dist(pos, list(new_pos.values()), 0.2):
                    new_pos[seg_id] = pos
                    break

        new_pos[seg_id] = pos

    return new_pos


def highlight_connections(network_specs, segment_ids, alpha=0.5):
    """Highlight connections that neurons of interest make."""
    segment_ids = [
        segment_id
        for segment_id in list(segment_ids)
        if segment_id in network_specs["nodes"]
    ]
    new_transparency = []

    for one_edge in network_specs["edges"]:
        if any([segment_id in one_edge for segment_id in segment_ids]):
            new_transparency.append(alpha)
        else:
            new_transparency.append(0)

    return new_transparency


def highlight_path_btw_edges(network_specs, paths_to_highlight, alpha=0.5):
    """Highlight connections that neurons of interest make."""
    new_transparency = []

    for one_edge in network_specs["edges"]:
        if one_edge in paths_to_highlight:
            new_transparency.append(alpha)
        else:
            new_transparency.append(0)

    return new_transparency


def plot_connectivity(
    connectivity_df: pd.DataFrame,
    neurons_of_interest: List = None,
    highlight_neuron: List = None,
    highlight_paths: List = None,
    synapse_th: int = 0,
    **kwargs,
) -> Tuple:
    """Wrap function around `draw_network`

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        Connectivity table.
    neurons_of_interest : List, optional
        Neurons to be plotted, by default None
    pos : Dict, optional
        Positions of nodes, by default None
    highlight_neuron : List, optional
        Highlight the connections that certain
        neurons, provide a list, by default None
    synapse_th : int, optional
        Synapse threshold, by default 0

    Returns
    -------
    Tuple
        Dictionary containing the network specifications
        and node positions
    """
    pos = kwargs.pop("pos", None)

    neurons_of_interest = (
        neurons.ALL_NEURONS_REV if neurons_of_interest is None else neurons_of_interest
    )

    if synapse_th:
        connectivity_df = connectivity_df[
            connectivity_df.syn_count > synapse_th
        ].reset_index(drop=True)

    #  with node positions determined by the algorithm
    network_specs = get_network_specs(
        connectivity_df, neurons_of_interest=neurons_of_interest
    )

    if highlight_neuron is not None:
        network_specs["node_transparency"] = [
            0.8 if neuron in highlight_neuron else 0.1
            for neuron in network_specs["nodes"]
        ]
        network_specs["transparency"] = highlight_connections(
            network_specs, highlight_neuron
        )

    if highlight_paths is not None:
        network_specs["transparency"] = highlight_path_btw_edges(
            network_specs, highlight_paths
        )
    # from IPython import embed; embed()
    pos = (
        get_node_position(
            network_specs["node_labels"], default_pos=ALL_POSITION_DEFAULT
        )
        if pos is None
        else pos
    )
    # pos = None
    G, pos = plot_network(network_specs, pos=pos, **kwargs)

    return network_specs, G


def find_connection_type(connectivity_df, connection_type):
    """Returns the number of synapses of the same connection type."""
    nt_list = ["GABA", "GLUT"] if connection_type.startswith("i") else ["ACH"]

    return connectivity_df.apply(
        lambda row: (row["syn_count"] if row["nt_type"] in nt_list else 0), axis=1
    ).sum()


def analyze_neuron_connection(segment_id, connectivity_df):
    """Returns the number of excitatory and inhibitoy pre- and post-synaptic connection numbers."""
    # look at the upstream (where the neuron is post-synaptic)
    upstream_neurons = connectivity_df[connectivity_df["post_root_id"] == segment_id]

    # look at the downstream (where the neuron is pre-synaptic)
    downstream_neurons = connectivity_df[connectivity_df["pre_root_id"] == segment_id]

    # classify the connections (for now only the neurotransmitter)
    no_excitatory_upstream = find_connection_type(upstream_neurons, "excitatory")
    no_inhibitory_upstream = find_connection_type(upstream_neurons, "inhibitory")

    no_excitatory_downstream = find_connection_type(downstream_neurons, "excitatory")
    no_inhibitory_downstream = find_connection_type(downstream_neurons, "inhibitory")

    return {
        "segment_id": segment_id,
        "exc_up": no_excitatory_upstream,
        "exc_down": no_excitatory_downstream,
        "inh_up": no_inhibitory_upstream,
        "inh_down": no_inhibitory_downstream,
    }


def plot_neuron_connection_type(conn_dict, export_path=None, title=""):
    """Plots the connection type as a heatmap."""
    x_axis = ["inhibitory", "excitatory"]
    y_axis = ["upstream", "downstream"]

    conn_type = np.array(
        [
            [conn_dict["inh_up"], conn_dict["exc_up"]],
            [conn_dict["inh_down"], conn_dict["exc_down"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(conn_type, cmap="Reds")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_axis)), labels=x_axis, size=10)
    ax.set_yticks(np.arange(len(y_axis)), labels=y_axis, size=10)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            text = ax.text(
                j,
                i,
                conn_type[i, j],
                ha="center",
                va="center",
                color="lightgrey",
                fontdict={"size": 13},
            )

    ax.set_title(title, size=12)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=6)
    ax.spines[:].set_visible(False)

    fig.tight_layout()

    if export_path is not None:
        plt.savefig(export_path, bbox_inches="tight", dpi=300)

    plt.show()


def create_graph(
    index_name_list,
    set_position: bool = True,
    pos: Dict = None,
    show_segment_id: bool = False,
):
    """Creates a graph without edges for the animation."""
    # Create graph from adjacency matrix
    graph = nx.MultiDiGraph()

    for node in index_name_list:
        graph.add_node(node)

    # These are algorithms to set the positions. Otherwise, you can set  the position by having:
    # "pos = {0:(0,0), 1:(1,1), 2:(2,2)}"
    if pos is None and set_position:
        pos = get_node_position(index_name_list, default_pos=ALL_POSITION_DEFAULT)
    elif pos is None and not set_position:
        pos = nx.kamada_kawai_layout(graph)

    if not show_segment_id:
        node_labels = {idx: name for idx, name in enumerate(index_name_list)}

    return graph, node_labels, pos


def plot_network(
    network_specs: Dict,
    ax: plt.Axes = None,
    pos: Dict = None,
    node_size: int = 300,
    font_size: int = 8,
    font_color: str = "black",
    title: str = "",
    export_path: str = None,
    legend: bool = False,
    show_segment_id: bool = True,
    layout: str = "kamada_kawai",
    normalize_w: bool = True,
    max_w: int = 7,
    alpha: float = 0.5,
    node_alpha: float = 0.8,
) -> Tuple:
    """Plots the network using the network specsa and
    returns the positions of the nodes.

    Parameters
    ----------
    network_specs : Dict
        Dictionary containing the network characteristics.
    ax : plt.Axes, optional
        Ax that the network will be plotted on, by default None
    node_size : int, optional
        Size of the network nodes, by default 300
    title : str, optional
        Title of the figure, by default ''
    export_path : str, optional
        Path to save the figure, by default None
    activate_click : bool, optional
        Prints out the click location on the figure, by default False
    legend : bool, optional
        Displays the edge characteristics, by default False
    show_segment_id : bool, optional
        Displays the segment ID of nodes, by default True

    Returns
    -------
    Tuple
        Returns the networkX.Graph and a dictionary containing node ID
        as keys and positions of nodes on the figure as values.


    Example Usage:
    >>> connectivity_df = connectivity_df[
        connectivity_df['pre_root_id'] == 720575940624319124
        )
    >>> neurons_of_interest = {
        720575940624319124: {'name': 'aDN1_R', 'color': 'seagreen'}
        }
    >>> network_specs = get_network_specs(
        connectivity_df, neurons_of_interest=neurons_of_interes
        t)
    >>> plot_network(
            network_specs, title='Connectivity of aDN1 right',
            export_path='./adn1_conn.png'
        )
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)

    # Create graph from adjacency matrix
    G = nx.MultiDiGraph()

    # Normalize the edge width of the network
    if normalize_w:
        normalized_weights = (
            (network_specs["weights"] - network_specs["weights"].min())
            / (network_specs["weights"].max() - network_specs["weights"].min())
            + 0.1
        ) * max_w
    else:
        normalized_weights = network_specs["weights"]

    for node in network_specs["nodes"]:
        G.add_node(node)

    for edge_pair, w, c in zip(
        network_specs["edges"], network_specs["weights"], network_specs["edge_colors"]
    ):
        # G.add_edge(*edge_pair)
        # be careful here
        G.add_edge(*edge_pair, weight=w * MAP_COLORS[c])

    # These are algorithms to set the positions. Otherwise, you can set  the position by having:
    # "pos = {0:(0,0), 1:(1,1), 2:(2,2)}"
    if pos is None:
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)

    if not show_segment_id:
        network_specs["node_labels"] = {
            idx: "" if isinstance(name, int) else name
            for idx, name in network_specs["node_labels"].items()
        }

    node_alpha = (
        network_specs["node_transparency"]
        if "node_transparency" in network_specs.keys()
        else node_alpha
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=network_specs["nodes"],
        node_color=network_specs["node_colors"],
        node_size=node_size,
        edgecolors="lightgrey",
        alpha=node_alpha,
        ax=ax,
    )

    alpha = (
        network_specs["transparency"]
        if "transparency" in network_specs.keys()
        else alpha
    )
    # edges
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=alpha,
        edgelist=network_specs["edges"],
        width=normalized_weights,
        style=network_specs["edge_linestyle"],
        edge_color=network_specs["edge_colors"],
        # arrowsize=1,
        # connectionstyle="arc3,rad=0.075",
        ax=ax,
    )

    # change node_labels if not highligted

    node_labels = {
        idx: (
            name if (not isinstance(node_alpha, list)) or (node_alpha[i] == 0.8) else ""
        )
        for i, (idx, name) in enumerate(network_specs["node_labels"].items())
    }

    # node labels
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=font_size,
        font_color=font_color,
        font_family="sans-serif",
        ax=ax,
    )

    if legend:
        ax.text(
            -0.05,
            0.10,
            "inhibitory",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontdict={"color": "navy"},
        )
        ax.text(
            -0.05,
            0.05,
            "excitatory",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontdict={"color": "darkred"},
        )
        ax.text(
            -0.05,
            0.0,
            "edge width = # synapses",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontdict={"color": "black"},
        )

    ax.margins(0.1)
    plt.axis("off")

    if title:
        ax.set_title(title)

    if export_path is not None:
        plt.savefig(export_path, bbox_inches="tight", transparent=True, dpi=300)

    return G, pos


def make_adjacency_matrix(graph, neurons_ordered, return_type="binary"):
    """Create an adjacency matrix from a networkx graph"""
    # create an adjacency matrix with the new order
    adjacency_matrix = nx.adjacency_matrix(
        graph, nodelist=neurons_ordered.keys(), weight="syn_count"
    ).todense()

    # multiply the rows with -1 where the node is inhibitory
    inhibitory_rows = [
        neurons_ordered[seg_id]["nt"] == "inhibitory" for seg_id in neurons_ordered
    ]

    adjacency_matrix_weighted = adjacency_matrix.copy()
    adjacency_matrix_weighted[inhibitory_rows] *= -1

    if return_type == "binary":
        adj_matrix_binary = np.where(adjacency_matrix_weighted > 0, 1, 0) + np.where(
            adjacency_matrix_weighted < 0, -1, 0
        )
        return adj_matrix_binary

    return adjacency_matrix_weighted


def get_connectivity_btw_neurons_clean(
    connectivity_df: pd.DataFrame,
    neurons_of_interest1: List,
    neurons_of_interest2: List = None,
    col_name1: str = "pre_root_id",
    col_name2: str = "post_root_id",
) -> pd.DataFrame:
    """Gets the connectivity among the neurons of interest

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        Table containing the connections between two neurons
        Should have two columns as `pre_root_id` and `post_root_id`
    neurons_of_interest1 : List
        List of neuron segment IDs os interest for presynaptic partner
    neurons_of_interest2 : List, by default None
        List of neuron segment IDs os interest for postsynaptic partner
        If None, then it is equal to `neurons_of_interest1`
    col_name1 : str, optional
        Column name of presynaptic neurons, by default 'pre_root_id'
    col_name2 : str, optional
        Column name of postsynaptic neurons, by default 'post_root_id'

    Returns
    -------
    pd.DataFrame
        Dataframe containing the connections between the neuron of interest makes.
    """
    if neurons_of_interest2 is None:
        neurons_of_interest2 = neurons_of_interest1

    connectivity_df = connectivity_df[
        connectivity_df[col_name1].isin(neurons_of_interest1)
        & connectivity_df[col_name2].isin(neurons_of_interest2)
    ]
    return connectivity_df.copy()


def scale_distribution(nt_dict, scale, nt_types):
    scaled = {
        nt: 100 * count / scale for nt, count in nt_dict.items() if nt in nt_types
    }
    scaled["OTHER"] = 100 - sum(scaled.values())
    return scaled


def sort_dictionary_by_values(dictionary):
    return dict(
        sorted(
            #  Reverse is true for ascending order
            dictionary.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )


def get_nt_information(network):
    # Load the FAFB dataset
    neurons_table = pd.read_csv("../data/codex_v783_Oct2023_neurons.csv")
    connections_table = pd.read_csv(
        "../data/codex_v783_Oct2023_connections_cleaned.csv"
    )

    # Calculate neurotransmitter distribution by synapse
    nt_synapse_dict = sort_dictionary_by_values(
        dict(network.groupby("nt_type").syn_count.count())
    )
    scale_nt = sum(nt_synapse_dict.values())

    # Calculate neurotransmitter distribution by neuron
    neurons_table_ant = neurons_table[
        neurons_table.root_id.isin(network.pre_root_id.unique())
    ]

    nt_neuron_dict = sort_dictionary_by_values(
        dict(neurons_table_ant.groupby("nt_type").root_id.count())
    )
    scale_nt_neuron = sum(nt_neuron_dict.values())

    # Scale distributions
    nt_types = ["GABA", "ACH", "GLUT"]
    scaled_synapse = scale_distribution(nt_synapse_dict, scale_nt, nt_types)
    scaled_neuron = scale_distribution(nt_neuron_dict, scale_nt_neuron, nt_types)

    return scaled_synapse, scaled_neuron


def fig_to_array(fig):
    """Converts a matplotlib figure into an array."""

    canvas = FigureCanvas(fig)
    canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


ALL_POSITION_LIST = [
    meshgrid_positions(
        offs_dict["neuron_dict"], nx=offs_dict["nx"], offset=offs_dict["offset"]
    )
    for stage, offs_dict in OFFSET_DICT.items()
]

ALL_POSITION_DEFAULT = ChainMap(*ALL_POSITION_LIST)
