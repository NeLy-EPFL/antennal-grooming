import time
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter

import networkx as nx

from mycolorpy import colorlist as mcp

# Dictionary of colors
COLORS = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
    "grey": "#D3D3D3",
    "purple": "#504E96",
}

NODE_COLORS = {
    "JO-F": "#D45B83",
    'JO-C': "#934161",
    'JO-E': "#4F2334",
    "CENTRAL": "#C3B5D9",
    "ANTEN_PREM": "#C2DFA0",
    "NECK_PREM": "#59BAD6",
    "LEG_PREM": "#E9A686",
    "SHARED_PREM": "#CFCFCE",
    "ANTEN_MN": "#798963",
    "NECK_MN": "#356D7A",
    "OTHER": "black"
}


def calc_pearson(vector1, vector2):
    from scipy.stats import pearsonr
    r_value, _ = pearsonr(vector1, vector2)
    return r_value


def set_outward_spines(axs):
    try:
        axs = axs.flatten()
    except BaseException:
        axs = [axs]
    for ax in axs:
        ax.spines["left"].set_position(
            ("outward", 2.5)
        )  # 10 points outward for y-axis
        ax.spines["bottom"].set_position(
            ("outward", 2.5)
        )  # 10 points outward for x-axis
    return axs


def convert_dict_to_array(dict1, dict2, key_names1, key_names2):
    data1 = np.concatenate(
        [
            np.concatenate([dict1[x_key] for x_key in key_names1], axis=0).reshape(
                -1, 1
            ),
            np.concatenate([dict1[y_key] for y_key in key_names2], axis=0).reshape(
                -1, 1
            ),
        ],
        axis=1,
    )

    data2 = np.concatenate(
        [
            np.concatenate([dict2[x_key] for x_key in key_names1], axis=0).reshape(
                -1, 1
            ),
            np.concatenate([dict2[y_key] for y_key in key_names2], axis=0).reshape(
                -1, 1
            ),
        ],
        axis=1,
    )

    return data1, data2


def calculate_kde(data1, data2, binsize=100):
    """Calculate the KDE of two datasets and return the values."""
    from scipy.stats import gaussian_kde

    kde1 = gaussian_kde(data1.T)
    kde2 = gaussian_kde(data2.T)

    # Now we need to define the range over which we'll integrate
    x_min = min(data1[:, 0].min(), data2[:, 0].min())
    x_max = max(data1[:, 0].max(), data2[:, 0].max())
    y_min = min(data1[:, 1].min(), data2[:, 1].min())
    y_max = max(data1[:, 1].max(), data2[:, 1].max())

    # Generate a grid of values
    x_values = np.linspace(x_min, x_max, binsize)
    y_values = np.linspace(y_min, y_max, binsize)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid_coordinates = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Evaluate both KDEs on the grid
    kde1_values = kde1(grid_coordinates).reshape(binsize, binsize)
    kde1_values = kde1_values / kde1_values.max()

    kde2_values = kde2(grid_coordinates).reshape(binsize, binsize)
    kde2_values = kde2_values / kde2_values.max()

    return kde1_values, kde2_values, x_grid, y_grid


def calculate_overlap(hist1, hist2, threshold=0.1):
    """Calculate the overlap of two histograms."""
    overlap = np.count_nonzero(np.minimum(hist1, hist2) > threshold)
    union = np.count_nonzero(np.maximum(hist1, hist2) > threshold)

    return 100 * overlap / union


def plot_2d_histograms_3x3(data_all, figsize=(4, 4.7), titles=[""], export_path=None):
    """Plot the 2D histograms."""
    if not titles:
        titles = ["Intact", "Experiment", "Overlap", "2D occupancy histograms"]

    # Create the figure with constrained layout to avoid overlap
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Define grid spec for the figure
    gs = fig.add_gridspec(
        4, 6, height_ratios=[1, 1, 1, 0.05], width_ratios=[1, 1] * 3
    )
    ############################################
    # XY
    hist1, hist2, overlap, x_values, y_values, xlims, ylims = data_all["xy"]
    hist1 = 100 * hist1 / hist1.max()
    hist2 = 100 * hist2 / hist2.max()

    # Setup for first plot
    ax11 = fig.add_subplot(gs[0, 0:2])
    pcm = ax11.pcolormesh(
        x_values, y_values, hist1, cmap=plt.colormaps["Blues"], rasterized=True
    )
    # Setup for second plot
    ax12 = fig.add_subplot(gs[0, 2:4])
    pcm2 = ax12.pcolormesh(
        x_values, y_values, hist2, cmap=plt.colormaps["Reds"], rasterized=True
    )
    # Setup for third plot with two colorbar, label="Occupancy (%)"s
    ax13 = fig.add_subplot(gs[0, 4:6])
    # First color mesh and colorbar for kde1_value, label="Occupancy (%)"s
    pcm3 = ax13.pcolormesh(
        x_values,
        y_values,
        hist1,
        cmap=plt.colormaps["Blues"],
        rasterized=True,
        alpha=0.5,
    )
    pcm3 = ax13.pcolormesh(
        x_values,
        y_values,
        hist2,
        cmap=plt.colormaps["Reds"],
        rasterized=True,
        alpha=0.5,
    )
    ax13.text(
        1.05,
        0.5,
        f"{overlap:.2f}%",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax13.transAxes,
        # rotate
        rotation=270,
    )
    # Determine the limits
    for i, ax in enumerate([ax11, ax12, ax13]):
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(titles[i])
    ############################################

    # XZ
    hist1, hist2, overlap, x_values, y_values, xlims, ylims = data_all["xz"]
    hist1 = 100 * hist1 / hist1.max()
    hist2 = 100 * hist2 / hist2.max()
    # Setup for first plot
    ax21 = fig.add_subplot(gs[1, 0:2])
    pcm = ax21.pcolormesh(
        x_values, y_values, hist1, cmap=plt.colormaps["Blues"], rasterized=True
    )
    # Setup for second plot
    ax22 = fig.add_subplot(gs[1, 2:4])
    pcm2 = ax22.pcolormesh(
        x_values, y_values, hist2, cmap=plt.colormaps["Reds"], rasterized=True
    )
    # Setup for third plot with two colorbar, label="Occupancy (%)"s
    ax23 = fig.add_subplot(gs[1, 4:6])
    # First color mesh and colorbar for kde1_value, label="Occupancy (%)"s
    pcm3 = ax23.pcolormesh(
        x_values,
        y_values,
        hist1,
        cmap=plt.colormaps["Blues"],
        rasterized=True,
        alpha=0.5,
    )
    pcm3 = ax23.pcolormesh(
        x_values,
        y_values,
        hist2,
        cmap=plt.colormaps["Reds"],
        rasterized=True,
        alpha=0.5,
    )
    # write on the ax23 the overlap
    ax23.text(
        1.05,
        0.5,
        f"{overlap:.2f}%",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax23.transAxes,
        # rotate
        rotation=270,
    )
    # # Determine the limits
    for ax in [ax21, ax22, ax23]:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    ############################################
    # YZ
    hist1, hist2, overlap, x_values, y_values, xlims, ylims = data_all["yz"]
    hist1 = 100 * hist1 / hist1.max()
    hist2 = 100 * hist2 / hist2.max()

    # Setup for first plot
    ax31 = fig.add_subplot(gs[2, 0:2])
    pcm = ax31.pcolormesh(
        x_values, y_values, hist1, cmap=plt.colormaps["Blues"], rasterized=True
    )
    cax31 = fig.add_subplot(gs[-1, 0:2])
    fig.colorbar(pcm, cax=cax31, orientation="horizontal", label="Occupancy (%)")

    # Setup for second plot
    ax32 = fig.add_subplot(gs[2, 2:4])
    pcm2 = ax32.pcolormesh(
        x_values, y_values, hist2, cmap=plt.colormaps["Reds"], rasterized=True
    )
    cax32 = fig.add_subplot(gs[-1, 2:4])
    fig.colorbar(pcm2, cax=cax32, orientation="horizontal", label="Occupancy (%)")

    # Setup for third plot with two colorbar, label="Occupancy (%)"s
    ax33 = fig.add_subplot(gs[2, 4:6])
    # First color mesh and colorbar for kde1_value, label="Occupancy (%)"s
    pcm3 = ax33.pcolormesh(
        x_values,
        y_values,
        hist1,
        cmap=plt.colormaps["Blues"],
        rasterized=True,
        alpha=0.5,
    )
    # cax33 = fig.add_subplot(gs[-1, -2])
    # fig.colorbar(pcm3, cax=cax33, orientation="horizontal", label=f"{titles[0][:4]}. (%)")
    # Second color mesh and colorbar for kde2_values, with corrected subplot
    # for colorba, label="Occupancy (%)"r
    pcm4 = ax33.pcolormesh(
        x_values,
        y_values,
        hist2,
        cmap=plt.colormaps["Reds"],
        rasterized=True,
        alpha=0.5,
        label="Occupancy (%)",
    )

    ax33.text(
        1.05,
        0.5,
        f"{overlap:.2f}%",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax33.transAxes,
        # rotate
        rotation=270,
    )

    # cax33_right = fig.add_subplot(gs[-1, -1])
    # fig.colorbar(
    #     pcm4,
    #     cax=cax33_right,
    #     orientation="horizontal",
    #     label=f"{titles[1]} (%)",
    # )
    # # Determine the limits
    for ax in [ax31, ax32, ax33]:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel("Joint positions (mm)")

    # labels
    ax21.set_ylabel("Joint positions (mm)")

    # # set xticks with one decimal
    ax12.set_yticklabels([])
    ax13.set_yticklabels([])
    ax22.set_yticklabels([])
    ax32.set_yticklabels([])
    ax23.set_yticklabels([])
    ax33.set_yticklabels([])

    # plt.tight_layout(pad=-0.05)
    if export_path is not None:
        plt.savefig(export_path, dpi=300, bbox_inches="tight")

    plt.show()


def get_contact_forces(contact_array, collision_pair_idx, get_vector=False):
    """
    Get the contact forces for the given collision pair index
    """
    contact_forces = contact_array[:, collision_pair_idx, 6:9]
    if get_vector:
        return np.sum(contact_forces, axis=1)
    else:
        return np.linalg.norm(np.sum(contact_forces, axis=1), axis=1)


def plot_legs_collision_diagram_grooming(
    contact_data,
    collision_pairs,
    time_step=1e-4,
    ax=None,
    export_path=None,
    bar_width=0.7,
    title="",
    alpha=0.75,
):
    """
    Plots the collision diagram for grooming.
    Reverse if collision detected in the reversed order.

    Example:
        fig, ax = plt.subplotsfigsize=(7, 3))
        contact_normal = load_physics_data('./simulation_results/kinematic_replay_grooming_220216_144426', 'contact_flag')
        plot_collision_diagram_grooming(contact_normal, ax=ax, title='No concave')
        plt.tight_layout()
        plt.show()
    """
    total_length = contact_data.shape[0] * time_step

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 1))

    antennal_seg_colors = {"RAntenna": "#56B4E9", "LAntenna": "#E69F00"}
    leg_names = ["RFLeg", "LFLeg"]

    for i, (ant_segment, leg_segment) in enumerate(collision_pairs):
        index = leg_names.index(leg_segment)
        # print(f'indices between {ant_segment} and {leg_segment} are {collision_pairs[(ant_segment, leg_segment)]}')

        collision = get_collision_data(
            contact_data, collision_pairs[(ant_segment, leg_segment)]
        )
        color = antennal_seg_colors[ant_segment]
        #         print(ant, leg_segment)

        intervals = (
            np.where(np.abs(np.diff(collision, prepend=[0], append=[0])) == 1)[
                0
            ].reshape(-1, 2)
            * time_step
            * 100
        )
        intervals[:, 1] = intervals[:, 1] - intervals[:, 0]

        ax.broken_barh(
            intervals,
            (index - bar_width * 0.5, bar_width),
            facecolor=color,
            alpha=alpha,
        )

    ax.set_yticks((0, 1))
    ax.set_yticklabels(leg_names)

    ax.set_ylim(1.5, -0.5)

    if title:
        ax.set_title(title, fontsize=15)

    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")


def relu_function(x, threshold=0):
    return np.maximum(x, threshold)


def load_npz(file_path):
    return np.load(file_path, allow_pickle=True)["array"].item().toarray()


def sort_dictionary_by_values(dictionary):
    return dict(
        sorted(
            #  Reverse is true for ascending order
            dictionary.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )


def load_data(output_fname):
    # Directly creates a dataframe from the pickle files
    with open(output_fname, "rb") as f:
        pts = pickle.load(f)
    return pts


def save_data(output_fname, output_file):
    # Directly creates a dataframe from the pickle files
    with open(output_fname, "wb") as f:
        pickle.dump(output_file, f)


def load_csv(output_fname):
    return pd.read_csv(output_fname)


def get_data_from_df(data_frame, fly_id, date, trial_no):
    return data_frame.query("Fly == @fly_id & Date == @date & Trial == @trial_no")


def lighten_color(color, amount=0.3):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    import colorsys

    try:
        c = mcolors.cnames[color]
    except BaseException:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return mcolors.to_hex(colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]))


def plot_joint_angle(
    kinematics_data,
    ax=None,
    degrees=True,
    xlims=None,
    ylims=None,
    legend=False,
    diff=False,
    cols=None,
):
    """[summary]

    Parameters
    ----------
    kinematics_data : [type]
        [description]
    leg : str
        Name of the leg. e.g., RF
    ax : [type], optional
        [description], by default None
    """
    colors = get_color_list("Set1", 10)

    stim = np.array(kinematics_data["Stimulus"])
    stim_interval = get_stim_intervals(stim)

    time_ = np.array(kinematics_data["Time"])
    if cols is None:
        cols = [
            "Angle_head_roll",
            "Angle_head_pitch",
            "Angle_antenna_pitch_L",
            "Angle_antenna_pitch_R",
            "Pose_LF_Tarsus_y",
            "Pose_RF_Tarsus_y",
            "Pose_LF_Tarsus_z",
            "Pose_RF_Tarsus_z",
        ]
    pos_data = kinematics_data.loc[:, cols]

    if ax is None:
        ax = plt.gca()

    for i, (joint_name, joint_angles) in enumerate(pos_data.items()):
        label = " ".join((joint_name.split("_")[-2], joint_name.split("_")[-1]))

        if diff:
            joint_angles = np.diff(joint_angles)
            time = time_[:-1]
        else:
            time = time_

        if degrees and not "Pose" in joint_name:
            ax.plot(
                time, np.array(joint_angles) * 180 / np.pi, label=label, color=colors[i]
            )
        elif "Pose" in joint_name:
            ax.plot(time, np.array(joint_angles) * 20, label=label, color=colors[i])
        else:
            ax.plot(time, np.array(joint_angles), label=label, color=colors[i])

        for j in np.arange(0, len(stim_interval), 2):
            ax.axvspan(
                time[stim_interval[j]],
                time[stim_interval[j + 1]],
                alpha=0.03,
                color="red",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if legend:
        ax.legend(bbox_to_anchor=(1.5, 0.5), frameon=False)

    if xlims is not None:
        ax.set_xlim(xlims)

    if ylims is not None:
        ax.set_ylim(ylims)


def get_stim_intervals(stim_data):
    """Reads stimulus array and returns the stim intervals for plotting purposes.
    Use get_stim_array otherwise."""
    stim_on = np.where(stim_data)[0]
    stim_start_end = [stim_on[0]]
    for ind in list(np.where(np.diff(stim_on) > 1)[0]):
        stim_start_end.append(stim_on[ind])
        stim_start_end.append(stim_on[ind + 1])

    if not stim_on[-1] in stim_start_end:
        stim_start_end.append(stim_on[-1])
    return stim_start_end


def get_color_list(cmap="viridis", number=5):
    """Returns the color codes from a given color map."""
    return mcp.gen_color(cmap=cmap, n=number)


def convert_dict_to_dataframe(data_dict):
    """
    Convert a nested dictionary to a pandas DataFrame.

    Parameters:
    data_dict (dict): A nested dictionary with structure:
                      { "key": { "tita_y": [values], "label": [labels] }, ... }

    Returns:
    pd.DataFrame: A DataFrame with columns 'tita_y' and 'label'.
    """
    # Create a list to hold DataFrames for each key in the dictionary
    df_list = []

    # Iterate over each key-value pair in the dictionary
    for key, data in data_dict.items():
        # Convert each inner dictionary to a DataFrame and append to the list
        df_list.append(pd.DataFrame(data))

    # Concatenate all DataFrames in the list into a single DataFrame
    return pd.concat(df_list, ignore_index=True)


# tested


def hdf5_to_dict(filename, **kwargs):
    """HDF5 to dictionary
    Taken from farms_core.io.hdf5.py
    """
    data = {}
    hfile = hdf5_open(filename, mode="r", **kwargs)
    _hdf5_to_dict(hfile, data)
    hfile.close()
    return data


def hdf5_open(filename, mode="w", max_attempts=10, attempt_delay=0.1):
    """Open HDF5 file with delayed attempts
    Taken from farms_core.io.hdf5.py

    """
    for attempt in range(max_attempts):
        try:
            hfile = h5py.File(name=filename, mode=mode)
            break
        except OSError as err:
            if attempt == max_attempts - 1:
                print(
                    "File %s was locked during more than %s [s]",
                    filename,
                    max_attempts * attempt_delay,
                )
                raise err
            print(
                "File %s seems locked during attempt %s/%s",
                filename,
                attempt + 1,
                max_attempts,
            )
            time.sleep(attempt_delay)
    return hfile


def _hdf5_to_dict(handler, dict_data):
    """HDF5 to dictionary
    Taken from farms_core.io.hdf5.py
    """
    for key, value in handler.items():
        if isinstance(value, h5py.Group):
            new_dict = {}
            dict_data[key] = new_dict
            _hdf5_to_dict(value, new_dict)
        else:
            if value.shape:
                if value.dtype == np.dtype("O"):
                    if len(value.shape) == 1:
                        data = [val.decode("utf-8") for val in value]
                    elif len(value.shape) == 2:
                        data = [
                            tuple(val.decode("utf-8") for val in values)
                            for values in value
                        ]
                    else:
                        raise Exception(f"Cannot handle shape {value.shape}")
                else:
                    data = np.array(value)
            elif value.shape is not None:
                data = np.array(value).item()
            else:
                data = None
            dict_data[key] = data


def contact_duration(contact_array, collision_idx, collision_pair):
    return np.count_nonzero(
        get_collision_data(contact_array, collision_idx[collision_pair])
    )


def get_collision_pairs(contact_data):
    collision_pairs_sim = contact_data["sensors"]["contacts"]["names"]

    # From collision pairs, take the indices that contain the links of interest
    collision_pairs_to_plot = [
        ("LPedicel", "LFTibia"),
        ("LPedicel", "LFTarsus"),
        ("LFuniculus", "LFTibia"),
        ("LFuniculus", "LFTarsus"),
        ("LArista", "LFTibia"),
        ("LArista", "LFTarsus"),
        ("RPedicel", "RFTibia"),
        ("RPedicel", "RFTarsus"),
        ("RFuniculus", "RFTibia"),
        ("RFuniculus", "RFTarsus"),
        ("RArista", "RFTibia"),
        ("RArista", "RFTarsus"),
        ("RPedicel", "LFTibia"),
        ("RPedicel", "LFTarsus"),
        ("RFuniculus", "LFTibia"),
        ("RFuniculus", "LFTarsus"),
        ("RArista", "LFTibia"),
        ("RArista", "LFTarsus"),
        ("LPedicel", "RFTibia"),
        ("LPedicel", "RFTarsus"),
        ("LFuniculus", "RFTibia"),
        ("LFuniculus", "RFTarsus"),
        ("LArista", "RFTibia"),
        ("LArista", "RFTarsus"),
    ]

    collision_pairs = {}

    for pair in collision_pairs_to_plot:
        collision_pairs[pair] = [
            i
            for i, links in enumerate(collision_pairs_sim)
            if any([pair[0] in link for link in links])
            and any([pair[1] in link for link in links])
        ]

    return collision_pairs


def calculate_centrality_scores(graph):
    """Calculate the centrality scores for a given graph."""
    centrality_scores = {}

    centrality_scores["degree_centrality"] = nx.degree_centrality(graph)
    centrality_scores["in_degree_centrality"] = nx.in_degree_centrality(graph)
    centrality_scores["out_degree_centrality"] = nx.out_degree_centrality(graph)
    centrality_scores["betweenness_centrality"] = nx.betweenness_centrality(
        graph, weight="weight"
    )
    try:
        centrality_scores["eigenvector_centrality"] = nx.eigenvector_centrality(
            graph, weight="weight", max_iter=1000
        )
    except BaseException:
        centrality_scores["eigenvector_centrality"] = nx.eigenvector_centrality(graph)

    centrality_scores["closeness_centrality"] = nx.closeness_centrality(graph)
    # centrality_scores["pagerank"] = nx.pagerank(graph, weight="weight")

    return centrality_scores


def reduce_collision_pairs_ant(collision_pairs):
    antenna_segments = ["Pedicel", "Funiculus", "Arista"]
    reduced_collision_pairs = {}

    for pair in collision_pairs:
        if pair[0][1:] in antenna_segments:
            side_first = pair[0][0]
            new_pair = (f"{side_first}Antenna", pair[1])

            if new_pair not in reduced_collision_pairs:
                reduced_collision_pairs[new_pair] = []

            reduced_collision_pairs[new_pair] += collision_pairs[pair]

    return reduced_collision_pairs


def reduce_collision_pairs_ant_leg(collision_pairs):
    antenna_segments = ["Pedicel", "Funiculus", "Arista"]
    leg_segments = ["Tarsus", "Tibia"]
    reduced_collision_pairs = {}

    for pair in collision_pairs:
        if (pair[0][1:] in antenna_segments) and (pair[1][2:] in leg_segments):
            side_first = pair[0][0]
            side_second = pair[1][0]
            new_pair = (f"{side_first}Antenna", f"{side_second}FLeg")

            if new_pair not in reduced_collision_pairs:
                reduced_collision_pairs[new_pair] = []

            reduced_collision_pairs[new_pair] += collision_pairs[pair]

    return reduced_collision_pairs


def get_collision_data(collision, ind):
    """Returns the collision array."""

    if len(ind) > 1:
        collision_return = np.where(
            np.sum(np.sum(collision[:, ind, -3:], axis=1), axis=1), 1, 0
        )
    else:
        collision_return = np.where(np.sum(collision[:, ind, -3:], axis=2), 1, 0)
    return np.squeeze(collision_return)


def plot_reduced_collision_diagram_grooming(
    contact_data,
    collision_pairs,
    time_step=1e-4,
    ax=None,
    export_path=None,
    bar_width=0.7,
    title="",
    alpha=0.75,
):
    """
    Plots the collision diagram for grooming.
    Reverse if collision detected in the reversed order.

    Example:
        fig, ax = plt.subplotsfigsize=(7, 3))
        contact_normal = load_physics_data('/Users/ozdil/Desktop/GIT/neuromechfly-grooming/scripts/kinematic_replay/simulation_results/kinematic_replay_grooming_220216_144426', 'contact_flag')
        plot_collision_diagram_grooming(contact_normal, ax=ax, title='No concave')
        plt.tight_layout()
        plt.show()
    """
    total_length = contact_data.shape[0] * time_step

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 1))

    ANTENNA_SEGMENTS = ["RAntenna", "LAntenna"]

    LEG_SEG_COLORS = {
        "RFTibia": "#0073B2",
        "RFLeg": "#0073B2",
        "RFTarsus": "#00354C",
        "LFTibia": "#D36027",
        "LFLeg": "#D36027",
        "LFTarsus": "#5E2912",
    }

    for i, (ant, leg_segment) in enumerate(collision_pairs):

        index = ANTENNA_SEGMENTS.index(ant)

        collision = get_collision_data(
            contact_data, collision_pairs[(ant, leg_segment)]
        )
        color = LEG_SEG_COLORS[leg_segment]
        #         print(ant, leg_segment)

        intervals = (
            np.where(np.abs(np.diff(collision, prepend=[0], append=[0])) == 1)[
                0
            ].reshape(-1, 2)
            * time_step
        )
        intervals[:, 1] = intervals[:, 1] - intervals[:, 0]

        ax.broken_barh(
            intervals,
            (index - bar_width * 0.5, bar_width),
            facecolor=color,
            alpha=alpha,
        )

    ax.set_yticks((0, 1))
    ax.set_yticklabels(ANTENNA_SEGMENTS)

    ax.set_ylim(1.5, -0.5)

    if title:
        ax.set_title(title, fontsize=15)

    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")


def get_edge_weight(
    graph, nodes_pre, nodes_post, edge_name="weight", take_abs=False, use_pandas=True
):
    # FIXME I dont like how this is written
    if not isinstance(nodes_pre, list):
        nodes_pre = [nodes_pre]
    if not isinstance(nodes_post, list):
        nodes_post = [nodes_post]

    weights = []
    for node_pre in nodes_pre:
        for node_post in nodes_post:
            edges = graph.get_edge_data(node_pre, node_post)
            if edges is not None:
                if use_pandas:
                    # print(edges)
                    if take_abs:
                        weights.append(abs(edges[edge_name]))
                    else:
                        weights.append(edges[edge_name])
                else:
                    if take_abs:
                        weights.extend(
                            [abs(edge[edge_name]) for _, edge in edges.items()]
                        )
                    else:
                        weights.extend([edge[edge_name] for _, edge in edges.items()])

    return sum(weights)


def get_graph_voltage(
    voltage,
    neurons_ordered,
    cluster2neuron,
    node2name,
):
    """Get the population neural activity from single neuron recordings.
    Population is defined by the `neuronal clusters` in the graph.

    Parameters
    ----------
    voltage : np.ndarray
        Single neuron voltage recordings.
    neurons_ordered : list
        Ordered list of neurons.
    cluster2neuron : dict
        Dictionary of cluster to neuron mapping.
    node2name : dict
        Dictionary of node to name mapping.

    Returns
    -------
    np.ndarray
        Population neural activity given by the average neural
        activity of the neurons in the cluster.
    """

    graph_voltage = np.zeros((voltage.shape[0], len(node2name)))

    for node, node_name in node2name.items():
        neuron_idx = [
            neurons_ordered.index(neuron + node_name[-2:])
            for neuron in cluster2neuron[node_name[:-2].replace("c_", "cluster_")]
        ]
        graph_voltage[:, node] = np.mean(voltage[:, neuron_idx], axis=1)

    return graph_voltage
