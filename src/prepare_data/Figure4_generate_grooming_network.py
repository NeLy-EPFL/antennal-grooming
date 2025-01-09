""" Module to identify neurons that are monosynaptically connected to the antennal grooming neurons. """

import pickle
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import Figure4_neurons as neurons


def get_connectivity_btw_neurons_hairy(
    connectivity_df: pd.DataFrame,
    neurons_of_interest1: List,
    neurons_of_interest2: List = None,
    col_name1: str = "pre_root_id",
    col_name2: str = "post_root_id",
) -> pd.DataFrame:
    """
    Gets the connectivity of neurons of interest, including connections
    to/from other neurons that are not in the list.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        Table containing connections between two neurons
        (must include col_name1 and col_name2 as columns).
    neurons_of_interest1 : List
        List of neuron segment IDs of interest for the presynaptic partner.
    neurons_of_interest2 : List, optional
        List of neuron segment IDs of interest for the postsynaptic partner.
        If None, defaults to neurons_of_interest1.
    col_name1 : str, optional
        Column name of presynaptic neurons, by default 'pre_root_id'.
    col_name2 : str, optional
        Column name of postsynaptic neurons, by default 'post_root_id'.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only connections where at least one
        side of the connection is a neuron of interest.
    """
    if neurons_of_interest2 is None:
        neurons_of_interest2 = neurons_of_interest1

    connectivity_df = connectivity_df[
        connectivity_df[col_name1].isin(neurons_of_interest1)
        | connectivity_df[col_name2].isin(neurons_of_interest2)
    ]
    return connectivity_df.copy()


def merge_sensory_neurons(connectivity_df, sensory_neurons_dict):
    """
    Treat all small sensory neurons as one neuron by giving them the same ID.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        DataFrame of connectivity (pre, post, nt_type, syn_count).
    sensory_neurons_dict : dict
        Dictionary mapping a 'representative' sensory neuron ID to
        multiple smaller IDs.

    Returns
    -------
    pd.DataFrame
        Modified connectivity DataFrame with small sensory neurons replaced
        by their representative ID. Synaptic counts are merged accordingly.
    """
    for sn in sensory_neurons_dict:
        # Find all matching SENSORY_NEURONS that have the same suffix/prefix
        sensory_neuron_ids = [
            id_dict["id"]
            for neuron, id_dict in neurons.SENSORY_NEURONS.items()
            if (sn[-1] == neuron[-1]) and (sn[:4] == neuron[:4])
        ]
        connectivity_df = connectivity_df.replace(
            sensory_neuron_ids, sensory_neurons_dict[sn]["id"]
        )

    new_connectivity_df = connectivity_df.groupby(
        ["pre_root_id", "post_root_id", "nt_type"], as_index=False
    )["syn_count"].sum()

    return new_connectivity_df


def get_total_syn(connectivity_df, segment_id, col_name="post_root_id"):
    """
    Get total synapses onto or from a given neuron.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        DataFrame of connectivity.
    segment_id : int
        The segment ID of the neuron in question.
    col_name : str, optional
        Column to filter by, either 'pre_root_id' or 'post_root_id'.

    Returns
    -------
    int
        The total synaptic count onto or from the specified neuron.
    """
    segment_df = connectivity_df[connectivity_df[col_name] == segment_id].reset_index()
    return segment_df["syn_count"].sum()


def get_syn_w_neurons(
    connectivity_df,
    segment_id,
    segment_ids_of_interest,
    col_name="post_root_id",
):
    """
    Get total synapses between one neuron (segment_id) and a set of neurons of interest.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        DataFrame of connectivity.
    segment_id : int
        The segment ID of the neuron in question.
    segment_ids_of_interest : List[int]
        Neuron segment IDs of interest to filter on.
    col_name : str, optional
        Column to filter by, either 'pre_root_id' or 'post_root_id'.

    Returns
    -------
    int
        The total synapses between segment_id and segment_ids_of_interest.
    """
    second_col_name = "pre_root_id" if col_name == "post_root_id" else "post_root_id"
    segment_df = connectivity_df[
        (connectivity_df[col_name] == segment_id)
        & (connectivity_df[second_col_name].isin(segment_ids_of_interest))
    ].reset_index()
    return segment_df["syn_count"].sum()


def get_info_percentage(
    connectivity_df,
    segment_id,
    segment_ids_of_interest,
    verbose=False,
):
    """
    Calculate what percentage of a neuron's input and output synapses
    come from / go to neurons of interest.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        DataFrame containing connectivity info.
    segment_id : int
        Neuron segment ID to evaluate.
    segment_ids_of_interest : List[int]
        List of neuron IDs considered 'of interest'.
    verbose : bool, optional
        If True, prints debug info to stdout.

    Returns
    -------
    tuple of float
        (percentage_in, percentage_out) : % input from / % output to segment_ids_of_interest
    """
    total_in = get_total_syn(connectivity_df, segment_id, "post_root_id")
    grooming_in = get_syn_w_neurons(
        connectivity_df,
        segment_id,
        segment_ids_of_interest,
        "post_root_id",
    )

    total_out = get_total_syn(connectivity_df, segment_id, "pre_root_id")
    grooming_out = get_syn_w_neurons(
        connectivity_df,
        segment_id,
        segment_ids_of_interest,
        "pre_root_id",
    )

    perc_in = (grooming_in / (1e-6 + total_in)) * 100
    perc_out = (grooming_out / (1e-6 + total_out)) * 100

    if verbose:
        print("*******************")
        print(f"Segment ID: {segment_id}")
        print(
            f"This neuron takes {total_in} synapses, {grooming_in} of which come from grooming "
            f"(~{perc_in:.2f}% of total input)."
        )
        print(
            f"This neuron outputs {total_out} synapses, {grooming_out} of which go to grooming "
            f"(~{perc_out:.2f}% of total output)."
        )
        print("*******************")

    return perc_in, perc_out


def get_relevant_grooming_neurons(
    connectivity_df,
    classification_df,
    segment_ids,
    grooming_segment_ids,
    threshold,
    export_path=None,
    verbose=False,
):
    """
    Identify neurons that are relevant based on their percentage
    of input/output connections to grooming neurons.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        DataFrame containing connectivity info.
    classification_df : pd.DataFrame
        DataFrame classifying each neuron (root_id, super_class, etc.).
    segment_ids : List[int]
        List of segment IDs to evaluate.
    grooming_segment_ids : List[int]
        Known grooming-related segment IDs.
    threshold : float
        If input or output grooming percentage is above this threshold,
        the neuron is considered 'relevant'.
    export_path : str or Path, optional
        If provided, saves results to a pickle file in this directory.
    verbose : bool, optional
        If True, prints debug statements.

    Returns
    -------
    dict
        Dictionary with keys 'IN', 'DN', 'SN', 'other', each containing
        {segment_id: (perc_in, perc_out)} of identified neurons.
    """
    store_data_hairy_DN = {}
    store_data_hairy_IN = {}
    store_data_hairy_SN = {}
    store_data_others = {}

    for seg_id in tqdm(segment_ids, disable=False):
        if isinstance(seg_id, int):
            neuron_class = classification_df[
                classification_df["root_id"] == seg_id
            ]["super_class"].values[0]

            perc_in, perc_out = get_info_percentage(
                connectivity_df, seg_id, grooming_segment_ids, verbose
            )

            if neuron_class in ["central"] and (perc_in > threshold and perc_out > threshold):
                store_data_hairy_IN[seg_id] = (perc_in, perc_out)
            elif neuron_class in ["descending", "motor"] and (perc_in > threshold):
                store_data_hairy_DN[seg_id] = (perc_in, perc_out)
            elif neuron_class in ["sensory"] and (perc_out > threshold):
                store_data_hairy_SN[seg_id] = (perc_in, perc_out)
            elif perc_in > threshold and perc_out > threshold:
                store_data_others[seg_id] = (perc_in, perc_out)
            else:
                continue

    result_dict = {
        "IN": store_data_hairy_IN,
        "DN": store_data_hairy_DN,
        "SN": store_data_hairy_SN,
        "other": store_data_others,
    }

    if export_path is not None:
        save_path = Path(export_path) / f"threshold_{threshold}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(result_dict, f)

    return result_dict


def run_in_parallel(
    connectivity_df,
    classification_df,
    segment_ids,
    grooming_segment_ids,
    threshold_vals,
    export_path=None,
):
    """
    Execute get_relevant_grooming_neurons in parallel for multiple threshold values.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        DataFrame containing connectivity info.
    classification_df : pd.DataFrame
        DataFrame classifying each neuron (root_id, super_class, etc.).
    segment_ids : List[int]
        List of segment IDs to evaluate.
    grooming_segment_ids : List[int]
        Known grooming-related segment IDs.
    threshold_vals : iterable
        Collection of thresholds to evaluate.
    export_path : str or Path, optional
        If provided, saves results to a pickle file for each threshold.

    Returns
    -------
    dict
        Dictionary keyed by threshold, with values being the result
        of get_relevant_grooming_neurons.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                get_relevant_grooming_neurons,
                connectivity_df,
                classification_df,
                segment_ids,
                grooming_segment_ids,
                threshold=th,
                export_path=export_path,
            ): th
            for th in threshold_vals
        }
        for future in as_completed(futures):
            threshold_used = futures[future]
            results[threshold_used] = future.result()

    return results


if __name__ == "__main__":
    # Adjust paths as needed
    DATA_PATH = Path("../../data")
    EXPORT_PATH = Path("../../data")

    if not EXPORT_PATH.is_dir():
        EXPORT_PATH.mkdir(parents=True)

    # Load the entire brain connectome
    connectivity_all_merged = pd.read_parquet(DATA_PATH / "FAFB_connections_sensory_merged.parquet")

    # Source neurons - part of the foundational network
    source_neurons = (
        [neuron["id"] for _, neuron in neurons.neuron_dict["sensory_neurons"].items()]
        + [neuron["id"] for _, neuron in neurons.neuron_dict["ABN"].items()]
        + [neuron["id"] for _, neuron in neurons.neuron_dict["ADN"].items()]
    )

    # Target neurons - part of the foundational network
    target_neurons = [
        neuron["id"] for _, neuron in neurons.MOTOR_NEURONS.items()
    ] + source_neurons

    # Filter connectivity
    connectivity_from_source2target_merged = get_connectivity_btw_neurons_hairy(
        connectivity_df=connectivity_all_merged,
        neurons_of_interest1=source_neurons,
        neurons_of_interest2=target_neurons,
        col_name1="pre_root_id",
        col_name2="post_root_id",
    )

    # Gather unique segment IDs
    all_segment_ids = set(
        connectivity_from_source2target_merged["pre_root_id"].tolist()
        + connectivity_from_source2target_merged["post_root_id"].tolist()
    )
    # Remove source and target from the unique list
    grooming_segment_ids = target_neurons
    unique_segment_ids = all_segment_ids.difference(grooming_segment_ids)
    assert all(x in all_segment_ids for x in grooming_segment_ids)

    print("Starting the process...")

    # classification_table should be defined or imported
    # (ensure that it is a DataFrame with ['root_id', 'super_class'] columns)
    result = run_in_parallel(
        connectivity_all_merged,
        classification_table,  # Ensure this is defined elsewhere
        unique_segment_ids,
        grooming_segment_ids,
        threshold_vals=np.arange(0, 102.5, 2.5),
        export_path=EXPORT_PATH,
    )

    with open(EXPORT_PATH / "Fig4_parameter_sweep.pkl", "wb") as f:
        pickle.dump(result, f)

    for threshold_val, threshold_result in result.items():
        print(threshold_val)
        print("----------------")
        for class_key, neuron_dict in threshold_result.items():
            print(f"{class_key} - number of neurons found: {len(neuron_dict)}")
