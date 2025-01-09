""" Code to process the data for Figure 3. """

import os
import logging
import itertools
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

# Disable matplotlib logger
logging.getLogger("matplotlib.font_manager").disabled = True

# Paths
DATA_PATH = Path("../data")

# Constants
CHUNK_LENGTH, STRIDE = 30, 25
COLOR_BEHAVIORS = {
    "Intact R": "#8e0152",
    "Head-fixed R": "#fe92d0",
    "Intact L": "#276419",
    "Head-fixed L": "#b0e8a3",
}


def is_antenna_pitching(ant_right, ant_left, threshold=-30):
    """Check if the antenna is pitching by checking if the median of antennal pitch is above a threshold"""
    # print(np.median(ant_right), np.median(ant_left))
    return np.median(ant_right) > threshold and np.median(ant_left) > threshold


def is_head_pitching(head_pitch, threshold):
    """Check if the mean of head pitch is above the threshold."""
    return not has_median_above(
        head_pitch,
        threshold=threshold,
    )


def has_outliers(*args, threshold=0):
    # print([np.max(args) for arg in args])
    return any([np.max(arg) > threshold for arg in args])


def has_median_above(*args, threshold=0):
    # print([np.median(args) for arg in args])
    return any([np.median(arg) > threshold for arg in args])


def has_median_below(*args, threshold=0):
    # print([np.median(args) for arg in args])
    return any([np.median(arg) < threshold for arg in args])


def is_diff_above_threshold(data_one, data_two, threshold=9):
    # print(np.abs(np.median(data_one - data_two)))
    return np.abs(np.median(data_one - data_two)) > threshold


def divide_data_into_chunks_sliding_window(data, chunk_length, stride):
    """Divide the data into chunks of length chunk_length with a sliding window of step_size
     stride 2 and window length 10 means 8 overlap
    Divides the behavioral labels and returns the most repeated behavior in a chunk.
    """
    n_frames, n_bodyparts = data.shape

    n_chunks = len(range(0, n_frames - chunk_length + 1, stride))
    X = np.empty((n_chunks, chunk_length, n_bodyparts))

    count = 0
    for j in range(0, n_frames - chunk_length + 1, stride):
        X[count, ...] = data.values[j : j + chunk_length, :]
        count += 1

    return X


def divide_df_chunks(
    df: pd.DataFrame,
    chunk_length: int,
    stride: int,
    column_list: List[str],
    group_by: List[str] = ["Date", "Fly"],
) -> pd.DataFrame:
    """
    Iterate over the dataframe and divide it into chunks of length chunk_length with a sliding window of step_size
    """
    chunked_dict = {}
    for name, group in df.groupby(group_by):
        chunked_dict["_".join([str(n) for n in name])] = (
            divide_data_into_chunks_sliding_window(
                group.loc[:, column_list].reset_index(drop=True),
                chunk_length=chunk_length,
                stride=stride,
            )
        )

    return chunked_dict


def load_all_pds_filter(
    pkl_paths: List[Path],
    filter: bool = True,
) -> pd.DataFrame:
    """
    Load all the pandas dataframes from a path and divide them into chunks
    """
    df_all = []
    # append the dataframes in paths
    for i, pkl_p in enumerate(pkl_paths):
        df = pd.read_pickle(pkl_p).reset_index()
        df_all.append(df)

    df_all = pd.concat(df_all, ignore_index=True)
    if filter:
        # Filter the data
        df_filtered = filter_angle_pos(df_all, win_size=9).reset_index(drop=True)
        return df_filtered

    return df_all


# General checks
def check_criteria_hf(chunked_data, chunk_idx, cols, thresholds):
    return {
        "antenna_pitch": is_antenna_pitching(
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_R")],
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_L")],
            threshold=thresholds["antenna_pitch"],
        ),
        "has_outliers_antenna": has_outliers(
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_R")],
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_L")],
            threshold=thresholds["outlier_angle"],
        ),
        "has_outliers_tita": has_outliers(
            chunked_data[chunk_idx, :, cols.index("mean_TiTa_y")],
            threshold=thresholds["outlier_pos"],
        ),
    }


# Check which antenna pitched
def which_antenna_pitched(chunked_data, chunk_idx, cols, thresholds):
    if is_diff_above_threshold(
        chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_R")],
        chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_L")],
        threshold=thresholds["difference"],
    ):
        right_median = np.median(
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_R")]
        )
        left_median = np.median(
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_L")]
        )
        return "right" if right_median < left_median else "left"
    return None


def from_dict_to_iterable(dict_data, key_name="mean_TiTa_y_right"):
    data = []
    for fly_id in dict_data.keys():
        for angle_name, angle_list in dict_data[fly_id].items():
            if angle_list and key_name == angle_name:
                data.extend(np.array(angle_list))
            else:
                pass
                # print(f"No data for {fly_id} {angle_name}")
    return data


def get_chunks(df, exp_types, columns):
    """Get the chunks for the given experiment types and columns"""
    df_stimon = df[(df.Exp_Type.isin(exp_types)) & (df.Stimulus == True)].reset_index(
        drop=True
    )
    return divide_df_chunks(
        df_stimon,
        chunk_length=CHUNK_LENGTH,
        stride=STRIDE,
        column_list=columns,
        group_by=["Date", "Fly"],
    )


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


def prepare_data_panel_e(df_kinematics):
    """Prepares the kinematic data for panel E using the following steps:
    1- Divide the data into chunks
    2- For each experimental case (e.g., intact and head-fixed), first check if chunks contain any outliers, if not, then label each chunk based on the coordination between freely moving body parts.
    3- Gather the data for each condition and convert it into a dataframe.

    Parameters
    ----------
    df_kinematics : pd.DataFrame
        Dataframe containing the kinematic data of intact and experimental conditions.
    """

    threshold = {
        "antenna_pitch": -22,  # deg
        "outlier_angle": 0,  # deg
        "outlier_pos": 2,  # mm
        "difference": 6,  # deg
    }

    columns = [
        "Angle_antenna_pitch_R",
        "Angle_antenna_pitch_L",
        "mean_TiTa_y",
    ]

    # Mean tita y
    df_kinematics["mean_TiTa_y"] = df_kinematics[
        ["Pose_RF_Tarsus_y", "Pose_LF_Tarsus_y"]
    ].mean(axis=1)

    # Separate experimental and control groups
    chunks_intact = get_chunks(df_kinematics, ["Beh"], columns)
    chunks_hf = get_chunks(df_kinematics, ["HF", "HF_rest"], columns)

    # We will label each chunk based on the kinematic variables
    chunk_intact_beh_dict = {}
    for fly_id, chunked_data_intact in chunks_intact.items():
        chunk_intact_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_intact.shape[0]):
            if any(
                list(
                    check_criteria_hf(
                        chunked_data_intact, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            pitched_antenna = which_antenna_pitched(
                chunked_data_intact, chunk_no, columns, threshold
            )

            if pitched_antenna:
                for i, col in enumerate(columns):
                    chunk_intact_beh_dict[f"{fly_id}_beh"][
                        f"{col}_{pitched_antenna}"
                    ].extend(chunked_data_intact[chunk_no, :, i])

    chunk_hf_beh_dict = {}
    for fly_id, chunked_data_hf in chunks_hf.items():
        chunk_hf_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_hf.shape[0]):
            if any(
                list(
                    check_criteria_hf(
                        chunked_data_hf, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            pitched_antenna = which_antenna_pitched(
                chunked_data_hf, chunk_no, columns, threshold
            )

            if pitched_antenna:
                for i, col in enumerate(columns):
                    chunk_hf_beh_dict[f"{fly_id}_beh"][
                        f"{col}_{pitched_antenna}"
                    ].extend(chunked_data_hf[chunk_no, :, i])

    return chunk_intact_beh_dict, chunk_hf_beh_dict


def boxen_panel_e(intact_dict, exp_dict):
    def create_data(label, key, data_dict):
        data = from_dict_to_iterable(data_dict, key)
        return {"lateral_pos": data, "label": [label] * len(data)}

    data_beh = {
        "intact_r": create_data("Intact R", "mean_TiTa_y_right", intact_dict),
        "intact_l": create_data("Intact L", "mean_TiTa_y_left", intact_dict),
        "head_fixed_r": create_data("Head-fixed R", "mean_TiTa_y_right", exp_dict),
        "head_fixed_l": create_data("Head-fixed L", "mean_TiTa_y_left", exp_dict),
    }

    return convert_dict_to_dataframe(data_beh)


def scatter_panel_e(intact_dict, exp_dict):
    def median_positions(data_dict, key):
        return [
            np.median(np.array(data_dict[fly_id][key])) for fly_id in data_dict.keys()
        ]

    fly_data = {
        "intact": {
            "lateral_pos_ant_r": median_positions(intact_dict, "mean_TiTa_y_right"),
            "lateral_pos_ant_l": median_positions(intact_dict, "mean_TiTa_y_left"),
        },
        "hf": {
            "lateral_pos_ant_r": median_positions(exp_dict, "mean_TiTa_y_right"),
            "lateral_pos_ant_l": median_positions(exp_dict, "mean_TiTa_y_left"),
        },
    }

    return fly_data


def check_criteria_fleg_amp(chunked_data, chunk_idx, cols, thresholds):
    return {
        "antenna_pitch": is_antenna_pitching(
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_R")],
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_L")],
            threshold=thresholds["antenna_pitch"],
        ),
        "has_outliers_antenna": has_outliers(
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_R")],
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_L")],
            threshold=thresholds["outlier_angle"],
        ),
    }


def prepare_data_panel_f(df_kinematics):
    """Prepares the kinematic data for panel F using the following steps:
    1- Divide the data into chunks
    2- For each experimental case (e.g., intact and head-fixed), first check if chunks contain any outliers, if not, then label each chunk based on the coordination between freely moving body parts.
    3- Gather the data for each condition and convert it into a dataframe.

    Parameters
    ----------
    df_kinematics : pd.DataFrame
        Dataframe containing the kinematic data of intact and experimental conditions.
    """

    threshold = {
        "antenna_pitch": -22,  # deg
        "outlier_angle": 0,  # deg
        "difference": 6,  # deg
    }

    columns = [
        "Angle_head_roll",
        "Angle_head_pitch",
        "Angle_antenna_pitch_R",
        "Angle_antenna_pitch_L",
    ]

    # Separate experimental and control groups
    chunks_intact = get_chunks(df_kinematics, ["Beh"], columns)
    chunks_rlf = get_chunks(df_kinematics, ["RLF"], columns)

    # We will label each chunk based on the kinematic variables
    chunk_intact_beh_dict = {}

    for fly_id, chunked_data_intact in chunks_intact.items():
        chunk_intact_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }

        for chunk_no in range(chunked_data_intact.shape[0]):
            if any(
                list(
                    check_criteria_fleg_amp(
                        chunked_data_intact, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            pitched_antenna = which_antenna_pitched(
                chunked_data_intact, chunk_no, columns, threshold
            )

            if pitched_antenna:
                for i, col in enumerate(columns):
                    chunk_intact_beh_dict[f"{fly_id}_beh"][
                        f"{col}_{pitched_antenna}"
                    ].extend(chunked_data_intact[chunk_no, :, i])

    # Foreleg amputee
    chunk_rlf_beh_dict = {}
    for fly_id, chunked_data_rlf in chunks_rlf.items():
        chunk_rlf_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }

        for chunk_no in range(chunked_data_rlf.shape[0]):
            if any(
                list(
                    check_criteria_fleg_amp(
                        chunked_data_rlf, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            pitched_antenna = which_antenna_pitched(
                chunked_data_rlf, chunk_no, columns, threshold
            )

            if pitched_antenna:
                for i, col in enumerate(columns):
                    chunk_rlf_beh_dict[f"{fly_id}_beh"][
                        f"{col}_{pitched_antenna}"
                    ].extend(chunked_data_rlf[chunk_no, :, i])

    return chunk_intact_beh_dict, chunk_rlf_beh_dict


def boxen_panel_f(intact_dict, exp_dict):
    def create_data(label, key, data_dict):
        data = from_dict_to_iterable(data_dict, key)
        return {"head_roll": data, "label": [label] * len(data)}

    data_beh = {
        "intact_r": create_data("Intact R", "Angle_head_roll_right", intact_dict),
        "intact_l": create_data("Intact L", "Angle_head_roll_left", intact_dict),
        "rlf_r": create_data("RLF R", "Angle_head_roll_right", exp_dict),
        "rlf_l": create_data("RLF L", "Angle_head_roll_left", exp_dict),
    }

    return convert_dict_to_dataframe(data_beh)


def scatter_panel_f(intact_dict, exp_dict):
    def median_positions(data_dict, key):
        return [
            np.median(np.array(data_dict[fly_id][key])) for fly_id in data_dict.keys()
        ]

    fly_data = {
        "intact": {
            "head_roll_ant_r": median_positions(intact_dict, "Angle_head_roll_right"),
            "head_roll_ant_l": median_positions(intact_dict, "Angle_head_roll_left"),
        },
        "rlf": {
            "head_roll_ant_r": median_positions(exp_dict, "Angle_head_roll_right"),
            "head_roll_ant_l": median_positions(exp_dict, "Angle_head_roll_left"),
        },
    }

    return fly_data


def check_criteria_ant_amp(chunked_data, chunk_idx, cols, thresholds):
    return {
        "head_pitch": is_head_pitching(
            chunked_data[chunk_idx, :, cols.index("Angle_head_pitch")],
            threshold=thresholds["head_pitch"],
        ),
        "has_outliers_tita": has_outliers(
            chunked_data[chunk_idx, :, cols.index("mean_TiTa_y")],
            threshold=thresholds["outlier_pos"],
        ),
    }


def which_side_head_rotated(chunked_data, chunk_idx, cols, thresholds):
    # if true then continue
    median_roll = np.median(chunked_data[chunk_idx, :, cols.index("Angle_head_roll")])
    if median_roll > thresholds["head_roll"]:
        return "right"
    elif median_roll < -thresholds["head_roll"]:
        return "left"
    else:
        return False


def prepare_data_panel_i(df_kinematics):
    """Prepares the kinematic data for panel E using the following steps:
    1- Divide the data into chunks
    2- For each experimental case (e.g., intact and head-fixed), first check if chunks contain any outliers, if not, then label each chunk based on the coordination between freely moving body parts.
    3- Gather the data for each condition and convert it into a dataframe.

    Parameters
    ----------
    df_kinematics : pd.DataFrame
        Dataframe containing the kinematic data of intact and experimental conditions.
    """

    threshold = {
        "head_pitch": 8,  # deg
        "head_roll": 5,  # deg
        "outlier_angle": 0,  # deg
        "outlier_pos": 2,  # mm
    }

    columns = [
        "Angle_head_roll",
        "Angle_head_pitch",
        "mean_TiTa_y",
    ]

    # Mean tita y
    df_kinematics["mean_TiTa_y"] = df_kinematics[
        ["Pose_RF_Tarsus_y", "Pose_LF_Tarsus_y"]
    ].mean(axis=1)

    # Separate experimental and control groups
    chunks_intact = get_chunks(df_kinematics, ["Beh"], columns)
    chunks_rla = get_chunks(df_kinematics, ["RLA"], columns)

    # We will label each chunk based on the kinematic variables
    chunk_intact_beh_dict = {}
    for fly_id, chunked_data_intact in chunks_intact.items():
        chunk_intact_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_intact.shape[0]):
            if any(
                list(
                    check_criteria_ant_amp(
                        chunked_data_intact, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            head_rotation = which_side_head_rotated(
                chunked_data_intact, chunk_no, columns, threshold
            )

            if head_rotation:
                for i, col in enumerate(columns):
                    chunk_intact_beh_dict[f"{fly_id}_beh"][
                        f"{col}_{head_rotation}"
                    ].extend(chunked_data_intact[chunk_no, :, i])

    chunk_rla_beh_dict = {}
    for fly_id, chunked_data_rla in chunks_rla.items():
        chunk_rla_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_rla.shape[0]):
            if any(
                list(
                    check_criteria_ant_amp(
                        chunked_data_rla, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            head_rotation = which_side_head_rotated(
                chunked_data_rla, chunk_no, columns, threshold
            )

            if head_rotation:
                for i, col in enumerate(columns):
                    chunk_rla_beh_dict[f"{fly_id}_beh"][
                        f"{col}_{head_rotation}"
                    ].extend(chunked_data_rla[chunk_no, :, i])

    return chunk_intact_beh_dict, chunk_rla_beh_dict


def boxen_panel_i(intact_dict, exp_dict):
    def create_data(label, key, data_dict):
        data = from_dict_to_iterable(data_dict, key)
        return {"lateral_pos": data, "label": [label] * len(data)}

    data_beh = {
        "intact_r": create_data("Intact R", "mean_TiTa_y_right", intact_dict),
        "intact_l": create_data("Intact L", "mean_TiTa_y_left", intact_dict),
        "ant_amp_r": create_data("AA R", "mean_TiTa_y_right", exp_dict),
        "ant_amp_l": create_data("AA L", "mean_TiTa_y_left", exp_dict),
    }

    return convert_dict_to_dataframe(data_beh)


def scatter_panel_i(intact_dict, exp_dict):
    def median_positions(data_dict, key):
        return [
            np.median(np.array(data_dict[fly_id][key])) for fly_id in data_dict.keys()
        ]

    fly_data = {
        "intact": {
            "lateral_pos_ant_r": median_positions(intact_dict, "mean_TiTa_y_right"),
            "lateral_pos_ant_l": median_positions(intact_dict, "mean_TiTa_y_left"),
        },
        "ant_amp": {
            "lateral_pos_ant_r": median_positions(exp_dict, "mean_TiTa_y_right"),
            "lateral_pos_ant_l": median_positions(exp_dict, "mean_TiTa_y_left"),
        },
    }

    return fly_data


# General checks
def check_criteria_hf_rlf(chunked_data, chunk_idx, cols, thresholds):
    return {
        "antenna_pitch": is_antenna_pitching(
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_R")],
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_L")],
            threshold=thresholds["antenna_pitch"],
        ),
        "has_outliers_antenna": has_outliers(
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_R")],
            chunked_data[chunk_idx, :, cols.index("Angle_antenna_pitch_L")],
            threshold=thresholds["outlier_angle"],
        ),
        "has_outliers_tita": has_outliers(
            chunked_data[chunk_idx, :, cols.index("mean_TiTa_y")],
            threshold=thresholds["outlier_pos"],
        ),
    }


def prepare_data_panel_k(df_kinematics):
    """Prepares the kinematic data for panel E using the following steps:
    1- Divide the data into chunks
    2- For each experimental case (e.g., intact and head-fixed), first check if chunks contain any outliers, if not, then label each chunk based on the coordination between freely moving body parts.
    3- Gather the data for each condition and convert it into a dataframe.

    Parameters
    ----------
    df_kinematics : pd.DataFrame
        Dataframe containing the kinematic data of intact and experimental conditions.
    """

    threshold = {
        "antenna_pitch": -22,  # deg
        "outlier_angle": 0,  # deg
        "outlier_pos": 2,  # mm
        "difference": 6,  # deg
    }

    columns = [
        "Angle_antenna_pitch_R",
        "Angle_antenna_pitch_L",
        "Angle_head_pitch",
        "Angle_head_roll",
        "mean_TiTa_y",
    ]

    # Mean tita y
    df_kinematics["mean_TiTa_y"] = df_kinematics[
        ["Pose_RF_Tarsus_y", "Pose_LF_Tarsus_y"]
    ].mean(axis=1)

    # Separate experimental and control groups
    chunks_intact = get_chunks(
        df_kinematics, ["Beh", "step_3sec", "step_2sec"], columns
    )
    chunks_rlf = get_chunks(
        df_kinematics, ["RLF", "RLF_step_3sec", "RLF_step_2sec"], columns
    )
    chunks_rlf_hf = get_chunks(
        df_kinematics, ["RLF_HF", "RLF_HF_step_3sec", "RLF_HF_step_2sec"], columns
    )

    # We will label each chunk based on the kinematic variables
    chunk_intact_beh_dict = {}
    for fly_id, chunked_data_intact in chunks_intact.items():
        chunk_intact_beh_dict[f"{fly_id}_beh"] = {
            f"Angle_antenna_pitch_{side}": [] for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_intact.shape[0]):
            if any(
                list(
                    check_criteria_hf_rlf(
                        chunked_data_intact, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            pitched_antenna = which_antenna_pitched(
                chunked_data_intact, chunk_no, columns, threshold
            )

            if pitched_antenna:
                chunk_intact_beh_dict[f"{fly_id}_beh"][
                    f"Angle_antenna_pitch_{pitched_antenna}"
                ].extend(
                    chunked_data_intact[
                        chunk_no,
                        :,
                        columns.index(
                            f"Angle_antenna_pitch_{pitched_antenna[0].upper()}"
                        ),
                    ]
                )

    chunk_rlf_beh_dict = {}
    for fly_id, chunked_data_rlf in chunks_rlf.items():
        chunk_rlf_beh_dict[f"{fly_id}_beh"] = {
            f"Angle_antenna_pitch_{side}": [] for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_rlf.shape[0]):
            criteria_stat = check_criteria_hf_rlf(
                chunked_data_rlf, chunk_no, columns, threshold
            )
            if criteria_stat["antenna_pitch"] or criteria_stat["has_outliers_antenna"]:
                continue

            pitched_antenna = which_antenna_pitched(
                chunked_data_rlf, chunk_no, columns, threshold
            )

            if pitched_antenna:
                chunk_rlf_beh_dict[f"{fly_id}_beh"][
                    f"Angle_antenna_pitch_{pitched_antenna}"
                ].extend(
                    chunked_data_rlf[
                        chunk_no,
                        :,
                        columns.index(
                            f"Angle_antenna_pitch_{pitched_antenna[0].upper()}"
                        ),
                    ]
                )

    chunk_rlf_hf_beh_dict = {}
    for fly_id, chunked_data_rlf_hf in chunks_rlf_hf.items():
        chunk_rlf_hf_beh_dict[f"{fly_id}_beh"] = {
            f"Angle_antenna_pitch_{side}": [] for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_rlf_hf.shape[0]):
            criteria_stat = check_criteria_hf_rlf(
                chunked_data_rlf_hf, chunk_no, columns, threshold
            )
            if criteria_stat["antenna_pitch"] or criteria_stat["has_outliers_antenna"]:
                continue

            pitched_antenna = which_antenna_pitched(
                chunked_data_rlf_hf, chunk_no, columns, threshold
            )

            if pitched_antenna:
                chunk_rlf_hf_beh_dict[f"{fly_id}_beh"][
                    f"Angle_antenna_pitch_{pitched_antenna}"
                ].extend(
                    chunked_data_rlf_hf[
                        chunk_no,
                        :,
                        columns.index(
                            f"Angle_antenna_pitch_{pitched_antenna[0].upper()}"
                        ),
                    ]
                )

    return chunk_intact_beh_dict, chunk_rlf_beh_dict, chunk_rlf_hf_beh_dict


def from_dict_to_iterable2(dict_data):
    data = []
    for fly_id in dict_data.keys():
        for angle_name, angle_list in dict_data[fly_id].items():
            if angle_list:
                data.extend(-1 * np.array(angle_list))
            else:
                continue
                # print(f"No data for {fly_id} {angle_name}")
    return data


def boxen_panel_k(intact_dict, exp_dict, exp_dict2):
    def create_data(label, data_dict):
        data = from_dict_to_iterable2(data_dict)
        return {"antenna_pitch": data, "label": [label] * len(data)}

    data_beh = {
        "intact": create_data("Intact", intact_dict),
        "rlf": create_data("RLF", exp_dict),
        "rlf_hf": create_data("HF&RLF", exp_dict2),
    }

    return convert_dict_to_dataframe(data_beh)


def scatter_panel_k(intact_dict, exp_dict, exp_dict2):
    def get_perc_diff(dict_data, perc_upper, perc_lower):
        perc_list = []
        for fly_id, angles in dict_data.items():
            data = [
                -1 * angle for angle_list in angles.values() for angle in angle_list
            ]
            if data:
                perc_diff = np.percentile(data, perc_upper) - np.percentile(
                    data, perc_lower
                )
                perc_list.append(perc_diff)
            else:
                perc_list.append(np.nan)
        return perc_list

    fly_data = {
        "intact": {"antenna_pitch": get_perc_diff(intact_dict, 90, 10)},
        "rlf": {"antenna_pitch": get_perc_diff(exp_dict, 90, 10)},
        "hf_rlf": {"antenna_pitch": get_perc_diff(exp_dict2, 90, 10)},
    }

    return fly_data


def check_criteria_fleg_ant_amp(chunked_data, chunk_idx, cols, thresholds):
    return {
        "head_pitch": is_head_pitching(
            chunked_data[chunk_idx, :, cols.index("Angle_head_pitch")],
            threshold=thresholds["head_pitch"],
        ),
        "has_outliers_head": has_outliers(
            chunked_data[chunk_idx, :, cols.index("Angle_head_roll")],
            threshold=thresholds["outlier_angle"],
        ),
        "has_outliers_tita": has_outliers(
            chunked_data[chunk_idx, :, cols.index("mean_TiTa_y")],
            threshold=thresholds["outlier_pos"],
        ),
    }


def prepare_data_panel_l(df_kinematics):
    """Prepares the kinematic data for panel E using the following steps:
    1- Divide the data into chunks
    2- For each experimental case (e.g., intact and head-fixed), first check if chunks contain any outliers, if not, then label each chunk based on the coordination between freely moving body parts.
    3- Gather the data for each condition and convert it into a dataframe.

    Parameters
    ----------
    df_kinematics : pd.DataFrame
        Dataframe containing the kinematic data of intact and experimental conditions.
    """

    threshold = {
        "head_pitch": 8,  # deg
        "head_roll": 2,  # deg
        "outlier_angle": 90,  # deg
        "outlier_pos": 2,  # mm
    }

    columns = [
        "Angle_head_roll",
        "Angle_head_pitch",
        "mean_TiTa_y",
    ]

    # Mean tita y
    df_kinematics["mean_TiTa_y"] = df_kinematics[
        ["Pose_RF_Tarsus_y", "Pose_LF_Tarsus_y"]
    ].mean(axis=1)

    # Separate experimental and control groups
    chunks_intact = get_chunks(df_kinematics, ["Beh"], columns)
    chunks_rla = get_chunks(df_kinematics, ["RLA"], columns)
    chunks_rla_rlf = get_chunks(df_kinematics, ["RLA_RLF"], columns)

    # We will label each chunk based on the kinematic variables
    chunk_intact_beh_dict = {}
    for fly_id, chunked_data_intact in chunks_intact.items():
        chunk_intact_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_intact.shape[0]):
            if any(
                list(
                    check_criteria_ant_amp(
                        chunked_data_intact, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            head_rotation = which_side_head_rotated(
                chunked_data_intact, chunk_no, columns, threshold
            )

            if head_rotation:
                for i, col in enumerate(columns):
                    chunk_intact_beh_dict[f"{fly_id}_beh"][
                        f"{col}_{head_rotation}"
                    ].extend(chunked_data_intact[chunk_no, :, i])

    chunk_rla_beh_dict = {}
    for fly_id, chunked_data_rla in chunks_rla.items():
        chunk_rla_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_rla.shape[0]):
            if any(
                list(
                    check_criteria_ant_amp(
                        chunked_data_rla, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            head_rotation = which_side_head_rotated(
                chunked_data_rla, chunk_no, columns, threshold
            )

            if head_rotation:
                for i, col in enumerate(columns):
                    chunk_rla_beh_dict[f"{fly_id}_beh"][
                        f"{col}_{head_rotation}"
                    ].extend(chunked_data_rla[chunk_no, :, i])

    chunk_rla_rlf_beh_dict = {}
    for fly_id, chunked_data_rla_rlf in chunks_rla_rlf.items():
        chunk_rla_rlf_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_rla_rlf.shape[0]):
            criteria_stat = check_criteria_fleg_ant_amp(
                chunked_data_rla_rlf, chunk_no, columns, threshold
            )
            if criteria_stat["head_pitch"] or criteria_stat["has_outliers_head"]:
                continue

            head_rotation = which_side_head_rotated(
                chunked_data_rla_rlf, chunk_no, columns, threshold
            )

            if head_rotation:
                for i, col in enumerate(columns):
                    chunk_rla_rlf_beh_dict[f"{fly_id}_beh"][
                        f"{col}_{head_rotation}"
                    ].extend(chunked_data_rla_rlf[chunk_no, :, i])

    return chunk_intact_beh_dict, chunk_rla_beh_dict, chunk_rla_rlf_beh_dict


def from_dict_to_iterable3(dict_data, key_name="mean_TiTa_y"):
    data = []
    for fly_id in dict_data.keys():
        for angle_name, angle_list in dict_data[fly_id].items():
            if angle_list and key_name in angle_name:
                data.extend(np.array(angle_list))
            else:
                pass
    return data


def boxen_panel_l(intact_dict, exp_dict, exp_dict2):
    def create_data(label, key, data_dict):
        data = from_dict_to_iterable3(data_dict, key)
        return {"head_roll": data, "label": [label] * len(data)}

    data_beh = {
        "Intact": create_data("Intact", "Angle_head_roll", intact_dict),
        "RLA": create_data("AA", "Angle_head_roll", exp_dict),
        "RLA_RLF": create_data("AA&LA", "Angle_head_roll", exp_dict2),
    }

    return convert_dict_to_dataframe(data_beh)


def scatter_panel_l(intact_dict, exp_dict, exp_dict2):
    def get_perc_diff(
        dict_data, angle_name="Angle_head_roll", perc_upper=50, perc_lower=50
    ):
        perc_list = []
        for fly_id in dict_data.keys():
            if (
                f"{angle_name}_right" in dict_data[fly_id].keys()
                and f"{angle_name}_left" in dict_data[fly_id].keys()
            ):
                try:
                    perc_list.append(
                        np.percentile(
                            dict_data[fly_id][f"{angle_name}_right"], perc_upper
                        )
                        - np.percentile(
                            dict_data[fly_id][f"{angle_name}_left"], perc_lower
                        )
                    )
                except:
                    perc_list.append(np.nan)
        return perc_list

    fly_data = {
        "intact": {"head_roll": get_perc_diff(intact_dict)},
        "rla": {"head_roll": get_perc_diff(exp_dict)},
        "rla_rlf": {"head_roll": get_perc_diff(exp_dict2)},
    }

    return fly_data


def check_criteria_rla_hf(chunked_data, chunk_idx, cols, thresholds):
    return {
        "head_pitch": is_head_pitching(
            chunked_data[chunk_idx, :, cols.index("Angle_head_pitch")],
            threshold=thresholds["head_pitch"],
        ),
        "has_outliers_tita": has_outliers(
            chunked_data[chunk_idx, :, cols.index("mean_TiTa_y")],
            chunked_data[chunk_idx, :, cols.index("mean_TiTa_z")],
            threshold=thresholds["outlier_pos"],
        ),
        "tita_height": has_median_below(
            chunked_data[chunk_idx, :, cols.index("mean_TiTa_z")],
            threshold=thresholds["tita_height"],
        ),
    }


def check_which_side_tita(chunked_data, chunk_idx, columns):
    median_tita = np.median(chunked_data[chunk_idx, :, columns.index("mean_TiTa_y")])
    if median_tita > 0:
        return "left"
    elif median_tita < 0:
        return "right"
    else:
        return False


def prepare_data_panel_n(df_kinematics):
    """Prepares the kinematic data for panel E using the following steps:
    1- Divide the data into chunks
    2- For each experimental case (e.g., intact and head-fixed), first check if chunks contain any outliers, if not, then label each chunk based on the coordination between freely moving body parts.
    3- Gather the data for each condition and convert it into a dataframe.

    Parameters
    ----------
    df_kinematics : pd.DataFrame
        Dataframe containing the kinematic data of intact and experimental conditions.
    """

    threshold = {
        "head_pitch": 8,  # deg
        "tita_height": 0.8,  # mm
        "outlier_pos": 2,  # mm
    }

    columns = [
        "mean_TiTa_y",
        "mean_TiTa_z",
        "Angle_head_roll",
        "Angle_head_pitch",
    ]

    # Separate experimental and control groups
    chunks_intact = get_chunks(df_kinematics, ["Beh"], columns)
    chunks_rla = get_chunks(df_kinematics, ["RLA"], columns)
    chunks_rla_hf = get_chunks(df_kinematics, ["RLA_HF"], columns)

    # We will label each chunk based on the kinematic variables
    chunk_intact_beh_dict = {}
    for fly_id, chunked_data_intact in chunks_intact.items():
        chunk_intact_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_intact.shape[0]):
            if any(
                list(
                    check_criteria_rla_hf(
                        chunked_data_intact, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            tita_side = check_which_side_tita(chunked_data_intact, chunk_no, columns)

            if tita_side:
                for i, col in enumerate(columns):
                    chunk_intact_beh_dict[f"{fly_id}_beh"][f"{col}_{tita_side}"].extend(
                        chunked_data_intact[chunk_no, :, i]
                    )

    chunk_rla_beh_dict = {}
    for fly_id, chunked_data_rla in chunks_rla.items():
        chunk_rla_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_rla.shape[0]):
            if any(
                list(
                    check_criteria_rla_hf(
                        chunked_data_rla, chunk_no, columns, threshold
                    ).values()
                )
            ):
                continue

            tita_side = check_which_side_tita(chunked_data_rla, chunk_no, columns)

            if tita_side:
                for i, col in enumerate(columns):
                    chunk_rla_beh_dict[f"{fly_id}_beh"][f"{col}_{tita_side}"].extend(
                        chunked_data_rla[chunk_no, :, i]
                    )

    chunk_rla_hf_beh_dict = {}
    for fly_id, chunked_data_rla_hf in chunks_rla_hf.items():
        chunk_rla_hf_beh_dict[f"{fly_id}_beh"] = {
            f"{col}_{side}": [] for col in columns for side in ["right", "left"]
        }
        for chunk_no in range(chunked_data_rla_hf.shape[0]):
            criteria_stat = check_criteria_rla_hf(
                chunked_data_rla_hf, chunk_no, columns, threshold
            )
            if criteria_stat["tita_height"] or criteria_stat["has_outliers_tita"]:
                continue

            tita_side = check_which_side_tita(chunked_data_rla_hf, chunk_no, columns)

            if tita_side:
                for i, col in enumerate(columns):
                    chunk_rla_hf_beh_dict[f"{fly_id}_beh"][f"{col}_{tita_side}"].extend(
                        chunked_data_rla_hf[chunk_no, :, i]
                    )

    return chunk_intact_beh_dict, chunk_rla_beh_dict, chunk_rla_hf_beh_dict


def boxen_panel_n(intact_dict, exp_dict, exp_dict2):
    def create_data(label, key, data_dict):
        data = from_dict_to_iterable3(data_dict, key)
        return {"tita": data, "label": [label] * len(data)}

    data_beh = {
        "Intact": create_data("Intact", "mean_TiTa_y", intact_dict),
        "RLA": create_data("AA", "mean_TiTa_y", exp_dict),
        "RLA_HF": create_data("AA&HF", "mean_TiTa_y", exp_dict2),
    }

    return convert_dict_to_dataframe(data_beh)


def scatter_panel_n(intact_dict, exp_dict, exp_dict2):
    def get_perc_diff(
        dict_data, angle_name="mean_TiTa_y", percentile_upper=50, percentile_lower=50
    ):
        perc_list = []
        for fly_id in dict_data.keys():
            if (
                f"{angle_name}_right" in dict_data[fly_id].keys()
                and f"{angle_name}_left" in dict_data[fly_id].keys()
            ):
                try:
                    perc_list.append(
                        np.percentile(
                            dict_data[fly_id][f"{angle_name}_left"], percentile_upper
                        )
                        - np.percentile(
                            dict_data[fly_id][f"{angle_name}_right"], percentile_lower
                        )
                    )
                except:
                    perc_list.append(np.nan)
        return perc_list

    fly_data = {
        "intact": {"tita": get_perc_diff(intact_dict)},
        "rla": {"tita": get_perc_diff(exp_dict)},
        "rla_hf": {"tita": get_perc_diff(exp_dict2)},
    }

    return fly_data


# Main Execution
def main():
    pass


if __name__ == "__main__":
    main()
