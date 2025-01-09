import scipy
import numpy as np
from collections import Counter

def get_keypoints():
    """Return keypoints and angles to consider for head and legs."""
    angles = {
        'head': [
            "Angle_head_roll", "Angle_head_pitch",
            "Angle_antenna_pitch_L", "Angle_antenna_pitch_R",
        ],
        'legs': [
            "Angle_RF_ThC_roll", "Angle_RF_ThC_pitch",
            "Angle_RF_CTr_pitch", "Angle_RF_CTr_roll",
            "Angle_LF_ThC_roll", "Angle_LF_ThC_pitch",
            "Angle_LF_CTr_pitch", "Angle_LF_CTr_roll",
        ]
    }
    keypoints = {
        'head': [
            "Pose_R_head_Antenna_base_x", "Pose_R_head_Antenna_base_y", "Pose_R_head_Antenna_base_z",
            "Pose_R_head_Antenna_edge_x", "Pose_R_head_Antenna_edge_y", "Pose_R_head_Antenna_edge_z",
            "Pose_L_head_Antenna_base_x", "Pose_L_head_Antenna_base_y", "Pose_L_head_Antenna_base_z",
            "Pose_L_head_Antenna_edge_x", "Pose_L_head_Antenna_edge_y", "Pose_L_head_Antenna_edge_z",
        ],
        'legs': [
            "Pose_RF_Tarsus_y", "Pose_RF_Tarsus_z",
            "Pose_LF_Tarsus_y", "Pose_LF_Tarsus_z",
        ]
    }
    return angles, keypoints

def extract_and_zscore_data(group_df, keypoints):
    """Extract and z-score keypoint data from the DataFrame."""
    pose_angle = group_df.loc[:, keypoints].to_numpy()
    zscored_data = (pose_angle - np.mean(pose_angle, axis=0)) / np.std(pose_angle, axis=0)
    return zscored_data

def chunk_and_label_data(zscored_data, beh_labels, chunk_length=10, stride=8):
    """Chunk data into overlapping windows and filter based on behavior labels."""
    all_chunks, all_labels = [], []
    n_frames = zscored_data.shape[0]

    for start_idx in range(0, n_frames - chunk_length + 1, stride):
        window_labels = beh_labels[start_idx : start_idx + chunk_length]
        label_counts = Counter(window_labels)

        # Filter windows with at least 6 frames of the same behavior
        if max(label_counts.values()) <= 6:
            continue

        popular_label = max(label_counts, key=label_counts.get)
        if popular_label == "background":
            continue

        all_chunks.append(zscored_data[start_idx : start_idx + chunk_length, :])
        all_labels.append(popular_label)

    return np.array(all_chunks), np.array(all_labels)

def prepare_pca_data(kinematics_df):
    """Prepare PCA data by extracting, z-scoring, and chunking kinematics data."""
    angles, keypoints = get_keypoints()
    selected_keypoints = angles['legs'] + keypoints['legs'] + angles['head'] + keypoints['head']

    all_chunks, all_labels = [], []

    for _, group_df in kinematics_df.groupby(by=["Fly", "Date", "Trial"]):
        zscored_data = extract_and_zscore_data(group_df, selected_keypoints)
        beh_labels = group_df.loc[:, "beh_label"].to_list()

        chunks, labels = chunk_and_label_data(zscored_data, beh_labels)
        all_chunks.extend(chunks)
        all_labels.extend(labels)

    all_chunks = np.array(all_chunks)
    all_labels = np.array(all_labels)

    assert all_chunks.shape[0] == all_labels.shape[0]
    assert all_chunks.shape[1] == 10  # Ensure chunk length is 10

    all_chunks_2d = all_chunks.reshape(all_chunks.shape[0], -1)
    return all_chunks, all_chunks_2d, all_labels, selected_keypoints


def get_correlation_matrix(chunk_data, labels, selected_keypoints):
    beh_labels_to_consider = [
        "unilateral_t_left",
        "unilateral_nt_left",
        "bilateral",
        "unilateral_nt_right",
        "unilateral_t_right",
        "nc_grooming",
    ]

    # now we loop through the behaviors and correlate two columns with each other
    angles_to_correlate = [
        ("Angle_antenna_pitch_R", "Angle_head_roll"),
        ("Angle_antenna_pitch_R", "Angle_head_pitch"),
        ("Angle_antenna_pitch_R", "Angle_RF_ThC_pitch"),
        ("Angle_antenna_pitch_R", "Angle_RF_ThC_roll"),
        ("Angle_antenna_pitch_R", "Angle_RF_CTr_roll"),
        ("Angle_antenna_pitch_R", "Angle_LF_ThC_pitch"),
        ("Angle_antenna_pitch_R", "Angle_LF_ThC_roll"),
        ("Angle_antenna_pitch_R", "Angle_LF_CTr_roll"),
        ("Angle_antenna_pitch_L", "Angle_head_roll"),
        ("Angle_antenna_pitch_L", "Angle_head_pitch"),
        ("Angle_antenna_pitch_L", "Angle_LF_ThC_pitch"),
        ("Angle_antenna_pitch_L", "Angle_LF_ThC_roll"),
        ("Angle_antenna_pitch_L", "Angle_LF_CTr_roll"),
        ("Angle_antenna_pitch_L", "Angle_RF_ThC_pitch"),
        ("Angle_antenna_pitch_L", "Angle_RF_ThC_roll"),
        ("Angle_antenna_pitch_L", "Angle_RF_CTr_roll"),
    ]
    correlation_matrix_median = np.empty(
        (len(angles_to_correlate), len(beh_labels_to_consider))
    )

    for i, (first_col_name, second_col_name) in enumerate(angles_to_correlate):
        for j, beh_label in enumerate(beh_labels_to_consider):
            rows_of_chunks_beh = np.where(np.array(labels) == beh_label)[0]
            X_no_bg_beh = chunk_data[rows_of_chunks_beh, ...]

            first_column = X_no_bg_beh[:, :, selected_keypoints.index(first_col_name)]
            second_column = X_no_bg_beh[:, :, selected_keypoints.index(second_col_name)]

            correlation_list = []

            # each chunk, calculate correlation
            for chunk_no in range(X_no_bg_beh.shape[0]):
                correlation = scipy.stats.pearsonr(
                    first_column[chunk_no, :], second_column[chunk_no, :]
                )[0]
                correlation_list.append(correlation)

            correlation_matrix_median[i, j] = np.median(correlation_list) ** 2

    return correlation_matrix_median