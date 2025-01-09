import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN


def calc_similarity_score(neuron_index, adj_matrix, indices_to_lookat=None):
    """Calculate the pearson correlation between the presynaptic and postsynaptic connections of a neuron."""

    if indices_to_lookat is None:
        indices_to_lookat = np.arange(adj_matrix.shape[0] // 2)

    similarity_scores = []

    # Iterate over all other neurons to compute similarity
    for i in indices_to_lookat:
        # Compute correlation for outgoing connections
        corr_outgoing, _ = pearsonr(adj_matrix[neuron_index, :], adj_matrix[i, :])

        # Compute correlation for incoming connections
        corr_incoming, _ = pearsonr(adj_matrix[:, neuron_index], adj_matrix[:, i])

        # Aggregate the correlations (example: simple average)
        avg_corr = corr_outgoing + corr_incoming

        if np.isnan(avg_corr):
            avg_corr = 0

        similarity_scores.append(avg_corr)

    return similarity_scores


def cluster_by_dbscan(similarity_matrix, eps=0.5, min_samples=1):
    """Cluster neurons using DBSCAN algorithm."""
    # Use DBSCAN to cluster the neurons
    # Distance is shorter if the similarity is higher
    distance_matrix = 1 - similarity_matrix * 0.5

    # Applying DBSCAN, we precomputed the distance matrix
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    clusters = dbscan.fit_predict(distance_matrix)

    order = np.argsort(clusters)
    ordered_sim_matrix = similarity_matrix[np.ix_(order, order)]

    return order, ordered_sim_matrix


def relu_function(x, threshold=0):
    return np.maximum(x, threshold)


def calculate_area_under_curve(
    voltage, start=2500, end=4500, filter=False, threshold=0
):
    if filter:
        voltage = savgol_filter(voltage, 11, 3)
        threshold = 0.05
    # Let's calculate the positive and negative area separately
    positive_area = np.sum(voltage[start:end][voltage[start:end] > threshold])
    negative_area = np.sum(voltage[start:end][voltage[start:end] < threshold * -1])

    return positive_area, negative_area


def get_act_difference(
    current_dict,
    seed_numbers_list,
    left_neurons,
    right_neurons,
    start=2500,
    end=4500,
    filter=False,
):
    """Calculates the ratio between the are under the right MN curve to that of left MN curve.

    Parameters
    ----------
    current_dict : dict
        Dictionary containing voltage values for each current.
        Voltage is an array of (time_steps, neurons)
    seed_numbers_list : list
        Seeds ordered.
    left_neurons : list
        Indices of neurons on the left hemisphere
    right_neurons : list
        Indices of neurons on the right hemisphere
    start : int, optional
        Start index of the area calculation, by default 2700
    end : int, optional
        End index of the area calculation, by default 4500

    Returns
    -------
    np.ndarray
        Array containing area ratio per seed
    """
    assert len(left_neurons) == len(
        right_neurons
    ), "Right and left neuron indices should have the same size."
    # Create a matrix to store the activity values
    activity_matrix = np.zeros((len(seed_numbers_list), len(left_neurons)))

    # Loop through seeds
    for i, seed in enumerate(seed_numbers_list):
        # Loop through neurons
        for j, (left_mn, right_mn) in enumerate(zip(left_neurons, right_neurons)):
            voltage_relu = relu_function(current_dict[seed])
            left_activity = voltage_relu[:, left_mn]
            right_activity = voltage_relu[:, right_mn]
            # Normalize by the max value of the activity of both neurons during start-end
            max_value = max(
                left_activity[start:end].max(), right_activity[start:end].max()
            )
            if max_value == 0:
                max_value = 1

            left_activity = left_activity / max_value
            right_activity = right_activity / max_value

            left_area = calculate_area_under_curve(
                left_activity, start=start, end=end, filter=filter
            )
            right_area = calculate_area_under_curve(
                right_activity, start=start, end=end, filter=filter
            )
            # Right over left ratio, if 1 then left is 0
            ratio = (right_area[0] - left_area[0]) / (right_area[0] + left_area[0])

            if np.isclose(left_area[0], 0) and np.isclose(right_area[0], 0):
                # print(left_area[0], right_area[0])
                ratio = np.nan

            activity_matrix[i, j] = ratio

    return activity_matrix
