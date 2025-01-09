import numpy as np


def get_area_under(neuron_act, start=2500, end=4500):
    # Let's calculate the positive and negative area separately
    positive_area = np.sum(neuron_act[start:end][neuron_act[start:end] > 0])
    negative_area = abs(np.sum(neuron_act[start:end][neuron_act[start:end] < 0]))

    return positive_area, negative_area


def calc_area_under_neuron(
    current_list, stim_interval, data_path, neurons_ordered, neuron_name="ANTEN_MN4"
):
    area_under_right = np.zeros((len(current_list), len(current_list)))
    area_under_left = np.zeros((len(current_list), len(current_list)))

    for i, current_r in enumerate(current_list):
        for j, current_l in enumerate(current_list):
            voltage = load_npz(
                data_path
                / f"voltage_JO-F_R_{round(current_r ,1)}_L_{round(current_l,1)}.npz"
            )
            voltage = relu_function(voltage)
            right_trace = voltage[:, neurons_ordered.index(f"{neuron_name}_R")]
            left_trace = voltage[:, neurons_ordered.index(f"{neuron_name}_L")]

            # Normalize by the max value of the activity
            max_value = max(right_trace.max(), left_trace.max())

            if max_value == 0:
                max_value = 1

            right_trace = right_trace / max_value
            left_trace = left_trace / max_value

            area_under_right[i, j], _ = get_area_under(
                right_trace,
                start=stim_interval[0],
                end=stim_interval[1],
            )
            area_under_left[i, j], _ = get_area_under(
                left_trace,
                start=stim_interval[0],
                end=stim_interval[1],
            )

    return area_under_right, area_under_left

def count_number_of_neurons_active(voltage_exp, voltage_intact, interval=(2600, 4500)):
    """ Count the number of neurons that are more active in the experimental condition than
    the intact condition.

    Parameters
    ----------
    voltage_exp : np.ndarray
        Neural activity of the experimental condition
    voltage_intact : np.ndarray
        Neural activity of the intact condition
    interval : tuple, optional
        Interval in which the neural activity will be compared

    Returns
    -------
    int
        Number of neurons that are more active


    This function is applied on the population neural activity, which is obtained by running the following code:
    ```
        graph_voltage_intact = get_graph_voltage(
            voltage_intact, neurons_ordered, cluster2neuron, node2name
        )
        number_of_neurons_active = count_number_of_neurons_active(
            graph_voltage,
            graph_voltage_intact,
            interval=stim_interval,
        )
    ```
    """
    active_neurons = []
    for node in range(voltage_exp.shape[1]):
        if np.mean(voltage_exp[interval[0] : interval[1], node]) > 5 * np.mean(
            voltage_intact[interval[0] : interval[1], node]
        ):
            active_neurons.append(node)

    return len(active_neurons)

def identify_important_neurons_silencing(
    seed_number, neurons_to_check, current_pair, neuron_index, interval=(2600, 4000)
):
    """ Calculates the Unilateral Selectivity Index for the given neurons
    in silencing experiments.

    Parameters
    ----------
    seed_number : int
        Seed number of the model
    neurons_to_check : list
        List of neurons to check
    current_pair : tuple
        Values of the left and right input current
    neuron_index : list
        All neuron names
    interval : tuple, optional
        Interval in which the calculation will be performed,
        by default (2500, 4500)

    Returns
    -------
    tuple
        Metric values and neuron voltages
    """
    seed_path = (
        lab_server_path / f"connectome_sym_adj_MLP_decoder_reLu_seed_{seed_number}"
    )
    metric = {}
    neuron_voltage = {}

    left_ind = neuron_index.index("ANTEN_MN4_L")
    right_ind = neuron_index.index("ANTEN_MN4_R")

    for neuron_name in neurons_to_check:
        voltage = load_npz(
            seed_path
            / f"silence_{neuron_name}/voltage_JO-F_R_{round(current_pair[1],1)}_L_{round(current_pair[0],1)}.npz"
        )
        # Relu
        voltage = relu_function(voltage)

        ratio = get_diagonal_current(
            voltage, left_ind, right_ind, stim_interval=interval
        )

        neuron_voltage[neuron_name] = (voltage[:, left_ind], voltage[:, right_ind])
        metric[neuron_name] = ratio

    return metric, neuron_voltage

def identify_important_neurons_activate(
    seed_number,
    neurons_to_check,
    neuron_index,
    interval=(2600,4000)
):
    """ Calculates the Unilateral Selectivity Index for the given neurons
    in activation experiments.

    Parameters
    ----------
    seed_number : int
        Seed number of the model
    neurons_to_check : list
        List of neurons to check
    neuron_index : list
        All neuron names
    interval : tuple, optional
        Interval in which the calculation will be performed,
        by default (2500, 4500)

    Returns
    -------
    tuple
        Metric values and neuron voltages
    """

    seed_path = lab_server_path / f"connectome_sym_adj_MLP_decoder_reLu_seed_{seed_number}"
    metric = {}
    neuron_voltage = {}

    left_ind = neuron_index.index("ANTEN_MN4_L")
    right_ind = neuron_index.index("ANTEN_MN4_R")

    for neuron_name in neurons_to_check:
        voltage = load_npz(
            seed_path / f"activate_{neuron_name}/voltage_JO-F_R_1.5_L_1.5_{neuron_name}_L_10.0.npz"
        )
        # Relu
        voltage = relu_function(voltage)

        ratio = get_diagonal_current(
            voltage,
            left_ind,
            right_ind,
            stim_interval=interval
        )
        metric[neuron_name] = ratio

        neuron_voltage[neuron_name] = (
            voltage[:, left_ind],
            voltage[:, right_ind]
        )

    return metric, neuron_voltage
