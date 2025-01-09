"""
Neuron IDs and colors of the known neurons in the grooming circuitry.
"""

import yaml
from pathlib import Path

current_dir = Path(__file__).parents[0]

with open(current_dir / "Connectome_neurons.yaml", "r") as file:
    neuron_dict = yaml.safe_load(file)


SENSORY_NEURONS = {
    **neuron_dict["BM-Ant"],
    **neuron_dict["JO-C"],
    **neuron_dict["JO-E"],
    **neuron_dict["JO-F"],
}

INTERNEURONS = {
    **neuron_dict["WED"],
    **neuron_dict["ABN"],
    **neuron_dict["IN"],
}

MOTOR_NEURONS = {
    **neuron_dict["ANTEN_MN"],
    **neuron_dict["NECK_MN"],
}

DESCENDING_NEURONS = {
    **neuron_dict["ADN"],
    **neuron_dict["12A_DN"],
    **neuron_dict["DN"],
}


ALL_NEURONS = {
    **SENSORY_NEURONS,
    **INTERNEURONS,
    **DESCENDING_NEURONS,
    **MOTOR_NEURONS,
}

ALL_NEURONS_REV = {
    v["id"]: {
        "name": k,
        "color": v["color"],
        "nt": v["nt"],
        "cell_type": v.get("cell_type", ""),
    }
    for k, v in ALL_NEURONS.items()
}

ALL_NEURONS_JO = {
    **neuron_dict["sensory_neurons"],
    **INTERNEURONS,
    **DESCENDING_NEURONS,
    **MOTOR_NEURONS,
}

ALL_NEURONS_REV_JO = {
    v["id"]: {"name": k, "color": v["color"], "nt": v["nt"]}
    for k, v in ALL_NEURONS_JO.items()
}


# utils
def get_name_from_id(neuron_id):
    return ALL_NEURONS_REV[neuron_id]["name"]


def is_sensory(neuron_name):
    return neuron_name in SENSORY_NEURONS


def is_motor(neuron_name):
    return neuron_name in MOTOR_NEURONS


def is_interneuron(neuron_name):
    return neuron_name in INTERNEURONS


def is_descending_neuron(neuron_name):
    return neuron_name in DESCENDING_NEURONS


def is_excitatory(neuron_name):
    return ALL_NEURONS[neuron_name]["nt"] == "excitatory"


def is_inhibitory(neuron_name):
    return ALL_NEURONS[neuron_name]["nt"] == "inhibitory"
