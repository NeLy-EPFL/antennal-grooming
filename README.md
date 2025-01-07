# 🧠 Centralized brain networks underlie grooming body part coordination

This repository holds code for reproducing the results from
[**Centralized brain networks underlie grooming body part coordination**](https://www.biorxiv.org/content/10.1101/2024.12.17.628844v1).

---

## 📂 Repository Structure

The repository is organized as follows:

- `assets/`
  Contains the paper figures.

- `data/`
  Contains data necessary for reproduction.

- `src/`
  Contains Jupyter notebooks for generating figures.
  - `FigureX.ipynb` – Notebook to generate Figure X.
  - `prepare_data/` – Additional code for preprocessing the data.

- `results/`
  Directory where generated results will be saved.

- `download_data.sh`
  Script to automatically download the required dataset.

- `generate_figures.sh`
  Script to run all notebooks at once and save figures under `/results`.

- `requirements.txt`
  Lists the Python dependencies.

---

## ⚙️ Installation

Follow these steps to set up your environment and get started:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NeLy-EPFL/antennal-grooming.git
   cd antennal-grooming
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   conda create -n grooming python>=3.8
   conda activate grooming
   ```
    Alternatively, you may use a Python virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---
## 🚀 Usage Instructions

1. **Download the data necessary for running the code:**
   ```bash
   ./download_data.sh
   ```
    Verify that the dataset is successfully downloaded and places under `/data`. If it fails, download the data manually through [this link](https://dataverse.harvard.edu/dataverse/ozdil_2024_antennal_grooming).

2. **Navigate to the src folder and open the desired Jupyter notebook:**
   ```bash
   cd src
   jupyter notebook Figure1.ipynb
   ```
   Make sure that [Jupyter notebook](https://jupyter.org/install) is installed and your virtual environment is activated.

   Alternatively, you can run the following script to automatically run all notebooks under `/src` and save their outputs to `/results`.
   ```bash
   ./generate_figures.sh
   ```
---
## 👩‍💻 Other resources

Data used in this paper have been obtained by using several other repositories:

* **[kinematics3d](https://github.com/NeLy-EPFL/kinematics3d):** used to automate 2D & 3D pose estimation on video recordings, using DeepLabCut and Anipose. We used this repository in a separate virtual environment. Please refer to the repository for more information.
* **[SeqIKPy](https://github.com/NeLy-EPFL/sequential-inverse-kinematics):** used to estimate the joint angles of the fly legs and antennae from 3D kinematics. Please refer to the repository for more information.
**NOTE:** We used the `seqikpy` package in `kinematics3d` repository to estimate antennal grooming kinematics, used throughout the paper (Figs. 1,2,5).
* **[FARMS](https://github.com/farmsim):** used to simulate the fly grooming kinematics in MuJoCo. Please refer to the repository for more information.
**NOTE:** We used the `farms` package to perform kinematic replay experiments (Fig. 2).
* **[FlyVis](https://github.com/gizemozd/flyvis):** used to train connectome-derived artificial neural networks to emulate grooming behavior. Please refer to the repository for more information.
**NOTE:** We used the `flyvis` package to train the neural networks and perform neural perturbation experiments (Fig. 5,6).


---
## 🐞 Questions
Please get in touch if you have any questions or comments!
You can open an issue on our [issues page](https://github.com/NeLy-EPFL/antennal-grooming/issues) or e-mail us directly at pembe.ozdil@epfl.ch

---
## 💬 Citing
If you find this package useful in your research, please consider citing it using the following BibTeX entry:
```bibtex
@article{
    ozdil_2024_centralized,
    author = {{\"O}zdil, Pembe Gizem and Arreguit, Jonathan and Scherrer, Clara and Ijspeert, Auke and Ramdya, Pavan},
    title = {Centralized brain networks underlie body part coordination during grooming},
    year = {2024},
    doi = {10.1101/2024.12.17.628844},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2024/12/17/2024.12.17.628844},
    journal = {bioRxiv}
}
```
