# DeepRec: Reconstructing Pre-GRACE Terrestrial Water Storage Anomalies Using Deep Learning
<p align="center">
    <img src="docs/figures/cover/steelblue_coastlines_southam_blue.png" alt="Globe of TWS reconstruction" title="Model architecture" width="350"/>
<p align="center">

This repository contains the code base accompanying the master's thesis "Reconstructing Pre-GRACE Terrestrial Water Storage Anomalies Using Deep Learning" (Luis Gentner, 2024). The data processing, model training and evaluation is implemented in Python and heavily depends on the packages [xarray](https://docs.xarray.dev/en/stable/), [PyTorch](https://pytorch.org/docs/stable/index.html), and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

## Project structure

    .
    ├── config                          <- Stores configuration files for data preprocessing, training, and plotting
    │   ├── ensembles_paper             <- Configurations of ensemble members with different input features
    │   ├── prepocessing_config.yaml    <- Data preprocessing configuration
    │   └── style_paper.mplstyle        <- Matplotlib style sheet for plots
    ├── data                            <- Raw and processed inputs (content excluded from this repository)
    ├── deeprec                         <- Source code
    │   ├── data                        <- PyTorch Lightning Datasets and DataModules
    │   ├── models                      <- PyTorch model implementations
    │   ├── preprocessing               <- Functions for data preprocessing
    │   ├── accessors.py                <- Custom xarray and pandas "dr" accessors (e.g., use "DataArray.dr.select_basins()")
    │   └── ...
    ├── jobs                            <- Slurm scripts to train models on the ETHZ Cluster
    ├── models                          <- Model predictions and final products (content excluded from this repository)
    ├── notebooks                       <- Notebooks used for evaluations
    ├── scripts                         <- Scripts for data processing and model training / predicting
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment
    └── pyproject.toml                  <- makes `deeprec` pip installable ("pip install -e .")

The project structure is based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/).

## Replicating the analysis

Clone this repository to a directory of your choice:

    git clone https://github.com/lqgentner/deeprec.git

Make sure that Python 3.11 or later is installed on your system:

    python --version

Create a new virtual environment:

    cd deeprec
    python -m venv .venv

Activate the virtual environment. On Windows, run one of the following scripts:

    # In cmd.exe
    .venv\Scripts\activate.bat
    # In PowerShell
    .venv\Scripts\Activate.ps1

On Linux or macOS, use the source command:

    source .venv/bin/activate

Now you can install `deeprec` as editable package. This also installs all dependencies for the package itself and for running all scripts (preprocessing and model training).

    pip install -e .

If you want to download the input and model data sets required for model training or want to run the notebooks used for creating the plots, install the optional dependenices with:

    # For downloading the data sets
    pip install -e ".[download]"
    # For running the notebooks
    pip install -e ".[interactive]"
