# DeepWaters: Reconstructing Pre-GRACE Terrestrial Water Storage Anomalies Using Deep Learning
<p align="center">
    <img src="earth_header.png" alt="Globe of TWS reconstruction" title="Model architecture" width="350"/>
<p align="center">

This repository contains the code base accompanying the master's thesis "Reconstructing Pre-GRACE Terrestrial Water Storage Anomalies Using Deep Learning" (Luis Gentner, 2024). The data processing, model training and evaluation is implemented in Python and heavily depends on the packages [Xarray](https://docs.xarray.dev/en/stable/), [PyTorch](https://pytorch.org/docs/stable/index.html), and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

## Project structure

    .
    ├── config                          <- Stores configuration files for data preprocessing, training, and plotting
    │   ├── global-ensemble_alltrain    <- Configurations of ensemble trained on all available data
    │   ├── global-ensemble_crossval    <- Configurations of cross-validation ensemble
    │   ├── prepocessing_config.yaml    <- Data preprocessing configuration
    │   └── style_paper.mplstyle        <- Matplotlib style sheet for plots
    ├── data                            <- Raw and processed inputs (content excluded from this repository)
    ├── deepwaters                      <- Source code 
    │   ├── data                        <- PyTorch Lightning Datasets and DataModules
    │   ├── models                      <- PyTorch model implementations
    │   ├── preprocessing               <- Functions for data preprocessing
    │   ├── accessors.py                <- Custom xarray and pandas "dw" accessors (e.g., use with "DataArray.dw.select_basins()")
    │   └── ...
    ├── jobs                            <- Slurm scripts to train models on the ETHZ Cluster
    ├── models                          <- Model predictions and final products (content excluded from this repository)
    ├── notebooks                       <- Notebooks used for evaluations
    ├── scripts                         <- Scripts for data processing and model training / predicting
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment
    └── pyproject.toml                  <- makes `deepwaters` pip installable ("pip install -e .")

The project structure is based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/).

## Replicating the analysis

Clone this repository to a directory of your choice:

    git clone https://github.com/lqgentner/deepwaters.git

Make sure that Python 3.11 or later is installed on your system:

    python --version

Create a new virtual environment:

    cd deepwaters
    python -m venv .venv

Activate the virtual environment. On Windows, run one of the following scripts:

    # In cmd.exe
    .venv\Scripts\activate.bat
    # In PowerShell
    .venv\Scripts\Activate.ps1

On Linux or macOS, use the source command:

    source .venv/bin/activate

Now you can install `deepwaters` as editable package. This also installs all dependencies for the DeepWaters package as well as running all scripts (data download, preprocessing, and model training).

    pip install -e .

If you additionally want to run the notebooks used to create the plots of the evaluations, install the optional dependencies with:

    pip install -e ".[interactive]"
