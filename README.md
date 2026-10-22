# EstIP-Estimation-Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23558309.svg)](https://doi.org/10.5281/zenodo.23558309)  
Developed at [IOER](https://www.ioer.de/en/)  
Copyright © Denis Reiter, Dresden, 2026  
All rights reserved.


## Overview
This is the second part of the **EstIP population estimation pipeline** and serves as the foundation for the corresponding [dataset series](https://data.fdz.ioer.de/dataverse/root/?q=EstIP). It is meant to be used in conjuction with the [building feature toolbox](https://github.com/denisreiter/EstIP-Building-Toolbox) as the main source of input data.
The features processed by the toolbox serve as independant features in this XGBoost-based machine learning pipeline, which consists of multiple interactive notebooks for training and analysing the model first, and a standalone script for population estimation.

The pipeline delivers results on both building and 100m grid level. For reasons of data protection when publishing, the results on grid level use the CKM anonymisation process. Further details on this method can be found [here](). Although the model only produces synthetic data, it is advised that results on building level are treated as sensitive data.

### Related publications:

1. Reiter, D., Jehling, M. and Hecht, R. (2023) ‘Benefits of using address-based dasymetric mapping in micro-level census disaggregation’, _AGILE: GIScience Series_, 4, pp. 1–7.Available at: https://doi.org/10.5194/agile-giss-4-38-2023.

2. Reiter, D., Jehling, M. and Hecht, R. (2026) ‘Urban form metrics for building occupancy modelling: applications and impacts’, _Buildings & Cities_, 7(1), pp. 1035–1051. Available at: https://doi.org/10.5334/bc.749.

3. Reiter, D., Jehling, M. and Hecht, R. (2026) ‘Intercensal building‑based population estimation’, _International Journal of Digital Earth_, 19(2), p. 2738267. Available at: https://doi.org/10.1080/17538947.2026.2738267.



## Features
- Built for 3D building (LoD2) & census data
- Interactive Jupyter notebooks for training and analysing the model
- Optional deep model evaluation included
- Prediction is handled via predefined config files for reproducability
- Delivers building & 100m grid level data


## Requirements
- Windows 11 (currently the only supported OS)
- Python 3.11
- Working conda/miniforge installation
- Graphviz binaries


## Installation

1. Clone the repository.
    ```bash
    git clone https://github.com/denisreiter/EstIP-Estimation-Pipeline.git
    ```
2. Open a conda/miniforge prompt.
3. Change to the directory where you cloned the repo into.
    ```bash
    cd <path/to/your/cloned/repo>
    ```
4. Create and activte a new environment.
   ```bash
   conda env create --file environment.yml
   conda activate estip-estimation-pipeline
   ```


## Configuration
For `train.ipynb`, paths are setup inside the notebook itself, in section 1 "Setup". Train parameters can be set there, too. Additionally, there is a config file `utils/train_config.py`, which contains the column setup (ID, target, features, categorical columns) for model trainin. This file is preconfigured for the EstIP-Building-Toolbox output. Adapt this as necessary if you are using a different or reduced input dataset.

For `population_estimation.py`, a separate configuration file is used. There are two example configurations provided under `configs/`, one for application during a training run for evaluation (`example_eval_config.yaml`) and one for production runs (`example_estimation_config.yaml`). Additionally, for the optional correction factor application to work, official population counts per municipality are needed. If none are available, set:
```bash
apply_correction: false
```
inside the config file. There are also more options concerning thresholding and rounding found inside the config file, which can be turned off/changed.

Notebooks `test.ipynb` and `analysis.ipynb` are using the same `eval_config.yaml` as `population_estimation.py`, the path to which needs to be set at the beginning of each notebook. For both these notebooks to work properly, ground truth census data is necessary. If none are available, these two notebooks can be skipped.

## Assumptions & Limitations 
- All data should be projected to EPSG:3035.
- The entire pipeline is preconfigured for [EstIP-Building-Toolbox](https://github.com/denisreiter/EstIP-Building-Toolbox) input data, to be used in conjuction with German [3D building](https://doi.org/10.71830/GVNGUA) and [census](https://ergebnisse.zensus2022.de/datenbank/online/) data. Adaptation is necessary for different input data formats/regions.
- Multiple ancillary datasets need to be prepared to execute the full pipeline: _Census data_ and _population on municipal level_. The parts requiring these are optional.


## Usage

### Preliminary steps:
1. Generate a viable ptable for CKM anonymisation using the corresponding [R package](https://doi.org/10.32614/CRAN.package.ptable).
2. Generate a hex-salt-key and add it to your `eval_config.yaml` / `estimation_config.yaml` files.
    ```bash
    python utils/generate_salt.py
    ```

### For initial model training and evaluation:
1. Adapt paths in `train.ipynb`, check if fields in `utils/train_config.py` match your input data.
2. Activate the environment in your Conda Prompt.
    ```bash
    conda activate estip-estimation-pipeline
    ```
3. Change to the directory where you cloned the repo into.
    ```bash
    cd <path/to/your/cloned/repo>
    ```
4. Run `train.ipynb` in Juypter Lab (or an IDE like VSCode).
5. Adapt paths and variables in `eval_config.yaml`.
6. Run `population_estimation.py` in your Conda Prompt.
    ```bash
    python scripts/population_estimation.py --config <path/to/your/config/eval_config.yaml>
    ```
7. **Optionally**: Adapt paths in `test.ipynb` and `analysis.ipynb` to point to your `eval_config.yaml` and run them in Juypter Lab (or an IDE like VSCode).

### For production:
1. Activate the environment in your Conda Prompt.
    ```bash
    conda activate estip-estimation-pipeline
    ```
2. Change to the directory where you cloned the repo into.
    ```bash
    cd <path/to/your/cloned/repo>
    ```
3. Adapt paths and variables in `estimation_config.yaml`
4. Run `population_estimation.py` in your Conda Prompt.
    ```bash
    python scripts/population_estimation.py --config <path/to/your/config/estimation_config.yaml>
    ```


## Citation 
Reiter, D. (2026). EstIP-Estimation-Pipeline (Version 0.1.0) [Computer software]. https://doi.org/10.5281/zenodo.23558309
