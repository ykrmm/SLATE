# SLATE Model (Under Review Neurips 2024 )

This is the official repository for the paper SLATE: Supra Laplacian Encoding for Transformer on Dynamic Graph.

This code is intended solely for the use of the reviewers and the Area Chair. We trust your discretion and kindly ask that you do not share it until the final decisions are released.

This is a provisional code; due to time constraints, it is not perfectly commented.

![tw](https://github.com/ykrmm/SLATE/blob/main/slate_model.png)

## Installation
```
conda create -n SLATE python=3.9
conda activate SLATE
pip install -e .
```
Try : 
```
pip install -r requirements.txt
```
If you encounter a bug, please install at least the following packages in the following order:

```
torch==2.2.2+cu118
torch_cluster==1.6.3+pt22cu118
torch_geometric==2.4.0
torch_scatter==2.1.2+pt22cu118
torch_sparse==0.6.18+pt22cu118
torch_spline_conv==1.2.2+pt22cu118
torcheval==0.0.6
-----------------------------------
networkx==3.2.1
hydra-colorlog==1.2.0
hydra-core==1.3.2
pandas==1.3.5
```


## Code explaination
### About Hydra: 
We use Hydra to execute scripts. Hydra overwrites the configuration files located in `slate/config/`. For example, the configuration file for the SLATE model is in `slate/config/model/SLATE.yaml`.

### Code structure and main files to review
The main files of interest according to us are:

- `slate/lib/supra.py`: This file contains the code for transforming a discrete dynamic graph into a multilayer graph.

- `slate/lib/encoding.py`: This file contains the code for performing the spectral decomposition of the supra-adjacency matrix built in `lib/supra.py`.

- `slate/models/slate.py`: This file contains our SLATE model. Here you can see how the 4 steps of our model are performed (Supra-adjacency, construction of the spatio-temporal encoding, Fully-connected Transformer and Edgemodule cross-attention).

- `slate/engine/engine_link_pred.py`: This is the main file where we load the models, datasets, train the model, and perform evaluation.

- `slate/data`: The discrete dynamic graph data used for the experiments.

- `scripts`: The example execution scripts are located in this folder

## Notebooks

### `notebooks/example_SLATE_toygraph.ipynb`
We have provided a demonstration of the SLATE model as well as the main operations:

- Transform a dynamic graph into a spatio-temporally connected multilayer graph.
- Show the projections of the eigenvectors forming the basis of our spatio-temporal encoding on the transformed multilayer graph.
- Demonstrate how to generate the token sequence as input for the transformer.

### `notebooks/supralaplacian_visu.ipynb`

The visualizations from the rebuttal as well as the code used to generate them are in this notebook. You can find all the generated figures in `figures/`.

## Run experiments 
### Exemple : run SLATE sur le dataset UNtrade

Before running an experiment, it is necessary to go to the configuration file: `slate/config/dataset/UNtrade`.yaml and replace `datadir:YourAbsolute/Path/data` with your absolute path to the `data/` folder.

You can then launch the script:


```
sh scripts/slate_trade.sh 
```