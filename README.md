# Improved Expressivity of Hypergraph Neural Networks through High-Dimensional Generalized Weisfeiler-Leman Algorithms

We prepared all codes and a subset of datasets used in our experiments. 

Regarding the k-tuple hypergraph dataset of k-GWL, preprocessing is performed on the first run. When k=2, depending on CPU performance, most datasets will be preprocessed within 12-24h. Please be patient.

The main codes are in the folder `model`,`utils`, and a subset of raw data are provided in folder `data`. You can run the train.py to run the experiments. 

## Enviroment requirement:
This repo is tested with the following enviroment, higher version of torch PyG may also be compatible. 

First let's setup a conda enviroment
```
conda create -n "KGWL" python=3.8
conda activate KGWL
```

First, we need to install the hyeprgraph deep learning library.
```
pip install dhg
```

Then install pytorch and PyG packages with specific version.
```
pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install torch-geometric
```
Finally, install some relative packages

```
pip install ipdb
pip install tqdm
pip install scipy
pip install matplotlib
pip install pandas
```

## Run experiments with one model and one dataset with specified lr and wd: 
```
python main_graph.py --model_name $model_name --dataset $dataset
```

## Run all experiments with one model with specified lr and wd: 
```
source run_one_model_g.sh [method]
```

