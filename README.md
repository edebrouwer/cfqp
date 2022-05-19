# Deep Counterfactuals Estimation with Categorical Background Variables

<img src=https://i.imgur.com/1ic5Ezk.png | width="500">

## Requirements

We use poetry as a package manager, which should take care of all dependencies.  You can install poetry [here](https://python-poetry.org/) 

The only requirement is `python>=3.9` and `python<3.11`.

## Installation

Simply run `poetry install`

## Data Generation

The data will be generated automatically when running the models.

## Running Experiments

`cd condgen/counterfactuals`

### Image Data Set

`poetry run python train_cf_cluster.py --EM=true --data_type=MNIST --max_epochs1=50 --max_epochs2=50 --noise_std=0.05 -non_additive_noise=True -num_classes_model=-1 --update_period=10`


### Harmonic Oscillator Data Set

`poetry run python train_cf_cluster.py --EM=true --data_type=SimpleTraj --max_epochs1=50 --max_epochs2=50 --noise_std=0.05 -non_additive_noise=True -num_classes_model=-1 --update_period=10`

### Harmonic Oscillator Data Set

`poetry run python train_cf_cluster.py --EM=true --data_type=CV --max_epochs1=50 --max_epochs2=50 --noise_std=0.05 -non_additive_noise=True -num_classes_model=-1 --update_period=10`

## Processing Results

The `CF_eval.ipynb` notebook is used to process the results of the counterfactual reconstructions experiments.

`MNIST_comparison.ipynb` produces the image comparison figure.




