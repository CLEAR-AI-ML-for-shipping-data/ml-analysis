# CLEAR AI pipeline

To use this pipeline, set up a python environment with the packages from `requirements.txt`.


The pipeline consists of several consecutive steps
1. converting trajectories into images of the trajectories
2. train the neural network using the BYOL framework
3. calculate the embeddings of the images

## Convert the trajectories into images
For this, we use `analysis/prepare_2layer_data.py`.
We supply the script with the trajectories output file from the FAIRSEA code (e.g. `reduced_data_20241022_1105.pkl`).


## Train the CNN
We train the neural network with `analysis/train_2layer_model.py`.
It is important to supply a configfile that follows the conventions from `analysis/example_config.toml`

## Calculate the embeddings
