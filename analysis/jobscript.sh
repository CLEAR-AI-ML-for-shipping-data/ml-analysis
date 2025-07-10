#!/usr/bin/env bash

#SBATCH -A C3SE2025-4-2 -p vera
#SBATCH -t 05:00:00
#SBATCH -n 16

module load virtualenv/20.23.1-GCCcore-12.3.0 PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 h5py/3.9.0-foss-2023a
module load scikit-learn/1.3.1-gfbf-2023a scikit-image/0.22.0-foss-2023a

module list

source training_venv/bin/activate

echo "Starting the training process"

cp -r ./* $TMPDIR/

cd $TMPDIR

python3 train_2layer_model.py -c double_layer_config.toml

cp logs/*.log $SLURM_SUBMIT_DIR/logs/
cp *.pt $SLURM_SUBMIT_DIR/saved_models/
cp -r runs/* $SLURM_SUBMIT_DIR/runs/

echo "Finito"
