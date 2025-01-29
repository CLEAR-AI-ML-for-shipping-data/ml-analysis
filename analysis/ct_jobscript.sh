#!/usr/bin/env bash

#SBATCH -A C3SE2025-4-2 -p vera
#SBATCH -t 05:00:00
#SBATCH -n 16

echo "Starting the training process"

cp -r ./* $TMPDIR/

cd $TMPDIR

# python3 train_2layer_model.py -c double_layer_config.toml
apptainer run my_container.sif

cp logs/*.log $SLURM_SUBMIT_DIR/logs/
cp *.pt $SLURM_SUBMIT_DIR/saved_models/
cp -r runs/* $SLURM_SUBMIT_DIR/runs/

echo "Finito"

