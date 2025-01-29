#!/usr/bin/env bash

#SBATCH -A C3SE2024-1-22 -p vera
#SBATCH -t 05:00:00
#SBATCH -n 16

echo "[`date`] Loading modules..."
module load virtualenv/20.23.1-GCCcore-12.3.0 PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 h5py/3.9.0-foss-2023a
module load scikit-learn/1.3.1-gfbf-2023a scikit-image/0.22.0-foss-2023a

module list

echo "[`date`] Sourcing virtual env..."
source training_venv/bin/activate


echo "[`date`] Copying data..."
cp -r ./* $TMPDIR/

cd $TMPDIR

echo "[`date`] Finished copying data..."
input_file=$1

eval `head -n $SLURM_ARRAY_TASK_ID $input_file | tail -1`
echo "[`date`] Running learning rate = ${learning_rate} batch size=$batch_size filename=lr_${learning_rate}_bs_$batch_size"

python3 train_2layer_model.py -c double_layer_config.toml -b $batch_size -l ${learning_rate} -n lr_${learning_rate}_bs_$batch_size

cp logs/*.log $SLURM_SUBMIT_DIR/logs/
cp *.pt $SLURM_SUBMIT_DIR/saved_models/
cp -r runs/* $SLURM_SUBMIT_DIR/runs/

echo "Finito"

