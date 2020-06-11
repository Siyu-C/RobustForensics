#!/usr/bin/env sh
mkdir -p logs
now=$(date +"%Y%m%d_%H%M%S")

srun --mpi=pmi2 -p $2 -n8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --job-name=$4 \
python -u main.py \
--config $1 \
--distributed_path $3 \
--datetime $now \
2>&1|tee logs/log-$now.out
