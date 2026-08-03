#!/bin/bash
#SBATCH --job-name=GFRaL
#SBATCH --partition=debuga100
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1          # use 1 GPU
#SBATCH --output=%j.out
#SBATCH --error=%j.err

singularity exec \
    --nv \
    --bind input:/root/af_input \
    --bind output:/root/af_output \
    --bind /dssg/share/data/alphafold3/models:/root/models \
    --bind /dssg/share/data/alphafold3/database:/root/public_databases \
    /dssg/share/imgs/ai/alphafold/alphafold3-a100.sif \
    /alphafold3_venv/bin/python /app/alphafold/run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --db_dir=/root/public_databases \
    --output_dir=/root/af_output
