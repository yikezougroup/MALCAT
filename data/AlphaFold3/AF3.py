import os
import json
import argparse
from pathlib import Path
import shutil

def sanitize_name(name):
    """清理非法目录字符"""
    return name.replace(" ", "_").replace("/", "-").replace("\\", "-")

# 硬编码的蛋白质序列
PROTEIN_SEQUENCE = (
    "HHHHHHGGWNLTTRSHHGFKGMWSCLEVAEACVGDVVCNAQLASYLKACSANGNPCDLKQCQAAIRFFYQNIPFNIAQMLAFCDCAQSDIPCQQSKEALHSKTCAVNMVPPPTCLSVIRSCQNDELCRRHYRTFQSKCWQRVTRKCHEDENCISTLSKQDLTCSGSDDCKAAYIDILGTVLQVQCTCRTITQSEESLCKIFQHMLHRKSCFNYPTLSNVKGMALYTRKHANKITLTGFHSPFNGE"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="作业名称")
    args = parser.parse_args()

    safe_name = sanitize_name(args.name)
    
    # 生成仅含蛋白质的JSON文件
    json_data = {
        "name": args.name,  # 使用原始名称
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": PROTEIN_SEQUENCE  # 使用硬编码序列
                }
            }
        ],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1
    }
    
    with open("input/fold_input.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # 生成SLURM脚本
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={safe_name}
#SBATCH --partition=debuga100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1          # use 1 GPU
#SBATCH --output=%j.out
#SBATCH --error=%j.err

singularity exec \\
    --nv \\
    --bind input:/root/af_input \\
    --bind output:/root/af_output \\
    --bind /dssg/share/data/alphafold3/models:/root/models \\
    --bind /dssg/share/data/alphafold3/database:/root/public_databases \\
    /dssg/share/imgs/ai/alphafold/alphafold3-a100.sif \\
    /alphafold3_venv/bin/python /app/alphafold/run_alphafold.py \\
    --json_path=/root/af_input/fold_input.json \\
    --model_dir=/root/models \\
    --db_dir=/root/public_databases \\
    --output_dir=/root/af_output
"""
    with open("submit.slurm", "w") as f:
        f.write(slurm_script)
    

if __name__ == "__main__":
    main()