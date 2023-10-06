#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --partition=rtx8000
#SBATCH --mem=64GB
#SBATCH --time=30:00:00
#SBATCH --job-name=llama2_finetuning
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soyuj@nyu.edu
#SBATCH --output=llama2_finetuning_%j.out

module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11
cd $HOME/improving-learned-index
python -m src.llama2.finetune.finetune_4bit.py