#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --time=47:59:59
#SBATCH --job-name=DeepPairwiseImpact
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soyuj@nyu.edu
#SBATCH --output=DeepPairwiseImpact

module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11
cd $HOME/improving-learned-index
torchrun --standalone --nproc_per_node=gpu -m src.deep_impact.train \
  --triples_path /scratch/sjb8193/qidpidtriples.train.small.tsv \
  --queries_path /scratch/sjb8193/queries.train.tsv \
  --collection_path /scratch/sjb8193/expanded_collection.tsv \
  --checkpoint_dir /scratch/sjb8193/checkpoints \
  --batch_size 64 \
  --save_every 5000
