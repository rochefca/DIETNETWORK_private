#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --account=def-hussinju
#SBATCH --time=1:00:00
#SBATCH --mem=12GB
#SBATCH -o /lustre04/scratch/sciclun4/exp_results/DIETNETWORK/1000G/slurm_output/slurm-%j.out

#  print arguments, node and date
echo    "Arguments: $@"
echo -n "Date:       "; date
echo    "JobId:      $SLURM_JOBID"
echo    "Node:       $HOSTNAME"
echo    "Nodelist:   $SLURM_JOB_NODELIST"

source /scratch/sciclun4/envs/dietnetwork/bin/activate

python create_dataset_less_snps.py "$@"
