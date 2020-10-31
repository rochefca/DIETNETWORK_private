#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --account=def-hussinju
#SBATCH --time=4:00:00
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH -o /lustre04/scratch/sciclun4/exp_results/DIETNETWORK/1000G/slurm_output/slurm-%j.out
#SBATCH --array=0-4
#SBATCH --wait

#  print arguments, node and date
echo    "Arguments: $@"
echo -n "Date:       "; date
echo    "JobId:      $SLURM_JOBID"
echo    "Node:       $HOSTNAME"
echo    "Nodelist:   $SLURM_JOB_NODELIST"

source /scratch/sciclun4/envs/dietnetwork/bin/activate

echo "beginning training"
python ../../train.py "$@" --which-fold $SLURM_ARRAY_TASK_ID

echo "finished training. now computing attributions"
python ../../make_attributions.py "${@:1:8}" --which-fold $SLURM_ARRAY_TASK_ID --batch_size 12 # only include first 2 args and their values (ignore epochs)
