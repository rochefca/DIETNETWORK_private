#!/bin/bash
#export SLURM_ACCOUNT="def-hussinju"
#export SBATCH_ACCOUNT=$SLURM_ACCOUNT

rounds=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 )
folds=( 0 1 2 3 4 )

cp run_snp_removal_experiment.sh ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train

#  zeroth iteration is original dataset and embedding (i.e. we did not remove any SNPs yet)
cp ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train/dataset.npz ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train/dataset-0.npz
cp ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train/embedding.npz ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train/embedding-0.npz

wait

echo sbatch --parsable sbatch_wrapper_allfolds.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train \
                                                  --exp-name "remove_attr_exp_0" \
                                                  --dataset "dataset-0.npz" \
                                                  --embedding "embedding-0.npz" \
                                                  --epochs 20000 \
                                                  --patience 2000

train_id=$(sbatch --parsable sbatch_wrapper_allfolds.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train \
                                                        --exp-name "remove_attr_exp_0" \
                                                        --dataset "dataset-0.npz" \
                                                        --embedding "embedding-0.npz" \
                                                        --epochs 20000 \
                                                        --patience 2000)

for i in ${rounds[@]}; do
    j=`expr $i + 1`

    echo sbatch --parsable --depend=afterok:${train_id} sbatch_wrapper_snp_removal.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train  \
                                                                                      --exp-name "remove_attr_exp_${i}" \
                                                                                      --genotypes "dataset-${i}.npz" \
                                                                                      --dataset "dataset-${i}.npz" \
                                                                                      --embedding "embedding-${i}.npz" \
                                                                                      --percentile-to-remove 85 \
                                                                                      --dataset-out "dataset-${j}.npz" \
                                                                                      --embedding-out "embedding-${j}.npz" \
                                                                                      --which-fold 0 1 2 3 4

    rm_id=$(sbatch --parsable --depend=afterok:${train_id} sbatch_wrapper_snp_removal.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train  \
                                                                                         --exp-name "remove_attr_exp_${i}" \
                                                                                         --genotypes "dataset-${i}.npz" \
                                                                                         --dataset "dataset-${i}.npz" \
                                                                                         --embedding "embedding-${i}.npz" \
                                                                                         --percentile-to-remove 85 \
                                                                                         --dataset-out "dataset-${j}.npz" \
                                                                                         --embedding-out "embedding-${j}.npz" \
                                                                                         --which-fold 0 1 2 3 4)

    wait

    echo sbatch --parsable --depend=afterok:${rm_id} sbatch_wrapper_allfolds.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train \
                                                                                --exp-name "remove_attr_exp_${j}" \
                                                                                --dataset "dataset-${j}.npz" \
                                                                                --embedding "embedding-${j}.npz" \
                                                                                --epochs 20000 \
                                                                                --patience 2000

    train_id=$(sbatch --parsable --depend=afterok:${rm_id} sbatch_wrapper_allfolds.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train \
                                                                                      --exp-name "remove_attr_exp_${j}" \
                                                                                      --dataset "dataset-${j}.npz" \
                                                                                      --embedding "embedding-${j}.npz" \
                                                                                      --epochs 20000 \
                                                                                      --patience 2000)

    wait

    echo "submitted round ${i} of SNP reduction"
done
