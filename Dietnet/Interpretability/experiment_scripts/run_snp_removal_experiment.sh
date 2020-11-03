#!/bin/bash
#export SLURM_ACCOUNT="def-hussinju"
#export SBATCH_ACCOUNT=$SLURM_ACCOUNT

rounds=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 )
folds=( 0 1 2 3 4 )

exp_path=${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train

cp run_snp_removal_experiment.sh ${exp_path}

#  zeroth iteration is original dataset and embedding (i.e. we did not remove any SNPs yet)
cp ${exp_path}/dataset.npz ${exp_path}/dataset-0.npz
cp ${exp_path}/embedding.npz ${exp_path}/embedding-0.npz

wait

echo sbatch --parsable sbatch_wrapper_allfolds.sh --exp-path ${exp_path} \
                                                  --exp-name "remove_attr_exp_0" \
                                                  --dataset "dataset-0.npz" \
                                                  --embedding "embedding-0.npz" \
                                                  --epochs 20000 \
                                                  --patience 2000

train_id=$(sbatch --parsable sbatch_wrapper_allfolds.sh --exp-path ${exp_path} \
                                                        --exp-name "remove_attr_exp_0" \
                                                        --dataset "dataset-0.npz" \
                                                        --embedding "embedding-0.npz" \
                                                        --epochs 20000 \
                                                        --patience 2000)

for i in ${rounds[@]}; do
    j=`expr $i + 1`

    echo sbatch --parsable --depend=afterok:${train_id} sbatch_wrapper_snp_removal.sh --exp-path ${exp_path}  \
                                                                                      --exp-name "remove_attr_exp_${i}" \
                                                                                      --genotypes "dataset-${i}.npz" \
                                                                                      --dataset "dataset-${i}.npz" \
                                                                                      --embedding "embedding-${i}.npz" \
                                                                                      --percentile-to-remove 85 \
                                                                                      --dataset-out "dataset-${j}.npz" \
                                                                                      --embedding-out "embedding-${j}.npz" \
                                                                                      --which-fold 0 1 2 3 4

    rm_id=$(sbatch --parsable --depend=afterok:${train_id} sbatch_wrapper_snp_removal.sh --exp-path ${exp_path}  \
                                                                                         --exp-name "remove_attr_exp_${i}" \
                                                                                         --genotypes "dataset-${i}.npz" \
                                                                                         --dataset "dataset-${i}.npz" \
                                                                                         --embedding "embedding-${i}.npz" \
                                                                                         --percentile-to-remove 85 \
                                                                                         --dataset-out "dataset-${j}.npz" \
                                                                                         --embedding-out "embedding-${j}.npz" \
                                                                                         --which-fold 0 1 2 3 4)

    wait

    echo sbatch --parsable --depend=afterok:${rm_id} sbatch_wrapper_allfolds.sh --exp-path ${exp_path} \
                                                                                --exp-name "remove_attr_exp_${j}" \
                                                                                --dataset "dataset-${j}.npz" \
                                                                                --embedding "embedding-${j}.npz" \
                                                                                --epochs 20000 \
                                                                                --patience 2000

    train_id=$(sbatch --parsable --depend=afterok:${rm_id} sbatch_wrapper_allfolds.sh --exp-path ${exp_path} \
                                                                                      --exp-name "remove_attr_exp_${j}" \
                                                                                      --dataset "dataset-${j}.npz" \
                                                                                      --embedding "embedding-${j}.npz" \
                                                                                      --epochs 20000 \
                                                                                      --patience 2000)

    wait

    echo "submitted round ${i} of SNP reduction"
done
