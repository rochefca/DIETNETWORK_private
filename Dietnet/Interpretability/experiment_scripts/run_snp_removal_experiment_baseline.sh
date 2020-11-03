#!/bin/bash
#export SLURM_ACCOUNT="def-hussinju"
#export SBATCH_ACCOUNT=$SLURM_ACCOUNT

# copy over this file for future reference
cp run_snp_removal_experiment_baseline.sh ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train

rounds=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 )
folds=( 0 1 2 3 4 )

# remove SNPs randomly (starting with all SNPs)
echo sbatch --parsable sbatch_wrapper_snp_removal.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train  \
                                                     --exp-name "remove_attr_exp_0" \
                                                     --genotypes "dataset-0.npz" \
                                                     --dataset "dataset-0.npz" \
                                                     --embedding "embedding-0.npz" \
                                                     --percentile-to-remove 85 \
                                                     --random-snp-removal \
                                                     --dataset-out "dataset-1-baseline.npz" \
                                                     --embedding-out "embedding-1-baseline.npz" \
                                                     --which-fold 0 1 2 3 4

rm_id=$(sbatch --parsable sbatch_wrapper_snp_removal.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train  \
                                                        --exp-name "remove_attr_exp_0" \
                                                        --genotypes "dataset-0.npz" \
                                                        --dataset "dataset-0.npz" \
                                                        --embedding "embedding-0.npz" \
                                                        --percentile-to-remove 85 \
                                                        --random-snp-removal \
                                                        --dataset-out "dataset-1-baseline.npz" \
                                                        --embedding-out "embedding-1-baseline.npz" \
                                                        --which-fold 0 1 2 3 4)

for i in ${rounds[@]}; do
    j=`expr $i + 1`
    
    # train networks on dataset and embedding generated from round i of random SNP removal
    echo sbatch --parsable --depend=afterok:${rm_id} sbatch_wrapper_allfolds.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train \
                                                                                --exp-name "remove_attr_exp_${i}_baseline" \
                                                                                --dataset "dataset-${i}-baseline.npz" \
                                                                                --embedding "embedding-${i}-baseline.npz" \
                                                                                --epochs 20000 \
                                                                                --patience 2000

    train_id=$(sbatch --parsable --depend=afterok:${rm_id} sbatch_wrapper_allfolds.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train \
                                                                                      --exp-name "remove_attr_exp_${i}_baseline" \
                                                                                      --dataset "dataset-${i}-baseline.npz" \
                                                                                      --embedding "embedding-${i}-baseline.npz" \
                                                                                      --epochs 20000 \
                                                                                      --patience 2000)

    wait

    # generate dataset and embedding generated for round j=i+1 of random SNP removal.
    # we remove random 15 percentag of SNPs from dataset-i.npz (recall this was produced during round i-1 of NON-RANDOM SNP removal)
    # see run_snp_removal_experiment.sh script for details of NON-RANDOM SNP removal
    echo sbatch --parsable --depend=afterok:${train_id} sbatch_wrapper_snp_removal.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train  \
                                                                                      --exp-name "remove_attr_exp_${i}" \
                                                                                      --genotypes "dataset-${i}.npz" \
                                                                                      --dataset "dataset-${i}.npz" \
                                                                                      --embedding "embedding-${i}.npz" \
                                                                                      --percentile-to-remove 85 \
                                                                                      --random-snp-removal \
                                                                                      --dataset-out "dataset-${j}-baseline.npz" \
                                                                                      --embedding-out "embedding-${j}-baseline.npz" \
                                                                                      --which-fold 0 1 2 3 4

    rm_id=$(sbatch --parsable --depend=afterok:${train_id} sbatch_wrapper_snp_removal.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train  \
                                                                                         --exp-name "remove_attr_exp_${i}" \
                                                                                         --genotypes "dataset-${i}.npz" \
                                                                                         --dataset "dataset-${i}.npz" \
                                                                                         --embedding "embedding-${i}.npz" \
                                                                                         --percentile-to-remove 85 \
                                                                                         --random-snp-removal \
                                                                                         --dataset-out "dataset-${j}-baseline.npz" \
                                                                                         --embedding-out "embedding-${j}-baseline.npz" \
                                                                                         --which-fold 0 1 2 3 4)

    wait

    echo "submitted round ${i} of SNP reduction"
done

# train networks on dataset and embedding generated from final round of random SNP removal
echo sbatch --parsable --depend=afterok:${rm_id} sbatch_wrapper_allfolds.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train \
                                                                            --exp-name "remove_attr_exp_21_baseline" \
                                                                            --dataset "dataset-21-baseline.npz" \
                                                                            --embedding "embedding-21-baseline.npz" \
                                                                            --epochs 20000 \
                                                                            --patience 2000

train_id=$(sbatch --parsable --depend=afterok:${rm_id} sbatch_wrapper_allfolds.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_long_train \
                                                                                  --exp-name "remove_attr_exp_21_baseline" \
                                                                                  --dataset "dataset-21-baseline.npz" \
                                                                                  --embedding "embedding-21-baseline.npz" \
                                                                                  --epochs 20000 \
                                                                                  --patience 2000)
