#!/bin/bash
#export SLURM_ACCOUNT="def-hussinju"
#export SBATCH_ACCOUNT=$SLURM_ACCOUNT

#  zeroth iteration is original dataset and embedding (i.e. we did not remove any SNPs yet)
cp ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_enc_0_baseline_all_folds/dataset.npz ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_enc_0_baseline_all_folds/dataset-0.npz
cp ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_enc_0_baseline_all_folds/embedding.npz ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_enc_0_baseline_all_folds/embedding-0.npz

rounds=( 0 1 ) # 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 )
folds=( 0 1 2 3 4 )

for i in ${rounds[@]}; do
    echo "performing round ${i} of SNP reduction"
    j=`expr $i + 1`
    sbatch sbatch_wrapper_allfolds.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_enc_0_baseline_all_folds \
                                      --exp-name "remove_attr_exp_${i}" \
                                      --dataset "dataset-${i}.npz" \
                                      --embedding "embedding-${i}.npz" \
                                      --epochs 2 \
                                      --patience 2
    wait

    sbatch sbatch_wrapper_snp_removal.sh --exp-path ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_enc_0_baseline_all_folds  \
                                         --exp-name "remove_attr_exp_${i}" \
                                         --genotypes "dataset-${i}.npz" \
                                         --dataset "dataset-${i}.npz" \
                                         --embedding "embedding-${i}.npz" \
                                         --percentile-to-remove 85 \
                                         --dataset-out "dataset-${j}.npz" \
                                         --embedding-out "embedding-${j}.npz" \
                                         --which-fold 0 1 2 3 4
    wait

    for k in ${folds[@]}; do
        rm ${SCRATCH}/exp_results/DIETNETWORK/1000G/remove_largest_attr_enc_0_baseline_all_folds/remove_attr_exp_${i}/remove_attr_exp_${i}_fold${k}/attrs.h5 # not needed
    done

    echo "finished round ${i} of SNP reduction"
done
