#!/bin/bash

folds=( 0 1 2 3 4 )
rounds=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 )

#  zeroth iteration is original dataset and embedding (i.e. we did not remove any SNPs yet)
cp /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/dataset.npz /mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/results/DietNetworks_experiments/remove_largest_attr_enc_0_baseline_all_folds/dataset-0.npz
cp /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/embedding.npz /mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/results/DietNetworks_experiments/remove_largest_attr_enc_0_baseline_all_folds/embedding-0.npz

#  copy other stuff needed
cp /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/folds_indexes.npz /mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/results/DietNetworks_experiments/remove_largest_attr_enc_0_baseline_all_folds/folds_indexes.npz

for f in ${folds[@]}; do
    bash single_run_with_attrs.sh ${f} ${f} 0 &
done

wait

for i in ${rounds[@]}; do
    echo "performing round ${i} of SNP reduction"
    j=`expr $i + 1`

    python create_dataset_less_snps.py --exp-path '/mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/results/DietNetworks_experiments/remove_largest_attr_enc_0_baseline_all_folds'  \
                                   --exp-name "remove_attr_exp_${i}" \
                                   --genotypes "dataset-${i}.npz" \
                                   --dataset "dataset-${i}.npz" \
                                   --embedding "embedding-${i}.npz" \
                                   --percentile-to-remove 85 \
                                   --dataset-out "dataset-${j}.npz" \
                                   --embedding-out "embedding-${j}.npz" \
                                   --which-fold 0 1 2 3 4

    for f in ${folds[@]}; do
        bash single_run_with_attrs.sh ${f} ${f} $j &
    done
    wait
    
    echo "finished round ${i} of SNP reduction"
done
