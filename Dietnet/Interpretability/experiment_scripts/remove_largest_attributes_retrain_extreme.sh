#!/bin/bash

# Here we start with the output of the previous round of SNP removal.
# We now are more aggressive, removing 50% of SNPs from the previous round
which_fold=0

# Now run on data with top 50%th percentile attributed SNPs removed. Repeat
rounds=( 9 10 11 12 13 )

for i in ${rounds[@]}; do
    echo "performing round ${i} of SNP reduction"
    j=`expr $i + 1`

    python create_dataset_less_snps.py --exp-path '/home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/' \
                                       --exp-name "remove_attr_exp_${i}" \
                                       --genotypes "dataset-${i}.npz" \
                                       --dataset "dataset-${i}.npz" \
                                       --embedding "embedding-${i}.npz" \
                                       --percentile-to-remove 50 \
                                       --dataset-out "dataset-${j}.npz" \
                                       --embedding-out "embedding-${j}.npz" \
                                       --which-fold ${which_fold}

    export CUDA_VISIBLE_DEVICES="2"
    echo $CUDA_VISIBLE_DEVICES

    python ../../train.py --exp-path /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/ \
                          --exp-name "remove_attr_exp_${j}" \
                          --dataset "dataset-${j}.npz" \
                          --embedding "embedding-${j}.npz" \
                          --which-fold ${which_fold} \
                          --patience 300 # train for longer

    export CUDA_VISIBLE_DEVICES="0,1,2"
    echo $CUDA_VISIBLE_DEVICES

    python ../../make_attributions.py --exp-path /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/ \
                                      --exp-name "remove_attr_exp_${j}" \
                                      --dataset "dataset-${j}.npz" \
                                      --embedding "embedding-${j}.npz" \
                                      --which-fold ${which_fold} \
                                      --batch_size 24

    rm /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/remove_attr_exp_${j}/remove_attr_exp_${j}_fold${which_fold}/attrs.h5 # not needed
    echo "finished round ${i} of SNP reduction"
done
