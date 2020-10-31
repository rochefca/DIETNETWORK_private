#!/bin/bash

which_fold=0

export CUDA_VISIBLE_DEVICES="2"
echo $CUDA_VISIBLE_DEVICES

python ../../train.py --exp-path /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/ \
                      --exp-name remove_attr_exp_0 \
                      --which-fold ${which_fold} \
                      --patience 75

export CUDA_VISIBLE_DEVICES="0,1,2"
echo $CUDA_VISIBLE_DEVICES

python ../../make_attributions.py --exp-path /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/ \
                                  --exp-name remove_attr_exp_0 \
                                  --which-fold ${which_fold} \
                                  --batch_size 24

rm /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/remove_attr_exp_0/remove_attr_exp_0_fold${which_fold}/attrs.h5 # not needed

#  zeroth iteration is original dataset and embedding (i.e. we did not remove any SNPs yet)
cp /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/dataset.npz /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/dataset-0.npz
cp /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/embedding.npz /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/embedding-0.npz

# Now run on data with the top 29443 attributed SNPs removed. Repeat
rounds=( 0 1 2 3 4 5 6 7 8 )

for i in ${rounds[@]}; do
    echo "performing round ${i} of SNP reduction"
    j=`expr $i + 1`

    python create_dataset_less_snps.py --exp-path '/home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP/' \
                                       --exp-name "remove_attr_exp_${i}" \
                                       --genotypes "dataset-${i}.npz" \
                                       --dataset "dataset-${i}.npz" \
                                       --embedding "embedding-${i}.npz" \
                                       --num-to-remove 29443 \
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
                          --patience 75

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
