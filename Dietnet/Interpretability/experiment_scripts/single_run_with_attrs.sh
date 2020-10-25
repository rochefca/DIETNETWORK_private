#!/bin/bash
echo Running on fold: $1 with GPU $2 during iteration $3

export CUDA_VISIBLE_DEVICES=${2}
echo device=${CUDA_VISIBLE_DEVICES}
    
python ../../train.py --exp-path /mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/results/DietNetworks_experiments/remove_largest_attr_enc_0_baseline_all_folds \
                      --exp-name remove_attr_exp_${3} \
                      --dataset "dataset-${3}.npz" \
                      --embedding "embedding-${3}.npz" \
                      --which-fold $1 \
                      --patience 75

python ../../make_attributions.py --exp-path /mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/results/DietNetworks_experiments/remove_largest_attr_enc_0_baseline_all_folds \
                                  --exp-name remove_attr_exp_${3} \
                                  --dataset "dataset-${3}.npz" \
                                  --embedding "embedding-${3}.npz" \
                                  --which-fold ${1} \
                                  --batch_size 8

echo rm /mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/results/DietNetworks_experiments/remove_largest_attr_enc_0_baseline_all_folds/remove_attr_exp_${3}/remove_attr_exp_${3}_fold${1}/attrs.h5 # not needed
