#!/bin/bash

# do NOT run in interactive mode
module load python/3.6
virtualenv --no-download dietnetwork

source dietnetwork/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r /lustre03/project/6009524/cam27/DIETNET_EXP/MATTHEW_DATA/DietnetEnv_requirements.txt
