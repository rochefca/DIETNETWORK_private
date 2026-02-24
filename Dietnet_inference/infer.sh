#!/bin/bash

# bash infer.sh /path/to/plink /path/to/dietnet_code /path/to/dietnet_files /path/to/plink_test output_name_prefix

# Assign command-line arguments to variables
plink_path="$1"
dietnet_code_path="$2"
dietnet_files_path="$3"
plink_test="$4"
output_name="$5"

#-----------------------------------------------------------------------------
# Pre-processing the plink files to get the same SNPs as used to train the DN
#-----------------------------------------------------------------------------
echo "---"
echo "Pre-processing plink files to match SNPs used to train the Dietnet"
echo ""

# Extract dietnet snps from the plink test file
${plink_path} \
    --bfile $plink_test \
    --extract ${dietnet_files_path}/DN_SNPS/dietnet_snps.txt \
    --make-bed \
    --real-ref-alleles \
    --out ${output_name}_dietnetsnps_extracted

# Check if the .bim file was created (no overlap if missing)
if [ ! -f ${output_name}_dietnetsnps_extracted.bim ]; then
    echo "Exiting program: No overlapping SNPs found between the plink query files and Diet Network SNPs."
    exit 1
fi

# Dietnet snps found in test set
cut -f2 ${output_name}_dietnetsnps_extracted.bim > ${output_name}_dietnetsnps_extractedsnps.txt

# Dietnet snps not found in test set
comm -23 <(sort ${dietnet_files_path}/DN_SNPS/dietnet_snps.txt) <(sort ${output_name}_dietnetsnps_extractedsnps.txt) > ${output_name}_dietnetsnps_notextractedsnps.txt

# Fill extracted plink with SNPs (set as missing) to match SNPs used to train the DN
# (This is done by merging the extracted plink file with a dummy plink file
# of one 1KGP sample with SNPs used to train the dietnet, 
# using merge-mode 5  which fills in missing SNPs)
cat ${output_name}_dietnetsnps_extracted.fam ${dietnet_files_path}/DN_SNPS/dummy.fam | cut -f1,2 > sample_order.txt
${plink_path} \
    --bfile ${output_name}_dietnetsnps_extracted \
    --bmerge ${dietnet_files_path}/DN_SNPS/dummy \
    --merge-mode 5 \
    --real-ref-alleles \
    --indiv-sort file sample_order.txt \
    --make-bed \
    --out ${output_name}_dietnetsnps_extracted_missingfilled

# Additive encoding (and remove dummy sample)
${plink_path} \
    --bfile ${output_name}_dietnetsnps_extracted_missingfilled \
    --remove ${dietnet_files_path}/DN_SNPS/dummy.sample \
    --recode A \
    --real-ref-alleles \
    --out ${output_name}_dietnetsnps_extracted_missingfilled

# Scaling for missing SNPs
scale=$(echo "$(wc -l < ${dietnet_files_path}/DN_SNPS/dietnet_snps.txt) / $(wc -l < ${output_name}_dietnetsnps_extractedsnps.txt)" | bc -l)
echo "Scaling factor for missing SNPs: $scale"

# Remove files that are not needed for inference
rm ${output_name}_dietnetsnps_extracted.*
rm ${output_name}_dietnetsnps_extracted_missingfilled.b*
rm ${output_name}_dietnetsnps_extracted_missingfilled.fam
rm ${output_name}_dietnetsnps_extracted_missingfilled.nosex
rm ${output_name}_dietnetsnps_extracted_missingfilled.log
rm sample_order.txt

echo "---"
echo ""

#-----------------------------------------
# Making the test set into hdf5 format
#-----------------------------------------

echo "---"
echo "Creating HDF5 dataset for inference with Dietnet"

python ${dietnet_code_path}/inference_dataset.py \
--genotypes ${output_name}_dietnetsnps_extracted_missingfilled.raw \
--scale $scale \
--ncpus 1 \
--out ${output_name}.hdf5

rm ${output_name}_dietnetsnps_extracted_missingfilled.raw

echo "---"
echo ""


#-----------------------------------------
# Inference with Dietnet
#-----------------------------------------

echo "---"
echo "Inference"

output_dir=${output_name}_DIETNET_RESULTS_BY_MODEL
if [ ! -d "$output_dir" ]; then
  mkdir $output_dir
fi

for fold in {1..5}; do
    for rep in {1..3}; do
        echo "Running inference with Diet Network model fold $fold, repeat $rep"
        python ${dietnet_code_path}/inference.py \
        --test-h5 ${output_name}.hdf5 \
        --model-param ${dietnet_files_path}/DN_MODELS/weights_model${fold}_${rep}.pt \
        --snps-emb ${dietnet_files_path}/DN_MODELS/snps_emb_model${fold}.npz \
        --norm-stats ${dietnet_files_path}/DN_MODELS/norm_stats_model${fold}.npz \
        --out ${output_dir}/${output_name}_model${fold}_${rep}_infered.npz
        echo ""
    done
done

echo "---"
echo ""

# Output file with all model predictions for each sample
python ${dietnet_code_path}/inference_combine_results.py \
--dietnet-results ${output_dir}/${output_name}_model{fold}_{rep}_infered.npz \
--out ${output_name}_dietnet_predictions.txt

echo "Inference with Dietnet completed. Predictions saved in ${output_name}_dietnet_predictions.txt"
echo ""
echo "The SNP overlap with Diet Network is: $(wc -l < ${output_name}_dietnetsnps_extractedsnps.txt) out of $(wc -l < ${dietnet_files_path}/DN_SNPS/dietnet_snps.txt) SNPs used to train the Diet Network."
echo "Note that we suggest to not consider predictions if the SNP overlap is less than 1000 SNPs."