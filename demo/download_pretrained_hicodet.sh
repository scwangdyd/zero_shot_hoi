#!/bin/bash -e

# Download our pretrained model on HICO-DET dataset.
mkdir ../output
echo "Downloading our pretrained models ..."
gdown "https://drive.google.com/uc?id=1J-C2z9ZhJCJd3e3MwpgdXEqgvxL5stue" -O ../output/hico_det_pretrained.pkl
gdown "https://drive.google.com/uc?id=13aytw34aNUYlSp9_ASMBcqX3nrvC-Olf" -O ../output/hico_det_pretrained_agnostic.pkl
