#!/bin/bash -e

# Download our HICO-DET annotations in COCO's format
mkdir output
echo "Downlaoding our COCO's format HICO-DET annotations ..."
gdown "https://drive.google.com/uc?id=1J-C2z9ZhJCJd3e3MwpgdXEqgvxL5stue" -O output/hico_det_pretrained.pkl