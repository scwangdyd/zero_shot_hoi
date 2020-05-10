#!/bin/bash -e

# Download our pretrained model on HICO-DET dataset.
mkdir ../output
echo "Downlaoding our COCO's format HICO-DET annotations ..."
gdown "https://drive.google.com/uc?id=1J-C2z9ZhJCJd3e3MwpgdXEqgvxL5stue" -O ../output/hico_det_pretrained.pkl