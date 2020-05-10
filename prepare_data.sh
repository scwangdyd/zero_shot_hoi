#!/bin/bash -e
# Download some files needed for running our code.

# Download HICO-DET dataset. Comment out the following lines if you already have it.
echo "Downlaoding HICO-DET dataset ..."
gdown "https://drive.google.com/uc?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk" -O datasets/hico_det.tar.gz
tar -xvf datasets/hico_det.tar.gz --directory ./datasets

# Download our HICO-DET annotations in COCO's format
echo "Downlaoding our COCO's format HICO-DET annotations ..."
gdown "https://drive.google.com/uc?id=1lj-2C8WRHX3SJcRStPZJorJwTVpuWCBa" -O datasets/hico_det/annotations/instances_hico_train.json
gdown "https://drive.google.com/uc?id=1x0Eso9J5v_5Wb1Aa9xnFWyinowaHJn1b" -O datasets/hico_det/annotations/instances_hico_test.json

# Download Detectron2 Faster R-CNN with ResNet50-FPN
mkdir output
echo "Downloading pre-trained Faster R-CNN with ResNet50-FPN"
wget "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl" -O ./output/model_final_280758.pkl

# Download Glove vector representations for words. Refer to https://nlp.stanford.edu/projects/glove/.
echo "Downloading Glove work embeddings ..."
mkdir datasets/Glove
wget "http://nlp.stanford.edu/data/glove.6B.zip" -O datasets/Glove/glove.6B.zip
cd datasets/Glove
unzip "glove.6B.zip"
cd ../..