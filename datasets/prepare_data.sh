#!/bin/bash -e
# Download some files needed for running our code.

# Download HICO-DET dataset. Comment out the following lines if you already have it.
echo "Downlaoding HICO-DET dataset ..."
gdown "https://drive.google.com/uc?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk" -O hico_det.tar.gz
tar -xvf hico_det.tar.gz

# Download our HICO-DET annotations in COCO's format
echo "Downlaoding our COCO's format HICO-DET annotations ..."
mkdir hico_20160224_det/annotations
gdown "https://drive.google.com/uc?id=1lj-2C8WRHX3SJcRStPZJorJwTVpuWCBa" -O hico_20160224_det/annotations/instances_hico_train.json
gdown "https://drive.google.com/uc?id=1x0Eso9J5v_5Wb1Aa9xnFWyinowaHJn1b" -O hico_20160224_det/annotations/instances_hico_test.json
gdown "https://drive.google.com/uc?id=11e5mymZWOIc-2tf2xSoE4AqDf21qaUul" -O hico_20160224_det/annotations/instances_hico_train_seen.json

# Download VCOCO annotations
mkdir vcoco
mkdir vcoco/annotations
gdown "https://drive.google.com/uc?id=1IS0iIQwMJqD0IlVtpTtuKzN-ztqVjGX3" -O vcoco/annotations/instances_vcocotrain_known.json
gdown "https://drive.google.com/uc?id=1GsNSyr05jsKRKj_EOQBQHrqgYH26R4Ab" -O vcoco/annotations/instances_vcocotrain.json
gdown "https://drive.google.com/uc?id=1WuQ0eimCl47sHKxxEONjZJ1PuSjrKLvg" -O vcoco/annotations/instances_vcocoval_known.json
gdown "https://drive.google.com/uc?id=1tWwHw__py8BMQJhFnpyWXRdm-XoSWhWk" -O vcoco/annotations/instances_vcocoval.json
gdown "https://drive.google.com/uc?id=138NPkxyddKrgqcBAB0NBqMjXQExZxIcq" -O vcoco/annotations/instances_vcocotest.json

# Download Detectron2 Faster R-CNN with ResNet50-FPN
mkdir ../output
echo "Downloading pre-trained Faster R-CNN with ResNet50-FPN"
wget "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl" -O ../output/model_final_280758.pkl

# Download Glove vector representations for words. Refer to https://nlp.stanford.edu/projects/glove/.
echo "Downloading Glove work embeddings ..."
mkdir Glove
wget "http://nlp.stanford.edu/data/glove.6B.zip" -O ./Glove/glove.6B.zip
cd Glove
unzip "glove.6B.zip"
cd ..
