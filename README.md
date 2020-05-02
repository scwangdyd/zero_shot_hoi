**This code is for zero-shot human-object interaction detection (ZSHOI)**
# ZSHOI

This is the implementation of our paper "Discovering Human Interactions with Novel Objects via Zero-Shot Learning", in CVPR, 2020.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installation

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Training a model and running inference

### 1. Human-Object Region Proposals Network (HORPN) only
This example is provided for training the human-object region proposals network (note: not for the interacting object detection or HOI detection). The HORPN is used as the first stage of our full model to generate region proposals for interacting objects. This example will train the model on `vcoco_train_known` set which includes only the images and annotations of known objects. **Please hard-code the path to images and annotation files in `lib/data/datasets/builtin.py` before runing the code.**

```
# To train HORPN
python train_net.py --num-gpus 2 \
  --config-file configs/horpn_only.yaml OUTPUT_DIR ./output/horpn_only
```

To run inference on `vcoco_val` which includes both known and novel objects.

```
# To run inference to evaluate HORPN. Using multiple GPUs can reduce the total inference time.
python train_net.py --eval-only --num-gpus 2 \
  --config-file configs/horpn_only.yaml \
  MODEL.WEIGHTS ./output/horpn_only/model_final.pth \
  OUTPUT_DIR ./output/horpn_only
```

**Expected results**
- Inference time should around 0.069s/image (on V100 GPU)
- The evaluation results of generated proposals will be listed, e.g, AR, Recall
    | Expected results | Recall(IoU=0.5)@100 | Recall(IoU=0.5)@500 |
    | :--- | :---: | :---: |
    | Known objects | 92.34 | 96.53 |
    | Novel objects | 81.64 | 92.42 |

### 2. Interacting Object Detection
The following examples train a model to detecting interacting objects. In this case, objects but not interacting with humans will not be detected. We train the model on `hico-det_train` set using all 80 MS-COCO object categories.

```
# Interacting object detection
python train_net.py --num-gpus 2 \
  --config-file configs/HICO-DET/interacting_objects_R_50_FPN.yaml OUTPUT_DIR ./output/interacting_objects
```

To run inference on `hico-det_test`. We use COCO's metrics and APIs to conduct evaluation. Note that the ground-truth only includes interacting objects. Non-interacting objects will be seen as background.

```
# To run inference. Using multiple GPUs can reduce the total inference time.
python train_net.py --eval-only --num-gpus 2 \
  --config-file configs/HICO-DET/interacting_objects_R_50_FPN.yaml \
  MODEL.WEIGHTS ./output/HICO_interacting_objects/model_final.pth \
  OUTPUT_DIR ./output/HICO_interacting_objects
```

**Expected results**
- Inference time should around 0.098s/image (on V100 GPU)
- The results of COCO's metrics will be listed, e.g, per-class Average Precision (AP)
    | Expected results | AP | AP50 | AP75 |
    | :--- | :---: | :---: | :---: |
    | Interacting objects |  |  |

### 3. HOI Detection
The following examples train a model to detect human-object interactions using `hico-det_train` set. Here we use all 80 MS-COCO object categories.

```
# Interacting object detection
python train_net.py --num-gpus 2 \
  --config-file configs/HICO-DET/interaction_R_50_FPN.yaml OUTPUT_DIR ./output/HICO_interaction
```

To run inference on `hico-det_test`. This code will trigger the official HICO-DET MATLAB evaluation. Please make sure MATLAB is available in your machine and check the hard-coded path `cfg.TEST.HICO_OFFICIAL_ANNO_FILE` and `cfg.TEST.HICO_OFFICIAL_BBOX_FILE` can access direct to the original HICO-DET annotation files.

```
# To run inference. Using multiple GPUs can reduce the total inference time.
python train_net.py --eval-only --num-gpus 2 \
  --config-file configs/HICO-DET/interaction_R_50_FPN.yaml \
  MODEL.WEIGHTS ./output/interaction_R_50_FPN.yaml/model_final.pth \
  OUTPUT_DIR ./output/interaction_R_50_FPN.yaml
```

**Expected results**
- Inference time should around 0.102s/image (on V100 GPU)
- It will list the results of COCO's metrics on interacting object detection as above.
- The results of HICO-DET's metrics will be listed, e.g,
    | Expected results |  full   |   rare   |  non-rare |
    | :--- | :---: | :---: | :---: |
    | Default mAP |  |  | |

## Citation
If you use this code in your research or wish to refer to the baseline results published, please use the following BibTeX entry.
```
@InProceedings{Wang_2020_CVPR,
author = {Wang, Suchen and Yap, Kim-Hui and Yuan, Junsong and Tan, Yap-Peng},
title = {Discovering Human Interactions with Novel Objects via Zero-Shot Learning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
