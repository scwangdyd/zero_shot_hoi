**This code is for zero-shot human-object interaction detection (ZSHOI)**
# ZSHOI

This implements our paper "Discovering Human Interactions with Novel Objects via Zero-Shot Learning", in CVPR, 2020.

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
This example is provided for training the human-object region proposals network (note: not for the interacting object detection or HOI detection). The HORPN is used as the first stage of two-stage detectors to generate region proposals for interacting objects. This example will train the model on the `vcoco_train_known` set which includes only the images and annotations of known objects. Please hard-code the path to images and annotation files in `lib/data/datasets/builtin.py` before runing the code.

```
# To train HORPN
python train_net.py --num-gpus 2 \
  --config-file configs/horpn_only.yaml OUTPUT_DIR ./output/horpn_only
```

To run inference on `vcoco_val` which includes images of both known and novel objects. Using multiple GPUs can reduce the total inference time.

```
# To run inference to evaluate HORPN
python train_net.py --eval-only --num-gpus 2 \
  --config-file configs/horpn_only.yaml \
  MODEL.WEIGHTS ./output/horpn_only/model_final.pth \
  OUTPUT_DIR ./output/horpn_only
```

**Expected results**
- Inference time should around 0.168s/image (on V100 GPU)
- The evaluation results of generated proposals will be listed, e.g, AR@100, AR@500, Recall(IoU=0.5)@100, Recall(IoU=0.5)@500
  | Expected results | Recall(IoU=0.5)@100 | Recall(IoU=0.5)@500 |
  | :--- | :---: | :---: |
  | Known objects | 92.34 | 96.53 |
  | Novel objects | 81.64 | 92.42 |

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
