# Zero-Shot Human-Object Interaction Detection

Discovering human interaction with novel objects via zero-shot learning, 2019.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

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

## Training a Model

#### 1. Human-Object Region Proposals Network (HORPN) only
This example is provided for training the human-object region proposals network (i.e., not for the interactive object detection and interaction detection but only generating region proposals of human-interacting objects). The HORPN is used as the first stage of two-stage detectors (e.g., Faster R-CNN). Here we use Resnet-50-FPN as backbone. The model will be trained on the `VCOCO_train_seen` set which includes only the training images of seen objects. Please refer to the file `configs/horpn_only.yaml` for more configuration details. 

```
# To train HORPN (only)
python tools/train_net.py \
  --cfg configs/horpn_only.yaml \
  OUTPUT_DIR res/horpn_only
```

To run inference on `VCOCO_val` which includes images of both seen and novel/seen objects. 

```
# To run inference with HORPN (only)
python tools/test_net.py \
  --cfg configs/horpn_only.yaml \
  TEST.WEIGHTS res/horpn_only/vcoco_train_seen/generalized_rcnn/model_final.pth \
  OUTPUT_DIR res/horpn_only
```

**Expected results**
- The generated region proposals (`proposals.pkl`) will be saved under `res/horpn_only/vcoco_val`
- Inference time should around 0.168s/image (on V100 GPU)
- The evaluation results of generated proposals will be listed, e.g, AR@100, AR@500, Recall(IoU=0.5)@100, Recall(IoU=0.5)@500

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
