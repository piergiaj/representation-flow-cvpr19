# Representation Flow for Action Recognition

This repository contains the code for our [CVPR 2019 paper](https://arxiv.org/abs/1810.01455), [Project Page](https://piergiaj.github.io/rep-flow-site/):

    AJ Piergiovanni and Michael S. Ryoo
    "Representation Flow for Action Recognition"
    in CVPR 2019

If you find the code useful for your research, please cite our paper:

        @inproceedings{repflow2019,
              title={Representation Flow for Action Recognition},
              booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
              author={AJ Piergiovanni and Michael S. Ryoo},
              year={2019}
        }


# Representation Flow Layer
![tsf](/examples/flow-layer.png?raw=true "repflow")

We introduce the representation flow layer, which can be found in [rep_flow_layer.py](rep_flow_layer.py). This layer iteratively estimates the flow, can be applied to CNN feature maps, and is fully learnable to maximize classification performance.


# Activity Recognition Experiments
![model overview](/examples/flow-in-network.png?raw=true "model overview")

We include our pretrained models for HMDB and Kinetics in [models/](models).

We tested our models on HMDB and Kinetics and provide the files and class labes used in [data](/data/).

## Results

|  Method | Kinetics-400  |  HMDB | Runtime | 
| ------------- | ------------- | ----------- | ------- | 
| 2D Two-Stream | 64.5  | 66.6  | 8546ms  |
| TVNet (+RGB)  | -     | 71.0  | 785ms |
| (2+1)D Two-Stream | 75.4 | 78.7 | 8623ms |
| I3D Two-stream | 74.2 | 80.7 | 9354ms |
| (2+1)D + Rep-Flow | 75.5 | 77.1 | 622ms |
| (2+1)D + Flow-of-flow | 77.1 | 81.1 | 654ms |





# Visualization of learned flows
Examples of representation flows for various actions. The representation flow is computed after the 3rd residual block and captures some sematic motion information. At this point, the representations are low dimensional (28x28).


<img src="https://piergiaj.github.io/rep-flow-site/box_flow_c15.gif"> <img src="https://piergiaj.github.io/rep-flow-site/swing_flow_c1.gif"> <img src="https://piergiaj.github.io/rep-flow-site/handstand_flow_c21.gif">


Examples of representation flows for different channels for "clapping." Some channels capture the hand motion, while other channels focus on different features/motion patterns not present in this clip.

<img src="https://piergiaj.github.io/rep-flow-site/clap_flow_c8.gif"> <img src="https://piergiaj.github.io/rep-flow-site/clap_flow_c16.gif"> <img src="https://piergiaj.github.io/rep-flow-site/clap_flow_c21.gif">



# Requirements

Our code has been tested on Ubuntu 14.04 and 16.04 using python 3.6, [PyTorch](pytorch.org) version 0.4.1 with a Titan X GPU. Our data loading uses [lintel](https://github.com/dukebw/lintel) to extract frames from the video.


# Setup

1. Download the code ```git clone https://github.com/piergiaj/representation-flow-cvpr19.git```

2. Pre-trained models are avialable [here](https://drive.google.com/drive/folders/1mzTqycnew1aXnTza6MSQxmmIHCk2LGEn?usp=sharing).

3. [train_model.py](train_model.py) contains the code to train and evaluate models.
