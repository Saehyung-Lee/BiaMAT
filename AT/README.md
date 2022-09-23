# Biased Multi-domain Adversarial Training

This repository is the official implementation of Biased Multi-domain Adversarial Training (BiaMAT).  
The implementation is based on (https://github.com/MadryLab/cifar10_challenge).

## Prerequisites

* Python 3
* Tensorflow (1.13)
* Numpy
* CUDA
* tqdm

## Training

To train the BiaMAT model in the paper, run this command:

```train
python train_BiaMAT.py --primary cifar10 --auxiliary imagenet --alpha 1.0 --n 4 --gamma 0.55 --warmup-epoch 5 --suffix <name>
```

> Please specify the data paths for CIFAR-10 and ImageNet32x32 in config.json

## Evaluation

To evaluate the model on CIFAR-10, run:

```eval
python evaluate.py --model-dir <model path> --steps 100 --n 4
```
