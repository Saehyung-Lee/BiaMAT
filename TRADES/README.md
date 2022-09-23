# Biased Multi-domain Adversarial Training

This repository is the official implementation of Biased Multi-domain Adversarial Training (BiaMAT).  
The implementation is based on the official code of TRADES (https://github.com/yaodongyu/TRADES).

## Prerequisites

* Python 3
* Pytorch (0.4.1)
* Numpy
* CUDA
* tqdm

## Training

To train the BiaMAT model in the paper, run this command:

```train
python train_BiaMAT.py --model-dir <name> --primary cifar10 --alpha 0.5 --gamma 0.5 --warmup 5 --data-dir <path to dir containing cifar-10-batches-py> --aux-dataset-dir <path to Imagenet32_train>
```

## Evaluation

To evaluate the model on CIFAR-10, run:

```eval
python evaluate.py --model-path <model path> --primary cifar10 --auxiliary imagenet --num-steps 100 --BiaMAT --random --data-dir <path to dir containing cifar-10-batches-py>
```
