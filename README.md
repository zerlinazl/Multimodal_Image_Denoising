# Overview

This code is based off of [`uvcgan`] [uvcgan_paper].

From the repo:
"`uvcgan` introduces an improved method to perform an unpaired image-to-image
style transfer based on a CycleGAN framework. Combined with a new hybrid
generator architecture UNet-ViT (UNet-Vision Transformer) and a self-supervised
pre-training, it achieves state-of-the-art results on a multitude of style
transfer benchmarks."

For an example of how to apply `uvcgan` to a scientific dataset, see [uvcgan4slats](https://github.com/LS4GAN/uvcgan4slats) published by the same authors.


# Installation & Requirements

## Requirements

`uvcgan` was trained using
`pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime`.

Other necessary libraries:

`torch` version: 1.10.1
`torchvision`: 0.11.2
`torchaudio`: 0.10.1
`numpy`: 1.19.5
`pillow`: 8.4.0

## Installation

To install the `uvcgan` package one may run:
```
python setup.py develop --user
```
from the `uvcgan` source tree.

## Dataset

I used the [`SIDD`](https://www.eecs.yorku.ca/~kamel/sidd/) image dataset by Abdelhamed et al. Even the small image dataset is too large to put on GitHub, but it can be downloaded from the link. UVCGAN requires that the data be restructured to the following format:

`test`

  `Domain 1 (N)`

  `Domain 2 (GT)`

`train`

  `Domain 1 (N)`

  `Domain 2 (GT)`

`val`

  `Domain 1 (N)`

  `Domain 2 (GT)`

## Model Training

To train the model, run from `uvcgan`:
```
python3 sidd.py
```

## Model Evaluation

To perform image-to-image translation with the trained models, 
`scripts/translate_images.py` is provided. Run:
```
python3 translate_images.py PATH_TO_TRAINED_MODEL -n 100
```
`-n` specifies the number of images from the test set to
translate. The generated images will be saved in
`PATH_TO_TRAINED_MODEL/evals/final/translated`

Examples provided by the original published code can be found in the `uvcgan/examples` subdirectory.




[cyclegan_repo]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
[benchmarking_repo]: https://github.com/LS4GAN/benchmarking
[uvcgan_paper]: https://arxiv.org/abs/2203.02557
[pretrained_models]: https://zenodo.org/record/6336010

