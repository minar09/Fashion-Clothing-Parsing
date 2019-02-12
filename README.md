# Fashion parsing models in TensorFlow
1. Tensorflow implementation of [Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/pdf/1605.06211v1.pdf) (FCNs).
2. TensorFlow implementation of [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

The implementation is largely based on the reference code provided by the authors of the paper [link](https://github.com/shelhamer/fcn.berkeleyvision.org). 
1. [Prerequisites](#prerequisites)
2. [Training](#training)

## Prerequisites
 - pydensecrf installation in windows with conda: `conda install -c conda-forge pydensecrf`. For linux, use pip or something.

## Training
 - To train model simply execute `python FCN.py` or `python UNet.py`
 - To visualize results for a random batch of images use flag `--mode=visualize`
 - To test and evaluate results use flag `--mode=test`
 - `debug` flag can be set during training to add information regarding activations, gradients, variables etc.
