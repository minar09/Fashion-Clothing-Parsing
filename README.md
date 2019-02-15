# Fashion parsing models in TensorFlow
1. Tensorflow implementation of [Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/pdf/1605.06211v1.pdf) (FCNs).
2. TensorFlow implementation of [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

The implementation is largely based on the reference code provided by the authors of the paper [link](https://github.com/shelhamer/fcn.berkeleyvision.org). 
1. [Prerequisites](#prerequisites)
2. [Training](#training)
2. [Testing](#testing)
2. [Visualizing](#visualizing)

## Prerequisites
 - pydensecrf installation in windows with conda: `conda install -c conda-forge pydensecrf`. For linux, use pip: `pip install pydensecrf`.
 - Check dataset directory in `read_dataset` function of corresponding data reading script, for example, for LIP dataset, check paths in `read_LIP_data.py` and modify as necessary.

## Training
 - To train model simply execute `python FCN.py` or `python UNet.py`
 - You can add training flag as well: `python FCN.py --mode=train`
 - `debug` flag can be set during training to add information regarding activations, gradients, variables etc.

## Testing
 - To test and evaluate results use flag `--mode=test`
 - After testing and evaluation is complete, final results will be printed in the console, and the corresponding files will be saved in the "logs" directory.
 
## Visualizing
 - To visualize results for a random batch of images use flag `--mode=visualize`
 