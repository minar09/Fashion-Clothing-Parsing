# Fashion parsing models in TensorFlow
This is the source code for our paper for Fashion Clothing Parsing. (Link coming soon)
1. Tensorflow implementation of [Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/pdf/1605.06211v1.pdf) (FCNs).
2. TensorFlow implementation of [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

The implementation is largely based on the reference code provided by the authors of the paper [link](https://github.com/shelhamer/fcn.berkeleyvision.org).
1. [Prerequisites](#prerequisites)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Testing](#testing)
5. [Visualizing](#visualizing)
6. [CRF](#crf)
7. [BFSCORE](#bfscore)

## Directory Structure

```bash
├── parseDemo20180417
│   ├── clothparsing.py
├── tests
│   ├── __init__.py
│   ├── gt.png
│   ├── inference.py
│   ├── inp.png
│   ├── output.png
│   └── pred.png
│   └── test_crf.py
│   └── test_labels.py
└── .gitignore
└── __init__.py
└── BatchDatasetReader.py
└── bfscore.py
└── CalculateUtil.py
└── denseCRF.py
└── EvalMetrics.py
└── FCN.py
└── function_definitions.py
└── LICENSE
└── read_10k_data.py
└── read_CFPD_data.py
└── read_LIP_data.py
└── README.md
└── requirements.txt
└── TensorflowUtils.py
└── test_human.py
└── UNet.py
└── UNetAttention.py
└── UNetMSc.py
└── UNetPlus.py
└── UNetPlusMSc.py

```

## Prerequisites
 - For required packages installation, run `pip install -r requirements.txt`
 - pydensecrf installation in windows with conda: `conda install -c conda-forge pydensecrf`. For linux, use pip: `pip install pydensecrf`.
 - Check dataset directory in `read_dataset` function of corresponding data reading script, for example, for LIP dataset, check paths in `read_LIP_data.py` and modify as necessary.

## Dataset
 - Right now, there are dataset supports for 3 datasets. Set your directory path in the corresponding dataset reader script.
 - [CFPD](https://github.com/hrsma2i/dataset-CFPD) (For preparing CFPD dataset, you can visit here: https://github.com/minar09/dataset-CFPD-windows)
 - [LIP](http://www.sysu-hcp.net/lip/)
 - 10k (Fashion)
 - If you want to use your own dataset, please create your dataset reader. (Check `read_CFPD_data.py` for example, on how to put directory and stuff)

## Training
 - To train model simply execute `python FCN.py` or `python UNet.py`
 - You can add training flag as well: `python FCN.py --mode=train`
 - `debug` flag can be set during training to add information regarding activations, gradients, variables etc.
 - Set your hyper-parameters in the corresponding model script

## Testing
 - To test and evaluate results use flag `--mode=test`
 - After testing and evaluation is complete, final results will be printed in the console, and the corresponding files will be saved in the "logs" directory.
 - Set your hyper-parameters in the corresponding model script

## Visualizing
 - To visualize results for a random batch of images use flag `--mode=visualize`
 - Set your hyper-parameters in the corresponding model script

## CRF
 - Running testing will apply CRF by default.
 - If you want to run standalone, run `python denseCRF.py`, after setting your paths.

## BFSCORE
 - Run `python bfscore.py`, after setting your paths.
 - For more details, visit https://github.com/minar09/bfscore_python
