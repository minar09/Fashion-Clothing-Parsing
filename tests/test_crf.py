from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
from skimage.io import imread, imsave
from six.moves import xrange
import BatchDatsetReader as dataset
import datetime
import read_MITSceneParsingData as scene_parsing
import TensorflowUtils as utils
from PIL import Image
import tensorflow as tf
import time
import EvalMetrics as EM
import numpy as np
import scipy.misc as misc
import function_definitions as fd
import pydensecrf.densecrf as dcrf

    
def _read_annotation(filename):

    annotation = np.expand_dims(_transform(filename), axis=3)
    
    return annotation

        
def _transform(filename):
    # 1. read image
    image = misc.imread(filename)

    resize_image = misc.imresize(image, [224, 224], interp='nearest')

    return np.array(resize_image)
    
    
def crf(original_image, annotated_image, NUM_OF_CLASSESS, use_2d=True):
    # Converting annotated image to RGB if it is Gray scale
    print("crf function")

    # Gives no of class labels in the annotated image
    #n_labels = len(set(labels.flat))
    n_labels = NUM_OF_CLASSESS

    # Setting up the CRF model

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    processed_probabilities = annotated_image
    softmax = processed_probabilities.transpose((2, 0, 1))
    #print(softmax.shape)
    U = unary_from_softmax(softmax, scale=None, clip=1e-5)

    U = np.ascontiguousarray(U)
    #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=original_image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(3, 3), schan=(13, 13, 13),
                                      img=original_image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(5)
    #print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    output = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))

    return output

    
input = _transform("1200.jpg")
gt = _read_annotation("1200.png")
output = crf(input, gt, 23)
print(output)
