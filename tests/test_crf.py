from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
from skimage.io import imread, imsave
from six.moves import xrange
import datetime
from PIL import Image
import tensorflow as tf
import time
import numpy as np
import scipy.misc as misc
import pydensecrf.densecrf as dcrf

    
def _read_annotation(filename):

    annotation = np.expand_dims(_transform(filename), axis=3)
    
    return annotation

        
def _transform(filename):
    # 1. read image
    image = misc.imread(filename)

    resize_image = misc.imresize(image, [224, 224], interp='nearest')

    return np.array(resize_image)
    
    
def dense_crf(original_image, annotated_image, NUM_OF_CLASSESS, use_2d=True):
    # Converting annotated image to RGB if it is Gray scale
    print(original_image.shape, annotated_image.shape)

    # Gives no of class labels in the annotated image
    #n_labels = len(set(labels.flat))
    n_labels = NUM_OF_CLASSESS

    # Setting up the CRF model

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    processed_probabilities = annotated_image
    softmax = processed_probabilities.transpose((2, 0, 1))
    print(softmax.shape)
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
    print(Q)
    #print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    output = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))
    print(output.shape)
    
    plt.subplot(240 + 1)
    plt.imshow(output, cmap=plt.get_cmap('nipy_spectral'))
    plt.show()

    return output
    
    
def crf(original_image, annotated_image, NUM_OF_CLASSESS, use_2d=True):

    # Converting annotated image to RGB if it is Gray scale
    # print("crf function")
    print(original_image.shape, annotated_image.shape)

    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,
                                      :,
                                      0] + (annotated_image[:,
                                                            :,
                                                            1] << 8) + (annotated_image[:,
                                                                                        :,
                                                                                        2] << 16)

    # Convert the 32bit integer color to 0, 1, 2, ... labels.
    colors, labels = np.unique(annotated_image, return_inverse=True)

    # Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Gives no of class labels in the annotated image
    n_labels = NUM_OF_CLASSESS
    # print("No of labels in the Image are ")

    # Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(
            original_image.shape[1],
            original_image.shape[0],
            n_labels)

        # get unary potentials (neg log probability)
        processed_probabilities = annotated_image

        softmax = processed_probabilities.transpose((2, 0, 1))

        U = unary_from_softmax(softmax)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations
        # only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(
            sxy=(
                10,
                10),
            srgb=(
                13,
                13,
                13),
            rgbim=original_image,
            compat=10,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(20)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at
    # first.
    MAP = colorize[MAP, :]

    # Get output
    output = MAP.reshape(original_image.shape)
    output = rgb2gray(output)
    
    print(output.shape)
    
    plt.plot(output)
    #plt.imshow(crfoutput, cmap=plt.get_cmap('nipy_spectral'))
    plt.show()
    
    return MAP.reshape(original_image.shape), output

    
#input = _transform("1200.jpg")
input = np.ones((224, 224, 23))
#gt = _read_annotation("1200.png")
gt = np.zeros((224, 224, 23))
output = dense_crf(input, gt, 23)
#output = crf(input, gt, 23)
