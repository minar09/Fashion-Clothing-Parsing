"""
Adapted from the inference.py to demonstate the usage of the util functions.
"""

import sys
import os
import numpy as np
import pydensecrf.densecrf as dcrf
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
import scipy.misc as misc
from PIL import Image
from tqdm import tqdm

# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave

# fn_im = "inp.png"
# fn_anno = "pred.png"
# fn_output = "output.png"

OUTPUT_DIR = './output_crf/deeplabv2_10k/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

n_classes = 18

# colour map
label_colours = [(0, 0, 0),  # 0=Background
                 (128, 0, 0),  # hat
                 (255, 0, 0),  # hair
                 (170, 0, 51),  # sunglasses
                 (255, 85, 0),  # upper-clothes
                 (0, 128, 0),  # skirt
                 (0, 85, 85),  # pants
                 (0, 0, 85),  # dress
                 (0, 85, 0),  # belt
                 (255, 255, 0),  # Left-shoe
                 (255, 170, 0),  # Right-shoe
                 (0, 0, 255),  # face
                 (85, 255, 170),  # left-leg
                 (170, 255, 85),  # right-leg
                 (51, 170, 221),  # left-arm
                 (0, 255, 255),  # right-arm
                 (85, 51, 0),  # bag
                 (52, 86, 128)  # scarf
                 ]

# image mean
IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)


def decode_labels(mask, num_classes=n_classes):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.

    Returns:
      A batch with num_images RGB images of the same size as the input.
      :param mask:
      :param num_classes:
    """

    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask[:, :, 0]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
    outputs = np.array(img)
    return outputs


def init_path():

    val_anno_dir = 'E:/Dataset/LIP/output/parsing/val/'
    val_id_list = 'E:/Dataset/LIP/list/val_id.txt'
    val_img_dir = 'E:/Dataset/LIP/validation/images/'

    val_img_paths = []
    val_anno_paths = []
    val_img_id = []

    f = open(val_id_list, 'r')
    for line in f:
        val = line.strip("\n")
        val_img_paths.append(val_img_dir + val + '.jpg')
        val_anno_paths.append(val_anno_dir + val + '.png')
        val_img_id.append(val)

    return val_img_paths, val_anno_paths, val_img_id


def crf(fn_im, fn_anno, fn_output, NUM_OF_CLASSES=n_classes, use_2d=True):
    ##################################
    ### Read images and annotation ###
    ##################################
    img = imread(fn_im)
    # print(fn_anno.shape)

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_rgb = imread(fn_anno).astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + \
        (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)
    # labels = np.unique(fn_anno)
    #print(colors, labels)

    # But remove the all-0 black, that won't exist in the MAP!
    # HAS_UNK = 0 in colors
    HAS_UNK = False
    # if HAS_UNK:
    # print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
    # print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
    # colors = colors[1:]
    # else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    #n_labels = len(set(labels.flat)) - int(HAS_UNK)
    #print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))
    n_labels = NUM_OF_CLASSES

    ###########################
    ### Setup the CRF model ###
    ###########################
    #use_2d = False
    #use_2d = True
    if use_2d:
        #print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(
            labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)

        # get unary potentials (neg log probability)
        # processed_probabilities = fn_anno
        # softmax = processed_probabilities.transpose((2, 0, 1))
        # print(softmax.shape)
        # U = unary_from_softmax(softmax, scale=None, clip=None)
        # U = np.ascontiguousarray(U)

        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        # print("Using generic 2D functions")

        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(
            labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    # print(MAP.shape)
    crfoutput = MAP.reshape((img.shape[0], img.shape[1]))
    # print(crfoutput.shape)
    # print(np.unique(crfoutput))

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]
    # print(MAP.shape)
    # imwrite(fn_output+".png", MAP.reshape(img.shape))
    crfimage = MAP.reshape(img.shape)
    # print(crfimage.shape)

    msk = decode_labels(crfimage, num_classes=NUM_OF_CLASSES)
    parsing_im = Image.fromarray(msk)
    parsing_im.save(fn_output+'_vis.png')
    cv2.imwrite(fn_output+'.png', crfimage[:, :, 0])

    # Just randomly manually run inference iterations
    # Q, tmp1, tmp2 = d.startInference()
    # for i in range(5):
    # print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
    # d.stepInference(Q, tmp1, tmp2)

    # return crfimage, crfoutput


# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D fucntions will be applied
# else Generic functions will be applied

def crf_with_probs(original_input_image, predicted_probabilities, num_label, num_iter=5, use_2d=True):

    # Setting up the CRF model
    np.set_printoptions(threshold=10)
    predicted_probabilities = predicted_probabilities.transpose((2, 0, 1))
    # print("probs:", probs)
    # print("probs shape:", probs.shape)

    if use_2d:
        d = dcrf.DenseCRF2D(original_input_image.shape[1], original_input_image.shape[0], num_label)

        # get unary potentials (neg log probability)
        U = unary_from_softmax(predicted_probabilities)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=original_input_image,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(num_iter)
    # print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    output = MAP.reshape((original_input_image.shape[1], original_input_image.shape[0]))  # orig.shape[)
    return output


def crf_with_labels(original_input_image, predicted_segmentation, num_label, num_iter=5, use_2d=True):

    # Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(original_input_image.shape[1], original_input_image.shape[0], num_label)

        # get unary potentials (neg log probability)
        U = unary_from_labels(predicted_segmentation, num_label,
                              gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=original_input_image,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(num_iter)
    MAP = np.argmax(Q, axis=0)
    # original_image.shape[)
    output = MAP.reshape((original_input_image.shape[1], original_input_image.shape[0]))

    return output


if __name__ == "__main__":

    val_img_paths, val_anno_paths, val_img_id = init_path()
    for img_path, anno_path, img_id in tqdm(zip(val_img_paths, val_anno_paths, val_img_id)):
        crf(img_path, anno_path, OUTPUT_DIR+img_id)
