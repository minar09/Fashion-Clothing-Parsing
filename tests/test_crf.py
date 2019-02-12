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
import inference


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

    d = dcrf.DenseCRF2D(
        original_image.shape[1], original_image.shape[0], n_labels)

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
    feats = create_pairwise_gaussian(
        sdims=(3, 3), shape=original_image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=original_image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(5)
    print(Q)
    #print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    output = np.argmax(Q, axis=0).reshape(
        (original_image.shape[0], original_image.shape[1]))
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


"""
Function which returns the labelled image after applying CRF

"""

# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D fucntions will be applied
# else Generic functions will be applied


def image_crf(original_image, annotated_image, output_image, use_2d=True):

    # Converting annotated image to RGB if it is Gray scale
    if(len(annotated_image.shape) < 3):
        annotated_image = gray2rgb(annotated_image)

    imsave("testing2.png", annotated_image)

    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:, :, 0] + (
        annotated_image[:, :, 1] << 8) + (annotated_image[:, :, 2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    # Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat))

    print("No of labels in the Image are ")
    print(n_labels)

    # Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(
            original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]
    imsave(output_image, MAP.reshape(original_image.shape))
    return MAP.reshape(original_image.shape)


def final_crf(original_image, annotated_image, NUM_OF_CLASSESS, use_2d=True):
    # Converting annotated image to RGB if it is Gray scale
    print(original_image.shape, annotated_image.shape)

    # Gives no of class labels in the annotated image
    #n_labels = len(set(labels.flat))
    n_labels = NUM_OF_CLASSESS

    # Setting up the CRF model

    d = dcrf.DenseCRF2D(
        original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    processed_probabilities = annotated_image
    softmax = processed_probabilities.transpose((2, 0, 1))
    print(softmax.shape)
    U = unary_from_softmax(softmax, scale=None, clip=1e-5)

    U = np.ascontiguousarray(U)
    #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations
    # only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(
        sxy=(
            80,
            80),
        srgb=(
            13,
            13,
            13),
        rgbim=original_image,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(5)
    print(Q)
    #print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    output = np.argmax(Q, axis=0).reshape(
        (original_image.shape[0], original_image.shape[1]))
    print(output.shape)

    plt.subplot(240 + 1)
    plt.imshow(output, cmap=plt.get_cmap('nipy_spectral'))
    plt.show()

    return output


def _calcCrossMat(gtimage, predimage, num_classes):
    crossMat = []

    for i in range(num_classes):
        crossMat.append([0] * num_classes)
    # print(crossMat)
    height, width = gtimage.shape

    for y in range(height):
        # print(crossMat)

        for x in range(width):
            gtlabel = gtimage[y, x]
            predlabel = predimage[y, x]
            if predlabel >= num_classes or gtlabel >= num_classes:
                print('gt:%d, pr:%d' % (gtlabel, predlabel))
            else:
                crossMat[gtlabel][predlabel] = crossMat[gtlabel][predlabel] + 1

    return crossMat


def _calc_eval_metrics(gtimage, predimage, num_classes):

    pixel_accuracy_ = 0
    mean_accuracy = 0
    meanFrqWIoU = 0
    meanIoU = 0

    per_class_pixel_accuracy = []
    IoUs = []
    FrqWIoU = []

    for i in range(num_classes):
        IoUs.append([0] * num_classes)
        FrqWIoU.append([0] * num_classes)
        per_class_pixel_accuracy.append([0] * num_classes)

    try:
        height, width = gtimage.shape
        pixel_sum = height * width

        class_intersections = []
        gt_pixels = []

        #check_size(predimage, gtimage)

        # Check classes
        # gt_labels, gt_labels_count = extract_classes(gtimage)
        # print(gt_labels)
        # pred_labels, pred_labels_count = extract_classes(predimage)
        # print(pred_labels)
        # assert num_classes == gt_labels_count
        # print(num_classes, gt_labels_count, pred_labels_count)
        # assert gt_labels_count == pred_labels_count

        for label in range(num_classes):  # 0--> 17
            intersection = 0
            union = 0
            gt_class = 0

            for y in range(height):  # 0->223

                for x in range(width):  # =-->223
                    gtlabel = gtimage[y, x]
                    predlabel = predimage[y, x]

                    if predlabel >= num_classes or gtlabel >= num_classes:
                        print('gt:%d, pr:%d' % (gtlabel, predlabel))
                    else:
                        if(gtlabel == label and predlabel == label):
                            intersection = intersection + 1
                        if(gtlabel == label or predlabel == label):
                            union = union + 1
                        if(gtlabel == label):
                            gt_class = gt_class + 1

            # Calculate per class pixel accuracy
            if (gt_class == 0):
                per_class_pixel_accuracy[label] = 0
            else:
                per_class_pixel_accuracy[label] = (
                    float)(intersection / gt_class)

            # Calculate per class IoU and FWIoU
            if(union == 0):
                IoUs[label] = 0.0
                FrqWIoU[label] = 0.0
            else:
                IoUs[label] = (float)(intersection) / union
                FrqWIoU[label] = (float)(intersection * gt_class) / union

            class_intersections.append(intersection)
            gt_pixels.append(gt_class)

        # Check pixels
        # assert pixel_sum == get_pixel_area(gtimage)
        # assert pixel_sum == np.sum(gt_pixels)
        # print(pixel_sum, get_pixel_area(gtimage), np.sum(gt_pixels))

        # Calculate mean accuracy and meanIoU
        mean_accuracy = np.mean(per_class_pixel_accuracy)
        meanIoU = np.mean(IoUs)

        # hist = _calcCrossMat(gtimage, predimage, num_classes)
        # num_cor_pix = np.diag(hist)
        # # num of correct pixels
        # num_cor_pix = np.diag(hist)
        # # num of gt pixels
        # num_gt_pix = np.sum(hist, axis=1)
        # # num of pred pixels
        # num_pred_pix = np.sum(hist, axis=0)
        # # IU
        # denominator = (num_gt_pix + num_pred_pix - num_cor_pix)
        # print(np.sum(class_intersections), np.sum(num_cor_pix))

        # Calculate pixel accuracy and mean FWIoU
        if (pixel_sum == 0):
            pixel_accuracy_ = 0
            meanFrqWIoU = 0
        else:
            pixel_accuracy_ = (float)(np.sum(class_intersections)) / pixel_sum
            meanFrqWIoU = (float)(np.sum(FrqWIoU)) / pixel_sum

    except Exception as err:
        print(err)

    return pixel_accuracy_, mean_accuracy, meanIoU, meanFrqWIoU


# input = _transform("in.png")
# input = np.ones((224, 224, 23))
# anno = _read_annotation("anno.png")
# anno = np.zeros((224, 224, 23))
# output = dense_crf(input, anno, 23)
# output = crf(input, anno, 23)
_, crfoutput = inference.crf(
    "inp.png", "pred.png", "output.png", 23, use_2d=True)
#crfoutput = misc.imread("output.png")
#crfoutput = misc.imresize(crfoutput, [224, 224])
print(crfoutput.shape)
#crfoutput = np.argmax(crfoutput, axis=2)
# print(np.array([crfoutput]).astype(np.uint8))
#crfoutput = cv2.normalize(crfoutput, None, 0, 255, cv2.NORM_MINMAX)
print(np.unique(crfoutput))
gtimage = misc.imread("pred.png")
#gtimage = cv2.normalize(gtimage, None, 0, 255, cv2.NORM_MINMAX)
print(gtimage.shape)
print(np.unique(gtimage))
# crossmat = _calcCrossMat(gtimage, crfoutput.astype(np.uint8), 23)
# print(crossmat)
print(_calc_eval_metrics(gtimage.astype(np.uint8), crfoutput.astype(np.uint8), 23))
