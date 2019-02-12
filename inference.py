"""
Adapted from the inference.py to demonstate the usage of the util functions.
"""

import numpy as np
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax


# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
    # TODO: Use scipy instead.

# fn_im = "inp.png"
# fn_anno = "pred.png"
# fn_output = "output.png"


def crf(fn_im, fn_anno, fn_output, NUM_OF_CLASSESS, use_2d=True):
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
    #print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
    #print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
    #colors = colors[1:]
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
    n_labels = NUM_OF_CLASSESS

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
        #print("Using generic 2D functions")

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
    imwrite(fn_output, MAP.reshape(img.shape))
    crfimage = MAP.reshape(img.shape)
    # print(crfimage.shape)

    # Just randomly manually run inference iterations
    # Q, tmp1, tmp2 = d.startInference()
    # for i in range(5):
    # print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
    # d.stepInference(Q, tmp1, tmp2)

    return crfimage, crfoutput


"""
   Function which returns the labelled image after applying CRF
"""
# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D functions will be applied
# else Generic functions will be applied


def dense_crf(original_image, annotated_image, NUM_OF_CLASSES, use_2d=True):
    # Converting annotated image to RGB if it is Gray scale
    #print("crf function")
    #annotated_image = tf.expand_dims(annotated_image, dim=0)
    #print(original_image.shape, annotated_image.shape)

    # Gives no of class labels in the annotated image
    #n_labels = len(set(labels.flat))
    n_labels = NUM_OF_CLASSES

    # Setting up the CRF model

    d = dcrf.DenseCRF2D(
        original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    processed_probabilities = annotated_image
    softmax = processed_probabilities.transpose((2, 0, 1))

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

    #print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    output = np.argmax(Q, axis=0).reshape(
        (original_image.shape[0], original_image.shape[1]))
    # print(output.shape)

    return output


#####################################################Optimization functions###################################################
