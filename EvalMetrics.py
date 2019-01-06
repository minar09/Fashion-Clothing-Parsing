import TensorflowUtils as utils
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
    calculate evaluation metrics
"""


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

        check_size(predimage, gtimage)
        
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


"""
    calculate pixel accuracy
"""


def _calc_pixel_accuracy(gtimage, predimage, num_classes):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    pixel_accuracy_ = 0
    per_class_pixel_accuracy = []

    try:
        height, width = gtimage.shape
        pixel_sum = height * width
        class_intersections = []

        check_size(predimage, gtimage)

        for label in range(num_classes):  # 0--> 17
            intersection = 0
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
                        if(gtlabel == label):
                            gt_class = gt_class + 1

            if (gt_class == 0):
                per_class_pixel_accuracy.append(0)
            else:
                per_class_pixel_accuracy.append(intersection / gt_class)

            class_intersections.append(intersection)

        if (pixel_sum == 0):
            pixel_accuracy_ = 0
        else:
            pixel_accuracy_ = (float)(np.sum(class_intersections)) / pixel_sum

    except Exception as err:
        print(err)

    return pixel_accuracy_, np.mean(per_class_pixel_accuracy)


"""
    calculate IoU
"""


def _calcIOU(gtimage, predimage, num_classes):
    IoUs = []
    for i in range(num_classes):
        IoUs.append([0] * num_classes)

    height, width = gtimage.shape

    for label in range(num_classes):  # 0--> 17
        intersection = 0
        union = 0

        for y in range(height):  # 0->223
            # print(crossMat)

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

        if(union == 0):
            IoUs[label] = 0.0
        else:
            # print("label:", label , "intersection:", intersection, " -
            # union:", union)
            IoUs[label] = (float)(intersection) / union

    return IoUs


"""
    calculate confusion matrix
"""


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


"""
    calculate frequency weighted IoU
"""


def _calcFrequencyWeightedIOU(gtimage, predimage, num_classes):
    FrqWIoU = []
    for i in range(num_classes):
        FrqWIoU.append([0] * num_classes)

    gt_pixels = []
    height, width = gtimage.shape

    for label in range(num_classes):  # 0--> 17
        intersection = 0
        union = 0
        pred = 0
        gt = 0

        for y in range(height):  # 0->223
            # print(crossMat)

            for x in range(width):  # =-->223
                gtlabel = gtimage[y, x]
                predlabel = predimage[y, x]

                if predlabel >= num_classes or gtlabel >= num_classes:
                    print('gt:%d, pr:%d' % (gtlabel, predlabel))
                else:
                    if(gtlabel == label and predlabel == label):
                        intersection = intersection + 1
                        gt = gt + 1
                        pred = pred + 1
                    elif(gtlabel == label or predlabel == label):
                        union = union + 1
                        if(gtlabel == label):
                            gt = gt + 1
                        elif(predlabel == label):
                            pred = pred + 1

                gt_pixels.append(gt)

        # union = gt + pred - intersection
        # intersection = gt * pred
        # FrqWIoU[label] = (float)(intersection * gt) / union

        if(union == 0):
            FrqWIoU[label] = 0.0
        else:
            FrqWIoU[label] = (float)(intersection * gt) / union

    #pixel_sum = np.sum(gt_pixels)
    pixel_sum = get_pixel_area(gtimage)
    #pixel_sum = predimage.shape[0] * predimage[1]

    meanFrqWIoU = (float)(np.sum(FrqWIoU)) / pixel_sum

    return FrqWIoU, meanFrqWIoU


#####################################################Useful functions###################################################

"""
   Useful fucntions
"""


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


'''
    Managing average and current values
'''


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
