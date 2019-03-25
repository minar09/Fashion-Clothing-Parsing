
from PIL import Image
import numpy as np
from tqdm import tqdm

# Hide the warning messages about CPU/GPU, invalid error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.seterr(divide='ignore', invalid='ignore')


"""LIP functions"""


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(images, labels, n_cl=18):
    hist = np.zeros((n_cl, n_cl))

    for img_path, label_path in tqdm(zip(images, labels)):
        label = Image.open(label_path)
        label_array = np.array(label, dtype=np.int32)
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.int32)

        gtsz = label_array.shape
        imgsz = image_array.shape

        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.int32)

        hist += fast_hist(label_array, image_array, n_cl)

    return hist


def show_result(hist, n_cl=18):

    # Dressup 10K, 18 classes
    classes = ['background', 'hat', 'hair', 'sunglasses', 'upperclothes', 'skirt', 'pants', 'dress',
               'belt', 'leftShoe', 'rightShoe', 'face', 'leftLeg', 'rightLeg', 'leftArm', 'rightArm', 'bag', 'scarf']

    # CFPD, 23 classes
    if n_cl == 23:
        classes = ['bk', 'T-shirt', 'bag', 'belt', 'blazer', 'blouse', 'coat', 'dress', 'face', 'hair',
                   'hat', 'jeans', 'legging', 'pants', 'scarf', 'shoe', 'shorts', 'skin', 'skirt',
                   'socks', 'stocking', 'sunglass', 'sweater']

    # LIP, 20 classes
    if n_cl == 20:
        classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
                   'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
                   'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
                   'rightShoe']

    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print('=' * 50)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall/pixel accuracy', acc)
    print('-' * 50)

    # @evaluation 2: mean accuracy & per-class accuracy
    print('Accuracy for each class (pixel accuracy):')
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    print('>>>', 'mean accuracy', np.nanmean(acc))
    print('-' * 50)

    # @evaluation 3: mean IU & per-class IU
    print('IoU for each class:')
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IoU', np.nanmean(iu))
    print('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print('>>>', 'Freq Weighted IoU', (freq[freq > 0] * iu[freq > 0]).sum())
    print('=' * 50)


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

        # Calculate mean accuracy and meanIoU
        mean_accuracy = np.mean(per_class_pixel_accuracy)
        meanIoU = np.mean(IoUs)

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
    calculate eval metrics from confusion matrix
"""


def calculate_eval_metrics_from_cross_matrix(gtimage, predimage, num_classes):
    cross_mat = calculate_confusion_matrix(gtimage, predimage, num_classes)
    # cross_mat = fast_hist(gtimage, predimage, num_classes)

    class_intersections = np.diag(cross_mat)
    total_intersections = np.nansum(class_intersections)

    gt_pixels = np.sum(cross_mat, axis=1)
    pred_pixels = np.sum(cross_mat, axis=0)
    total_pixels = np.sum(cross_mat)

    # mean accuracy, mean IoU, frequency-weighted IoU
    class_accuracies = []
    IoUs = []
    FWIoUs = []

    for i in range(len(class_intersections)):
        try:
            acc = float(class_intersections[i] / gt_pixels[i])
            class_accuracies.append(acc)
        except:
            class_accuracies.append(0)

        try:
            iu = float(
                class_intersections[i] / (gt_pixels[i] + pred_pixels[i] - class_intersections[i]))
            IoUs.append(iu)
        except:
            IoUs.append(0)

        try:
            fwiu = float((class_intersections[i] * gt_pixels[i]) / (
                gt_pixels[i] + pred_pixels[i] - class_intersections[i]))
            FWIoUs.append(fwiu)
        except:
            FWIoUs.append(0)

    pixel_accuracy_ = float(total_intersections /
                            total_pixels)    # pixel accuracy
    mean_accuracy = float(np.nansum(class_accuracies) /
                          num_classes)    # mean accuracy
    meanIoU = float(np.nansum(IoUs) / num_classes)    # mean IoU
    # Total frequency-weighted IoU
    meanFrqWIoU = float(np.nansum(FWIoUs) / total_pixels)

    return pixel_accuracy_, mean_accuracy, meanIoU, meanFrqWIoU, cross_mat


def calculate_eval_metrics_from_confusion_matrix(cross_mat, num_classes):

    class_intersections = np.diag(cross_mat)
    total_intersections = np.nansum(class_intersections)

    gt_pixels = np.sum(cross_mat, axis=1)
    pred_pixels = np.sum(cross_mat, axis=0)
    total_pixels = np.sum(cross_mat)

    # mean accuracy, mean IoU, frequency-weighted IoU
    class_accuracies = []
    IoUs = []
    FWIoUs = []

    for i in range(len(class_intersections)):
        try:
            acc = float(class_intersections[i] / gt_pixels[i])
            class_accuracies.append(acc)
        except:
            class_accuracies.append(0)

        try:
            iu = float(
                class_intersections[i] / (gt_pixels[i] + pred_pixels[i] - class_intersections[i]))
            IoUs.append(iu)

            fwiu = float((gt_pixels[i] * iu) / total_pixels)
            FWIoUs.append(fwiu)
        except:
            IoUs.append(0)
            FWIoUs.append(0)

    pixel_accuracy_ = float(total_intersections /
                            total_pixels)    # pixel accuracy
    mean_accuracy = float(np.nansum(class_accuracies) /
                          num_classes)    # mean accuracy
    meanIoU = float(np.nansum(IoUs) / num_classes)    # mean IoU
    meanFrqWIoU = np.nansum(FWIoUs)    # Total frequency-weighted IoU

    return pixel_accuracy_, mean_accuracy, meanIoU, meanFrqWIoU, cross_mat


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


def calcuate_accuracy(mat, bprint=True):

    n = len(mat)

    acc = 0
    tot = 0
    for i in range(1, n):
        acc += mat[i][i]
        tot += sum(mat[i])
    acc1 = acc/tot

    if bprint:
        print("Acc-fg: %5.3f (%d/%d)" % (acc/tot, acc, tot))
    acc += mat[0][0]
    tot += sum(mat[0])
    if bprint:
        print("Acc-all: %5.3f (%d/%d)" % (acc/tot, acc, tot))

    return acc1, acc/tot  # fg accuracy, total accuracy


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

        if union == 0:
            IoUs[label] = 0.0
        else:
            # print("label:", label , "intersection:", intersection, " -
            # union:", union)
            IoUs[label] = float(intersection) / union

    return IoUs


"""
    calculate confusion matrix
"""


def calculate_confusion_matrix(gt_image, predicted_image, num_classes):
    cross_mat = []

    for i in range(num_classes):
        cross_mat.append([0] * num_classes)
    # print(cross_mat)
    height, width = gt_image.shape

    for y in range(height):
        # print(cross_mat)

        for x in range(width):
            gtlabel = gt_image[y, x]
            predlabel = predicted_image[y, x]
            if predlabel >= num_classes or gtlabel >= num_classes:
                print('gt:%d, pr:%d' % (gtlabel, predlabel))
            else:
                cross_mat[gtlabel][predlabel] = cross_mat[gtlabel][predlabel] + 1

    return cross_mat


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
