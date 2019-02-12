import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import EvalMetrics as EM


def main():
    val_pred_paths, val_label_paths = init_path()
    val_hist = compute_hist(val_pred_paths, val_label_paths)
    show_result(val_hist)


def init_path():

    val_output_dir = 'logs/UNet_CFPD/pred/'
    val_label_dir = 'logs/UNet_CFPD/gt/'

    val_pred_file_names = os.listdir(val_output_dir)
    val_label_file_names = os.listdir(val_label_dir)

    val_gt_paths = []
    val_pred_paths = []

    for file_name in tqdm(val_pred_file_names):
        val_pred_paths.append(os.path.join(val_output_dir, file_name))

    for file_name in tqdm(val_label_file_names):
        val_gt_paths.append(os.path.join(val_label_dir, file_name))

    return val_pred_paths, val_gt_paths


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(images, labels):
    n_cl = 23
    hist = np.zeros((n_cl, n_cl))
    crossMats = list()

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
        cm = EM._calcCrossMat(label_array, image_array, n_cl)
        crossMats.append(cm)

    T_CM = np.sum(crossMats, axis=0)
    np.savetxt("logs/UNet_CFPD/our_totalCM.csv",
               T_CM, fmt='%4i', delimiter=',')
    print(EM._calc_eval_metrics_from_confusion_matrix(T_CM, n_cl))

    return hist


def show_result(hist):
    np.savetxt("logs/UNet_CFPD/JPP_totalCM.csv",
               hist, fmt='%4i', delimiter=',')

    classes = ['bk', 'T-shirt', 'bag', 'belt', 'blazer', 'blouse', 'coat', 'dress', 'face', 'hair',
               'hat', 'jeans', 'legging', 'pants', 'scarf', 'shoe', 'shorts', 'skin', 'skirt',
               'socks', 'stocking', 'sunglass', 'sweater']
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
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    print('>>>', 'mean accuracy', np.nanmean(acc))
    print('-' * 50)

    # @evaluation 3: mean IU & per-class IU
    print('IoU for each class:')
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IoU', np.nanmean(iu))
    print('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print('>>>', 'Freq Weighted IoU', (freq[freq > 0] * iu[freq > 0]).sum())
    print('=' * 50)


if __name__ == '__main__':
    main()
