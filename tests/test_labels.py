import numpy as np
import scipy.misc as misc


def _labelize(filename):
    # 1. read image
    image = misc.imread(filename)
    # 3. resize it
    resize_image = misc.imresize(
        image, [224, 224], interp='nearest')

    resized_image = np.array(resize_image)
    image_label = make_one_hot(resized_image)

    return image_label


def make_one_hot(image):
    label = [0] * 23
    classes = np.unique(image)

    for each in range(23):
        if each in classes:
            label[each] = 1
        else:
            label[each] = 0

    return label


print(_labelize("E:/Dataset/CFPD/testimages/1.png"))
