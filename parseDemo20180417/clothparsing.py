# cloth parsing after segmentation

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


def load_images(origimagefile, gtimagefile,  segimagefile, crfsegimagefile):

    origimage = cv2.imread(origimagefile)
    gtimage = cv2.imread(gtimagefile, cv2.IMREAD_UNCHANGED)
    segimage = cv2.imread(segimagefile, cv2.IMREAD_UNCHANGED)
    crfsegimage = cv2.imread(crfsegimagefile, cv2.IMREAD_UNCHANGED)
    return origimage, gtimage, segimage, crfsegimage


# display
#
#  original   coarse-pred    fine-pred
#  gr-truth   error          error
#

def display_fullimages(origimage, gtimage, segimage, crfsegimage):

    fig = plt.figure()
    pos = 230 + 1
    plt.subplot(pos)
    plt.imshow(cv2.cvtColor(origimage, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('original')
    pos = 230 + 4
    plt.subplot(pos)
    plt.imshow(gtimage, cmap=plt.get_cmap('nipy_spectral'))
    plt.axis('off')
    plt.title('ground-truth')
    plt.subplot(232)
    plt.imshow(segimage, cmap=plt.get_cmap('nipy_spectral'))
    plt.axis('off')
    plt.title('1st pred')
    plt.subplot(235)
    ret, errorImage = cv2.threshold(cv2.absdiff(
        segimage, gtimage), 0.5, 255, cv2.THRESH_BINARY)
    plt.imshow(errorImage, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title('1st error')
    plt.subplot(233)
    plt.imshow(crfsegimage, cmap=plt.get_cmap('nipy_spectral'))
    plt.axis('off')
    plt.title('2nd pred')
    plt.subplot(236)
    ret, errorImage = cv2.threshold(cv2.absdiff(
        segimage, gtimage), 0.5, 255, cv2.THRESH_BINARY)
    plt.imshow(errorImage, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title('2nd error')
    fig.show()  # donot use plt.show()
    return fig


def display_parts(partImages, titleList):

    num = len(partImages)
    fig = plt.figure(figsize=(4*num, 4))
    plt.suptitle('Cloth Parsing (coarse-paring only): number is the proportion of all area or foreground\n an improvement is ready to come soon. \n((c) 2018 Seoultech H. Ahn')

    for i in range(num):
        pos = 100 + num*10 + (i+1)
        plt.subplot(pos)
        plt.imshow(cv2.cvtColor(partImages[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(titleList[i])

    fig.show()  # donot use plt.show()
    return fig


labels = ["background",  # 0
          "hat",  # 1
          "hair",  # 2
          "sunglass",  # 3
          "upper-clothes",  # 4
          "skirt",  # 5
          "pants",  # 6
          "dress",  # 7
          "belt",  # 8
          "left-shoe",  # 9
          "right-shoe",  # 10
          "face"  # 11
          "left-leg"  # 12
          "right-leg",  # 13
          "left-arm",  # 14
          "right-arm",  # 15
          "bag",  # 16
          "scarf"  # 17
          ]
maxn = 100
for n in range(maxn):

    origimagefile = "inp_" + str(n) + ".png"
    gtimagefile = "gt_" + str(n) + ".png"
    segimagefile = "pred_" + str(n) + ".png"
    crfsegimagefile = "crf_" + str(n) + ".png"

    origimage, gtimage, segimage, crfsegimage = load_images(
        origimagefile, gtimagefile,  segimagefile, crfsegimagefile)

    height, width, channels = origimage.shape
    #print("size=", height, width, channels)
    #height, width = gtimage.shape
    #print("size=", height, width)
    #height, width = segimage.shape
    #print("size=", height, width)

    if True:
        fig = display_fullimages(origimage, gtimage, segimage, crfsegimage)
        input()
        plt.close(fig)

    hist_pred = cv2.calcHist([segimage], [0], None, [256], [0, 256])
    hist_gt = cv2.calcHist([gtimage], [0], None, [256], [0, 256])
    hist_crf = cv2.calcHist([crfsegimage], [0], None, [256], [0, 256])
    print(">>>>>>>>>>Image", hist_crf)
    num_fg_pixels = 0
    print(">>>>>> Image %d  <<<<<<<<<<<<<<<<<<<<" % (n))
    print("%10.10s \t pro \t gt \t pred" % ('label'))
    for label in range(1, len(labels)):
        num_fg_pixels = num_fg_pixels + hist_pred[label][0]
    for label in range(1, len(labels)):
        print('%10.10s \t %3.2f \t %d \t %d \t %d' % (
            labels[label], hist_pred[label][0]/num_fg_pixels, hist_gt[label][0], hist_pred[label][0], hist_crf[label][0]))

    # reassign small labels into close label

    # extract the masked parts
    # now only a few important things.
    #    "upper-clothes", 4
    #    "skirt",    5
    #    "pants",    6
    #    "dress",    7

    imageList = []
    titleList = []

    plt.subplot(151)
    allImage = origimage.copy()  # [:,:,:]
    allImage[segimage == 0] = 0
    imageList.append(allImage)
    titleList.append("all:" + str(num_fg_pixels*100//(width*height)))

    upperMask = segimage.copy()
    upperMask[segimage != 4] = 0
    upperMask[segimage == 4] = 1
    upperImage = origimage.copy()  # [:,:,:]
    upperImage[upperMask != 1] = 0
    imageList.append(upperImage)
    titleList.append("upper:" + str(hist_pred[4][0]*100//num_fg_pixels))

    # skirt
    skirtMask = segimage.copy()
    skirtMask[segimage != 5] = 0
    skirtMask[segimage == 5] = 1
    skirtImage = origimage.copy()
    skirtImage[skirtMask != 1] = 0
    imageList.append(skirtImage)
    titleList.append("skirt:" + str(hist_pred[5][0]*100//num_fg_pixels))

    pantsMask = segimage.copy()
    pantsMask[segimage != 6] = 0
    pantsMask[segimage == 6] = 1
    pantsImage = origimage.copy()
    pantsImage[pantsMask != 1] = 0
    imageList.append(pantsImage)
    titleList.append("pants:" + str(hist_pred[6][0]*100//num_fg_pixels))

    dressMask = segimage.copy()
    dressMask[segimage != 7] = 0
    dressMask[segimage == 7] = 1
    dressImage = origimage.copy()
    dressImage[dressMask != 1] = 0
    imageList.append(dressImage)
    titleList.append("dress:" + str(hist_pred[7][0]*100//num_fg_pixels))

    fig2 = display_parts(imageList, titleList)

    # input()
    time.sleep(3)
    plt.close(fig2)
