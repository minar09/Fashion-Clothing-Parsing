
# Hide the warning messages about CPU/GPU
import TensorflowUtils as utils
import glob
from tensorflow.python.platform import gfile
from six.moves import cPickle as pickle
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
   get test and validation file list
   search the directory (*.jpg for input and *.png for annoation), save it into pickle file not to repeat the same serach process
"""


def read_dataset(data_dir):

    # sample record: {'image': f, 'annotation': annotation_file,
    # 'filename': filename}
    training_records = []

    testdir = "E:/Dataset/LIP/train_images/"

    print("## Training dir:", testdir)
    for filename in glob.glob(testdir + '*.jpg'):  # assuming jpg files
        record = {'image': None, 'annotation': None, 'filename': None}
        record['image'] = filename
        record['filename'] = filename
        record['annotation'] = filename.replace(
            "train_images", "train_segmentations").replace(
            "jpg", "png")
        training_records.append(record)

    validation_records = []

    validationdir = "E:/Dataset/LIP/val_images/"

    print("## Validation dir:", validationdir)
    for filename in glob.glob(
            validationdir + '*.jpg'):  # assuming jpg files
        record = {'image': None, 'annotation': None, 'filename': None}
        record['image'] = filename
        record['filename'] = filename
        record['annotation'] = filename.replace(
            "val_images", "val_segmentations").replace(
            "jpg", "png")
        validation_records.append(record)

    return training_records, validation_records
