from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
from skimage.io import imread, imsave
import pydensecrf.densecrf as dcrf
from six.moves import xrange
import BatchDatsetReader as dataset
import datetime
import read_MITSceneParsingData as scene_parsing
import read_CFPD_data as fashion_parsing
import TensorflowUtils as utils
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import function_definitions as fd

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_integer(
    "training_epochs",
    "60",
    "number of epochs for training")
tf.flags.DEFINE_string("logs_dir", "logs/UNet_CRF/", "path to logs directory")
#tf.flags.DEFINE_string("data_dir", "E:/Dataset/Dataset10k/", "path to dataset")
tf.flags.DEFINE_string("data_dir", "E:/Dataset/CFPD/", "path to dataset")

tf.flags.DEFINE_float(
    "learning_rate",
    "1e-8",
    "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1001)
# NUM_OF_CLASSESS = 18  # human parsing  59 #cloth   151  # MIT Scene
NUM_OF_CLASSESS = 23  # total parsing  23 #cloth main   13  # CFPD
IMAGE_SIZE = 224
DISPLAY_STEP = 300
TEST_DIR = FLAGS.logs_dir + "Image/"
VIS_DIR = FLAGS.logs_dir + "VIS_Image/"


"""
  UNET
"""


def unetinference(image, keep_prob):
    net = {}
    l2_reg = FLAGS.learning_rate
    # added for resume better
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter
    with tf.variable_scope("inference"):
        inputs = image
        teacher = tf.placeholder(
            tf.float32, [
                None, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSESS])
        is_training = True

        # 1, 1, 3
        conv1_1 = utils.conv(
            inputs,
            filters=64,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        conv1_2 = utils.conv(
            conv1_1,
            filters=64,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        pool1 = utils.pool(conv1_2)

        # 1/2, 1/2, 64
        conv2_1 = utils.conv(
            pool1,
            filters=128,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        conv2_2 = utils.conv(
            conv2_1,
            filters=128,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        pool2 = utils.pool(conv2_2)

        # 1/4, 1/4, 128
        conv3_1 = utils.conv(
            pool2,
            filters=256,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        conv3_2 = utils.conv(
            conv3_1,
            filters=256,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        pool3 = utils.pool(conv3_2)

        # 1/8, 1/8, 256
        conv4_1 = utils.conv(
            pool3,
            filters=512,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        conv4_2 = utils.conv(
            conv4_1,
            filters=512,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        pool4 = utils.pool(conv4_2)

        # 1/16, 1/16, 512
        conv5_1 = utils.conv(pool4, filters=1024, l2_reg_scale=l2_reg)
        conv5_2 = utils.conv(conv5_1, filters=1024, l2_reg_scale=l2_reg)
        concated1 = tf.concat([utils.conv_transpose(
            conv5_2, filters=512, l2_reg_scale=l2_reg), conv4_2], axis=3)

        conv_up1_1 = utils.conv(concated1, filters=512, l2_reg_scale=l2_reg)
        conv_up1_2 = utils.conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg)
        concated2 = tf.concat([utils.conv_transpose(
            conv_up1_2, filters=256, l2_reg_scale=l2_reg), conv3_2], axis=3)

        conv_up2_1 = utils.conv(concated2, filters=256, l2_reg_scale=l2_reg)
        conv_up2_2 = utils.conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg)
        concated3 = tf.concat([utils.conv_transpose(
            conv_up2_2, filters=128, l2_reg_scale=l2_reg), conv2_2], axis=3)

        conv_up3_1 = utils.conv(concated3, filters=128, l2_reg_scale=l2_reg)
        conv_up3_2 = utils.conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg)
        concated4 = tf.concat([utils.conv_transpose(
            conv_up3_2, filters=64, l2_reg_scale=l2_reg), conv1_2], axis=3)

        conv_up4_1 = utils.conv(concated4, filters=64, l2_reg_scale=l2_reg)
        conv_up4_2 = utils.conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg)
        outputs = utils.conv(
            conv_up4_2, filters=NUM_OF_CLASSESS, kernel_size=[
                1, 1], activation=None)
        annotation_pred = tf.argmax(outputs, dimension=3, name="prediction")

        return tf.expand_dims(annotation_pred, dim=3), outputs, net
        # return Model(inputs, outputs, teacher, is_training)


"""inference
  optimize with trainable paramters (Check which ones)
  loss_val : loss operator (mean(

"""


def train(loss_val, var_list, global_step):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step=global_step)


def main(argv=None):
    # 1. input placeholders
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(
        tf.float32,
        shape=(
            None,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3),
        name="input_image")
    annotation = tf.placeholder(
        tf.int32,
        shape=(
            None,
            IMAGE_SIZE,
            IMAGE_SIZE,
            1),
        name="annotation")
    # global_step = tf.Variable(0, trainable=False, name='global_step')

    # 2. construct inference network
    pred_annotation, logits, net = unetinference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=3)
    tf.summary.image(
        "ground_truth",
        tf.cast(
            annotation,
            tf.uint8),
        max_outputs=3)

    tf.summary.image(
        "pred_annotation",
        tf.cast(
            pred_annotation,
            tf.uint8),
        max_outputs=3)

    # 3. loss measure
    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.squeeze(
                annotation,
                squeeze_dims=[3]),
            name="entropy")))
    tf.summary.scalar("entropy", loss)

    # 4. optimizing
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)

    train_op = train(loss, trainable_var, net['global_step'])

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader from ", FLAGS.data_dir, "...")
    #train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    train_records, valid_records, test_records = fashion_parsing.read_dataset(
        FLAGS.data_dir)
    print("data dir:", FLAGS.data_dir)
    print("train_records length :", len(train_records))
    print("valid_records length :", len(valid_records))
    print("test_records length :", len(test_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(
            train_records, image_options)
        validation_dataset_reader = dataset.BatchDatset(
            valid_records, image_options)
        test_dataset_reader = dataset.BatchDatset(
            test_records, image_options)
    if FLAGS.mode == 'visualize':
        validation_dataset_reader = dataset.BatchDatset(
            valid_records, image_options)
    if FLAGS.mode == 'test':
        test_dataset_reader = dataset.BatchDatset(
            test_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    # 5. paramter setup
    # 5.1 init params
    sess.run(tf.global_variables_initializer())
    # 5.2 restore params if possible
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # 6. train-mode
    if FLAGS.mode == "train":

        fd.mode_train(sess, FLAGS, net, train_dataset_reader, validation_dataset_reader, train_records, pred_annotation,
                      image, annotation, keep_probability, logits, train_op, loss, summary_op, summary_writer, saver, DISPLAY_STEP)

        fd.mode_test(sess, FLAGS, TEST_DIR, test_dataset_reader, test_records,
                     pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSESS)

    # test-random-validation-data mode
    elif FLAGS.mode == "visualize":

        fd.mode_visualize(sess, FLAGS, VIS_DIR, validation_dataset_reader,
                          pred_annotation, image, annotation, keep_probability, NUM_OF_CLASSESS)

    # test-full-validation-dataset mode
    elif FLAGS.mode == "test":  # heejune added

        fd.mode_test(sess, FLAGS, TEST_DIR, test_dataset_reader, test_records,
                     pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSESS)

    sess.close()


if __name__ == "__main__":
    tf.app.run()
