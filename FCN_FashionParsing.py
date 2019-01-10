from __future__ import print_function
import time
import function_definitions as fd
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

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_integer(
    "training_epochs",
    "60",
    "number of epochs for training")
tf.flags.DEFINE_string("logs_dir", "logs/FCN_FP/", "path to logs directory")
#tf.flags.DEFINE_string("data_dir", "E:/Dataset/Dataset10k/", "path to dataset")
tf.flags.DEFINE_string("data_dir", "E:/Dataset/CFPD/", "path to dataset")
tf.flags.DEFINE_float(
    "learning_rate",
    "1e-4",
    "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
#tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")
#tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1001)
# NUM_OF_CLASSES = 18  # human parsing  59 #cloth   151  # MIT Scene
NUM_OF_CLASSES = 23  # total parsing  23 #cloth main   13  # CFPD
IMAGE_SIZE = 224
DISPLAY_STEP = 300
TEST_DIR = FLAGS.logs_dir + "Image/"
VIS_DIR = FLAGS.logs_dir + "VIS_Image/"


"""
   Train, Test
"""


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    # 1. donwload VGG pretrained model from network if not did before
    #    model_data is dictionary for variables from matlab mat file
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    # 2. construct model graph
    with tf.variable_scope("inference"):
        # 2.1 VGG
        image_net = fd.vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        # Pooling layer for Conv5 layer
        pool5 = utils.max_pool_2x2(conv_final_layer)

        # FC6 to Conv
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        # FC7 to Conv
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        """ Start outfit encoder 

        # FC1 @ 256
        attr_W1 = utils.weight_variable([1, 1, 4096, 256], name="attr_W1")
        attr_b1 = utils.bias_variable([256], name="attr_b1")
        attr_conv1 = utils.conv2d_basic(relu_dropout7, attr_W1, attr_b1)
        attr_relu1 = tf.nn.relu(attr_conv1, name="attr_relu1")
        attr_relu_dropout1 = tf.nn.dropout(attr_relu1, keep_prob=keep_prob)

        # FC2 @ NUM_OF_CLASSES
        attr_W2 = utils.weight_variable([1, 1, 256, NUM_OF_CLASSES], name="attr_W2")
        attr_b2 = utils.bias_variable([NUM_OF_CLASSES], name="attr_b2")
        attr_conv2 = utils.conv2d_basic(attr_relu_dropout1, attr_W2, attr_b2)
        print(attr_conv2)

        # Sigmoid
        sig_fc2 = tf.nn.sigmoid(attr_conv2)

         End outfit encoder """

        # FC8 to Conv
        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSES], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSES], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # print(conv8.shape) # (?, 7, 7, 23)

        """ outfit encoder """
        fc1_flat = tf.contrib.layers.flatten(conv8)  # 1127
        # print(fc1_flat.shape) # 1127

        # FC2 @ NUM_OF_CLASSES
        fc2w = tf.Variable(tf.truncated_normal([1127, NUM_OF_CLASSES],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='w2')
        fc2b = tf.Variable(tf.constant(1.0, shape=[NUM_OF_CLASSES], dtype=tf.float32),
                           trainable=True, name='b2')
        fcl2 = tf.nn.bias_add(tf.matmul(fc1_flat, fc2w), fc2b)
        # print(fcl2.shape)

        # Sigmoid
        sig_fc2 = tf.nn.sigmoid(fcl2)

        # FC2 @ NUM_OF_CLASSES
        fc3w = tf.Variable(tf.truncated_normal([NUM_OF_CLASSES, 1127],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='w3')
        fc3b = tf.Variable(tf.constant(1.0, shape=[1127], dtype=tf.float32),
                           trainable=True, name='b3')
        fcl3 = tf.nn.bias_add(tf.matmul(sig_fc2, fc3w), fc3b)
        # print(fcl3.shape)

        fcl3_conv = tf.reshape(fcl3, tf.shape(conv8))
        # print(fcl3_conv.shape)

        # product
        conv8 = tf.add(fcl3_conv, conv8, name="product")
        # print(conv8.shape)

        """ outfit encoder """

        """ now to upscale to actual image size """

        # Upscale1 + Skip/Fusion1
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable(
            [4, 4, deconv_shape1[3].value, NUM_OF_CLASSES], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(
            conv8, W_t1, b_t1, output_shape=tf.shape(
                image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        # print(fuse_1.shape) # (?, 14, 14, 512)

        # Upscale2 + Skip/Fusion2
        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable(
            [4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(
            fuse_1, W_t2, b_t2, output_shape=tf.shape(
                image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        """ Start Product with outfit encoder 

        # Deconv1
        fc3w = tf.Variable(tf.truncated_normal([NUM_OF_CLASSES, 256],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='w3')
        fc3b = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32),
                           trainable=True, name='b3')
        fcl3 = tf.nn.bias_add(tf.matmul(sig_fc2, fc3w), fc3b)
        relu3 = tf.nn.relu(fcl3)
        
        #
        fc4w = tf.Variable(tf.truncated_normal([256, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='w4')
        fc4b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='b4')
        fcl4 = tf.nn.bias_add(tf.matmul(relu3, fc4w), fc4b)
        relu4 = tf.nn.relu(fcl4)

        # Deconv2
        deconv_shape2 = image_net["pool3"].get_shape()
        W_de2 = utils.weight_variable(
            [4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_de2")
        b_de2 = utils.bias_variable([deconv_shape2[3].value], name="b_de2")
        deconv_sig2 = utils.conv2d_transpose_strided(
            relu4, W_de2, b_de2, output_shape=tf.shape(
                image_net["pool3"]))

        # Product (Gi = gi · Fi)
        #print(deconv_sig2.shape, fuse_2.shape)
        product_deconv = tf.add(deconv_sig2, fuse_2, name="product")

         End Product with outfit encoder """

        # Upscale3 to output
        shape = tf.shape(image)
        deconv_shape3 = tf.stack(
            [shape[0], shape[1], shape[2], NUM_OF_CLASSES])
        W_t3 = utils.weight_variable(
            [16, 16, NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSES], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(
            fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        # conv_t3 = utils.conv2d_transpose_strided(
        # product_deconv, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        # prob = tf.nn.softmax(conv_t3, axis =3)
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), sig_fc2, conv_t3, image_net


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
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
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
    label = tf.placeholder(
        tf.float32,
        shape=(
            None,
            NUM_OF_CLASSES),
        name="label")
    #global_step = tf.Variable(0, trainable=False, name='global_step')

    # 2. construct inference network
    pred_annotation, logits_encoder, logits_decoder, net = inference(
        image, keep_probability)
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
    encoder_loss = tf.reduce_mean(
        (tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_encoder,
            labels=label,
            name="encoder_entropy")))

    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_decoder,
            labels=tf.squeeze(
                annotation,
                squeeze_dims=[3]),
            name="entropy")))
    tf.summary.scalar("entropy", loss)

    # 4. optimizing
    train_encoder_op = tf.train.AdamOptimizer(
        FLAGS.learning_rate).minimize(encoder_loss)

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
        train_encoder_dataset_reader = dataset.BatchDatset(
            train_records, image_options, NUM_OF_CLASSES)
        validation_encoder_dataset_reader = dataset.BatchDatset(
            valid_records, image_options, NUM_OF_CLASSES)

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

        fd.mode_train_encoder(sess, FLAGS, net, train_records, pred_annotation, image, keep_probability, saver, encoder_loss,
                              train_encoder_op, label, train_encoder_dataset_reader, validation_encoder_dataset_reader, DISPLAY_STEP)

        fd.mode_train(sess, FLAGS, net, train_dataset_reader, validation_dataset_reader, train_records, pred_annotation, image,
                      annotation, keep_probability, logits_decoder, train_op, loss, summary_op, summary_writer, saver, DISPLAY_STEP)

        fd.mode_test(sess, FLAGS, TEST_DIR, test_dataset_reader, test_records, pred_annotation,
                     image, annotation, keep_probability, logits_decoder, NUM_OF_CLASSES)

    # test-random-validation-data mode
    elif FLAGS.mode == "visualize":

        fd.mode_visualize(sess, FLAGS, VIS_DIR, validation_dataset_reader,
                          pred_annotation, image, annotation, keep_probability, NUM_OF_CLASSES)

    # test-full-validation-dataset mode
    elif FLAGS.mode == "test":  # heejune added

        fd.mode_test(sess, FLAGS, TEST_DIR, test_dataset_reader, test_records,
                     pred_annotation, image, annotation, keep_probability, logits_decoder, NUM_OF_CLASSES)

    sess.close()


if __name__ == "__main__":
    tf.app.run()
