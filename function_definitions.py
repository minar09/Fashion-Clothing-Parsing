from __future__ import print_function

import matplotlib.pyplot as plt
from six.moves import xrange
import datetime

import numpy as np
import tensorflow as tf
import time
import EvalMetrics
import inference
import TensorflowUtils as Utils

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# color map for LIP
LIP_colors = [(0, 0, 0)                 # 0=Background
              # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
              # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
              # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
              , (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0), (0, 0, 85), (0, 119, 221), (85, 85, 0), (0, 85, 85), (85, 51, 0), (52, 86, 128), (0, 128, 0), (0, 0, 255), (51, 170, 221), (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)]
cmap_name = 'lip_cmap'
n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
# lip_cm = LinearSegmentedColormap.from_list(cmap_name, LIP_colors, N=n_bin)

"""
   Optimization functions
"""


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',


        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels,
            # out_channels]
            kernels = Utils.get_variable(np.transpose(
                kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = Utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = Utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            # if FLAGS.debug:
            # util.add_activation_summary(current)
        elif kind == 'pool':
            current = Utils.avg_pool_2x2(current)
        net[name] = current
        # added for resume better
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter

    return net


def mode_visualize(sess, FLAGS, TEST_DIR, validation_dataset_reader, pred_annotation, image, annotation, keep_probability, NUM_OF_CLASSES):
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    valid_images, valid_annotations = validation_dataset_reader.get_random_batch(
        FLAGS.batch_size)
    pred = sess.run(pred_annotation,
                    feed_dict={image: valid_images, annotation: valid_annotations,
                               keep_probability: 1.0})

    # pixel_acc_op, pixel_acc_update_op = tf.metrics.accuracy(labels=annotation, predictions=pred_annotation)
    # mean_iou_op, mean_iou_update_op = tf.metrics.mean_iou(labels=annotation, predictions=pred_annotation, num_classes=NUM_OF_CLASSES)

    # sess.run(tf.local_variables_initializer())
    # feed_dict={image: valid_images, annotation: valid_annotations, keep_probability: 1.0}
    # sess.run(
    # [pixel_acc_update_op, mean_iou_update_op],
    # feed_dict=feed_dict)
    # pixel_acc, tf_miou = sess.run(
    # [pixel_acc_op, mean_iou_op],
    # feed_dict=feed_dict)

    valid_annotations = np.squeeze(valid_annotations, axis=3)
    pred = np.squeeze(pred, axis=3)

    crossMats = list()
    crf_crossMats = list()

    for itr in range(FLAGS.batch_size):
        Utils.save_image(valid_images[itr].astype(
            np.uint8), TEST_DIR, name="inp_" + str(itr))
        Utils.save_image(valid_annotations[itr].astype(
            np.uint8) * 255 / NUM_OF_CLASSES, TEST_DIR, name="gt_" + str(itr))
        Utils.save_image(pred[itr].astype(
            np.uint8) * 255 / NUM_OF_CLASSES, TEST_DIR, name="pred_" + str(itr))
        print("Saved image: %d" % itr)

        # Eval metrics for this image prediction
        cm = EvalMetrics._calcCrossMat(
            valid_annotations[itr].astype(
                np.uint8), pred[itr].astype(
                np.uint8), NUM_OF_CLASSES)
        crossMats.append(cm)

        """ Generate CRF """
        crfimage, crfoutput = inference.crf(TEST_DIR + "inp_" + str(itr) + ".png", TEST_DIR + "pred_" + str(
            itr) + ".png", TEST_DIR + "crf_" + str(itr) + ".png", NUM_OF_CLASSES, use_2d=True)

        # Eval metrics for this image prediction with crf
        crf_cm = EvalMetrics._calcCrossMat(
            valid_annotations[itr].astype(
                np.uint8), crfoutput.astype(
                np.uint8), NUM_OF_CLASSES)
        crf_crossMats.append(crf_cm)

    print(">>> Prediction results:")
    total_cm = np.sum(crossMats, axis=0)
    EvalMetrics.show_result(total_cm, NUM_OF_CLASSES)

    print("\n")
    print(">>> Prediction results (CRF):")
    crf_total_cm = np.sum(crf_crossMats, axis=0)
    EvalMetrics.show_result(crf_total_cm, NUM_OF_CLASSES)


def mode_train(sess, FLAGS, net, train_dataset_reader, validation_dataset_reader, train_records, pred_annotation, image, annotation, keep_probability, logits, train_op, loss, summary_op, summary_writer, saver, display_step=300):
    print(">>>>>>>>>>>>>>>>Train mode")
    start = time.time()

    # Start decoder training

    valid = list()
    step = list()
    lo = list()

    global_step = sess.run(net['global_step'])
    global_step = 0
    max_iteration = round(
        (train_dataset_reader.get_num_of_records() //
         FLAGS.batch_size) *
        FLAGS.training_epochs)
    display_step = round(
        train_dataset_reader.get_num_of_records() // FLAGS.batch_size)
    print(
        "No. of maximum steps:",
        max_iteration,
        " Training epochs:",
        FLAGS.training_epochs)

    for itr in xrange(global_step, max_iteration):
        # 6.1 load train and GT images
        train_images, train_annotations = train_dataset_reader.next_batch(
            FLAGS.batch_size)

        feed_dict = {
            image: train_images,
            annotation: train_annotations,
            keep_probability: 0.85}

        # 6.2 training
        sess.run(train_op, feed_dict=feed_dict)

        if itr % 10 == 0:
            train_loss, summary_str = sess.run(
                [loss, summary_op], feed_dict=feed_dict)
            print("Step: %d, Train_loss:%g" % (itr, train_loss))
            summary_writer.add_summary(summary_str, itr)
            if itr % display_step == 0 and itr != 0:
                lo.append(train_loss)

        if itr % display_step == 0 and itr != 0:
            valid_images, valid_annotations = validation_dataset_reader.next_batch(
                FLAGS.batch_size)
            valid_loss = sess.run(
                loss,
                feed_dict={
                    image: valid_images,
                    annotation: valid_annotations,
                    keep_probability: 1.0})
            print(
                "%s ---> Validation_loss: %g" %
                (datetime.datetime.now(), valid_loss))
            global_step = sess.run(net['global_step'])
            saver.save(
                sess,
                FLAGS.logs_dir +
                "model.ckpt",
                global_step=global_step)

            valid.append(valid_loss)
            step.append(itr)
            # print("valid", valid, "step", step)

            try:
                plt.clf()
                plt.ylim(0, 1)
                plt.plot(np.array(step), np.array(lo))
                plt.title('Training Loss')
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.savefig(FLAGS.logs_dir + "training_loss.jpg")
            except Exception as err:
                print(err)

            try:
                plt.clf()
                plt.ylim(0, 1)
                plt.plot(np.array(step), np.array(valid))
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.title('Validation Loss')
                plt.savefig(FLAGS.logs_dir + "validation_loss.jpg")
            except Exception as err:
                print(err)

            try:
                plt.clf()
                plt.ylim(0, 1)
                plt.plot(np.array(step), np.array(lo))
                plt.plot(np.array(step), np.array(valid))
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.title('Result')
                plt.legend(['Training Loss', 'Validation Loss'],
                           loc='upper right')
                plt.savefig(FLAGS.logs_dir + "merged_loss.jpg")
            except Exception as err:
                print(err)

    try:
        np.savetxt(
            FLAGS.logs_dir +
            "training_steps.csv",
            np.c_[step, lo, valid],
            fmt='%4f',
            delimiter=',')
    except Exception as err:
        print(err)

    end = time.time()
    print("Learning time:", end - start, "seconds")


def mode_test(sess, FLAGS, TEST_DIR, validation_dataset_reader, valid_records, pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSES):
    print(">>>>>>>>>>>>>>>>Test mode")
    start = time.time()

    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    validation_dataset_reader.reset_batch_offset(0)

    crossMats = list()
    crf_crossMats = list()

    for itr1 in range(validation_dataset_reader.get_num_of_records() // FLAGS.batch_size):

        valid_images, valid_annotations = validation_dataset_reader.next_batch(
            FLAGS.batch_size)
        pred, logits1 = sess.run([pred_annotation, logits],
                                 feed_dict={image: valid_images, annotation: valid_annotations,
                                            keep_probability: 1.0})

        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred)
        print("logits shape:", logits1.shape)
        np.set_printoptions(threshold=np.inf)

        for itr2 in range(FLAGS.batch_size):

            fig = plt.figure()
            pos = 240 + 1
            plt.subplot(pos)
            plt.imshow(valid_images[itr2].astype(np.uint8))
            plt.axis('off')
            plt.title('Original')

            pos = 240 + 2
            plt.subplot(pos)
            plt.imshow(
                valid_annotations[itr2].astype(
                    np.uint8),
                cmap=plt.get_cmap('nipy_spectral'))
            plt.axis('off')
            plt.title('GT')

            pos = 240 + 3
            plt.subplot(pos)
            plt.imshow(
                pred[itr2].astype(
                    np.uint8),
                cmap=plt.get_cmap('nipy_spectral'))
            plt.axis('off')
            plt.title('Prediction')

            # Confusion matrix for this image prediction
            crossMat = EvalMetrics._calcCrossMat(
                valid_annotations[itr2].astype(
                    np.uint8), pred[itr2].astype(
                    np.uint8), NUM_OF_CLASSES)
            crossMats.append(crossMat)

            np.savetxt(TEST_DIR +
                       "Crossmatrix" +
                       str(itr1 *
                           FLAGS.batch_size +
                           itr2) +
                       ".csv", crossMat, fmt='%4i', delimiter=',')

            # Save input, gt, pred, crf_pred, sum figures for this image

            # ---------------------------------------------
            Utils.save_image(valid_images[itr2].astype(np.uint8), TEST_DIR,
                             name="inp_" + str(itr1 * FLAGS.batch_size + itr2))
            Utils.save_image(valid_annotations[itr2].astype(np.uint8), TEST_DIR,
                             name="gt_" + str(itr1 * FLAGS.batch_size + itr2))
            Utils.save_image(pred[itr2].astype(np.uint8),
                             TEST_DIR,
                             name="pred_" + str(itr1 * FLAGS.batch_size + itr2))

            # --------------------------------------------------
            """ Generate CRF """
            crfimage, crfoutput = inference.crf(TEST_DIR + "inp_" + str(itr1 * FLAGS.batch_size + itr2) + ".png", TEST_DIR + "pred_" + str(
                itr1 * FLAGS.batch_size + itr2) + ".png", TEST_DIR + "crf_" + str(itr1 * FLAGS.batch_size + itr2) + ".png", NUM_OF_CLASSES, use_2d=True)

            # Confusion matrix for this image prediction with crf
            crf_crossMat = EvalMetrics._calcCrossMat(
                valid_annotations[itr2].astype(
                    np.uint8), crfoutput.astype(
                    np.uint8), NUM_OF_CLASSES)
            crf_crossMats.append(crf_crossMat)

            np.savetxt(TEST_DIR +
                       "crf_Crossmatrix" +
                       str(itr1 *
                           FLAGS.batch_size +
                           itr2) +
                       ".csv", crf_crossMat, fmt='%4i', delimiter=',')

            pos = 240 + 4
            plt.subplot(pos)
            plt.imshow(crfoutput.astype(np.uint8),
                       cmap=plt.get_cmap('nipy_spectral'))
            plt.axis('off')
            plt.title('Prediction + CRF')

            plt.savefig(TEST_DIR + "resultSum_" +
                        str(itr1 * FLAGS.batch_size + itr2))

            plt.close('all')
            print("Saved image: %d" % (itr1 * FLAGS.batch_size + itr2))

    try:
        total_cm = np.sum(crossMats, axis=0)
        np.savetxt(
            FLAGS.logs_dir +
            "Crossmatrix.csv",
            total_cm,
            fmt='%4i',
            delimiter=',')

        print(">>> Prediction results:")
        EvalMetrics.show_result(total_cm, NUM_OF_CLASSES)

        # Prediction with CRF
        crf_total_cm = np.sum(crf_crossMats, axis=0)
        np.savetxt(
            FLAGS.logs_dir +
            "CRF_Crossmatrix.csv",
            crf_total_cm,
            fmt='%4i',
            delimiter=',')

        print("\n")
        print(">>> Prediction results (CRF):")
        EvalMetrics.show_result(crf_total_cm, NUM_OF_CLASSES)

    except Exception as err:
        print(err)

    end = time.time()
    print("Testing time:", end - start, "seconds")
