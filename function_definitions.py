from __future__ import print_function

import matplotlib.pyplot as plt
from six.moves import xrange
import datetime

import numpy as np
import tensorflow as tf
import time
import EvalMetrics
import denseCRF
import TensorflowUtils as Utils
from matplotlib.colors import ListedColormap, BoundaryNorm

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
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

label_colors = ['black',  # "background", #     0
                'sienna',  # "hat", #            1
                'gray',  # "hair", #           2
                'navy',  # "sunglass", #       3
                'red',  # "upper-clothes", #  4
                'gold',  # "skirt",  #          5
                'blue',  # "pants",  #          6
                'seagreen',  # "dress", #          7
                'darkorchid',  # "belt", #           8
                'firebrick',  # "left-shoe", #      9
                'darksalmon',  # "right-shoe", #     10
                'moccasin',  # "face",  #           11
                'darkgreen',  # "left-leg", #       12
                'royalblue',  # "right-leg", #      13
                'chartreuse',  # "left-arm",#       14
                'paleturquoise',  # "right-arm", #      15
                'darkcyan',  # "bag", #            16
                'deepskyblue'  # "scarf" #          17
                ]
clothnorm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,
                          7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], 18)


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
        cm = EvalMetrics.calculate_confusion_matrix(
            valid_annotations[itr].astype(
                np.uint8), pred[itr].astype(
                np.uint8), NUM_OF_CLASSES)
        crossMats.append(cm)

        """ Generate CRF """
        crfimage, crfoutput = denseCRF.crf(TEST_DIR + "inp_" + str(itr) + ".png", TEST_DIR + "pred_" + str(
            itr) + ".png", TEST_DIR + "crf_" + str(itr) + ".png", NUM_OF_CLASSES, use_2d=True)

        # Eval metrics for this image prediction with crf
        crf_cm = EvalMetrics.calculate_confusion_matrix(
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
            crossMat = EvalMetrics.calculate_confusion_matrix(
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
            crfimage, crfoutput = denseCRF.crf(TEST_DIR + "inp_" + str(itr1 * FLAGS.batch_size + itr2) + ".png", TEST_DIR + "pred_" + str(
                itr1 * FLAGS.batch_size + itr2) + ".png", TEST_DIR + "crf_" + str(itr1 * FLAGS.batch_size + itr2) + ".png", NUM_OF_CLASSES, use_2d=True)

            # Confusion matrix for this image prediction with crf
            crf_crossMat = EvalMetrics.calculate_confusion_matrix(
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


def mode_crftest(sess, FLAGS, TEST_DIR, validation_dataset_reader, valid_records, pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSES):

    accuracies = np.zeros(
        (validation_dataset_reader.get_num_of_records(), 3, 2))
    nFailed = 0
    validation_dataset_reader.reset_batch_offset(0)
    probability = tf.nn.softmax(logits=logits, axis=3)  # the axis!

    for itr1 in range(validation_dataset_reader.get_num_of_records() // FLAGS.batch_size):
        valid_images, valid_annotations = validation_dataset_reader.next_batch(
            FLAGS.batch_size)

        predprob, pred = sess.run([probability, pred_annotation], feed_dict={image: valid_images, annotation: valid_annotations,
                                                                             keep_probability: 1.0})
        np.set_printoptions(threshold=10)
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred)
        predprob = np.squeeze(predprob)

        # @TODO: convert np once not repeatedly
        for itr2 in range(FLAGS.batch_size):

            # 1. run CRF
            crfwithlabeloutput = denseCRF.crf_with_labels(valid_images[itr2].astype(
                np.uint8), pred[itr2].astype(np.uint8), NUM_OF_CLASSES)
            crfwithprobsoutput = denseCRF.crf_with_probs(
                valid_images[itr2].astype(np.uint8), predprob[itr2], NUM_OF_CLASSES)

            original = valid_images[itr2].astype(np.uint8)
            groundtruth = valid_annotations[itr2].astype(np.uint8)
            fcnpred = pred[itr2].astype(np.uint8)
            crfwithlabelpred = crfwithlabeloutput.astype(np.uint8)
            crfwithprobspred = crfwithprobsoutput.astype(np.uint8)

            # 2. Calculate confusion matrix between gtimage and prediction image and store to file
            pred_confusion_matrix = EvalMetrics.calculate_confusion_matrix(
                groundtruth, fcnpred, NUM_OF_CLASSES)
            crfwithlabelpred_confusion_matrix = EvalMetrics.calculate_confusion_matrix(
                groundtruth, crfwithlabelpred, NUM_OF_CLASSES)
            crfwithprobspred_confusion_matrix = EvalMetrics.calculate_confusion_matrix(
                groundtruth, crfwithprobspred, NUM_OF_CLASSES)

            accuracies[itr1*FLAGS.batch_size +
                       itr2][0] = EvalMetrics.calcuate_accuracy(pred_confusion_matrix, False)
            accuracies[itr1*FLAGS.batch_size + itr2][1] = EvalMetrics.calcuate_accuracy(
                crfwithlabelpred_confusion_matrix, False)
            accuracies[itr1*FLAGS.batch_size + itr2][2] = EvalMetrics.calcuate_accuracy(
                crfwithprobspred_confusion_matrix, True)

            T_full = 0.9
            T_fgnd = 0.85
            if accuracies[itr1 * FLAGS.batch_size + itr2][2][1] < T_full or accuracies[itr1 * FLAGS.batch_size + itr2][2][0] < T_fgnd:
                nFailed += 1
                print("Failed Image (%d-th): %d" %
                      (nFailed, itr1*FLAGS.batch_size + itr2))

            # 4. saving result
            # now we have 0-index image
            filenum = str(itr1 * FLAGS.batch_size + itr2)

            Utils.save_image(original, FLAGS.logs_dir,
                             name="in_" + filenum)
            Utils.save_image(
                groundtruth, TEST_DIR, name="gt_" + filenum)
            Utils.save_image(crfwithprobspred,
                             TEST_DIR, name="crf_" + filenum)

            # ---End calculate cross matrix
            print("Saved image: %s" % filenum)

    np.save(FLAGS.logs_dir + "accuracy", accuracies)


def mode_predonly(sess, FLAGS, TEST_DIR, validation_dataset_reader, valid_records, pred_annotation, image, annotation, keep_probability, logits, NUM_OF_CLASSES):

    nFailed = 0
    validation_dataset_reader.reset_batch_offset(0)
    probability = tf.nn.softmax(logits=logits, axis=3)  # the axis!

    for itr1 in range(validation_dataset_reader.get_num_of_records() // FLAGS.batch_size):
        valid_images, _ = validation_dataset_reader.next_batch(
            FLAGS.batch_size)

        predprob, pred = sess.run([probability, pred_annotation], feed_dict={
                                  image: valid_images, keep_probability: 1.0})

        np.set_printoptions(threshold=10)

        pred = np.squeeze(pred)
        predprob = np.squeeze(predprob)

        # @TODO: convert np once not repeatedly
        for itr2 in range(FLAGS.batch_size):

            # 1. run CRF
            crfwithlabeloutput = denseCRF.crf_with_labels(valid_images[itr2].astype(
                np.uint8), pred[itr2].astype(np.uint8), NUM_OF_CLASSES)
            crfwithprobsoutput = denseCRF.crf_with_probs(
                valid_images[itr2].astype(np.uint8), predprob[itr2], NUM_OF_CLASSES)

            # 2. show result display
            orignal = valid_images[itr2].astype(np.uint8)
            fcnpred = pred[itr2].astype(np.uint8)
            crfwithlabelpred = crfwithlabeloutput.astype(np.uint8)
            crfwithprobspred = crfwithprobsoutput.astype(np.uint8)

            # 4. saving result
            # now we have 0-index image
            filenum = str(itr1 * FLAGS.batch_size + itr2)

            # Utils.save_image(orignal, TEST_DIR, name="in_" + filenum)
            Utils.save_image(crfwithprobspred, TEST_DIR,
                             name="probcrf_" + filenum)
            Utils.save_image(crfwithlabelpred, TEST_DIR,
                             name="labelcrf_" + filenum)

            # ---End calculate cross matrix
            print("Saved image: %s" % filenum)


def mode_full_test(sess, flags, save_dir, validation_dataset_reader, valid_records, pred_annotation, image, annotation, keep_probability, logits, num_classes):
    print(">>>>>>>>>>>>>>>>Test mode")
    start = time.time()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    validation_dataset_reader.reset_batch_offset(0)
    probability = tf.nn.softmax(logits=logits, axis=3)

    crossMats = list()
    label_crf_crossMats = list()
    prob_crf_crossMats = list()

    for itr1 in range(validation_dataset_reader.get_num_of_records() // flags.batch_size):

        valid_images, valid_annotations = validation_dataset_reader.next_batch(
            flags.batch_size)

        predprob, pred = sess.run([probability, pred_annotation], feed_dict={
            image: valid_images, keep_probability: 1.0})

        np.set_printoptions(threshold=10)

        pred = np.squeeze(pred)
        predprob = np.squeeze(predprob)
        valid_annotations = np.squeeze(valid_annotations, axis=3)

        for itr2 in range(flags.batch_size):

            fig = plt.figure()
            pos = 240 + 1
            plt.subplot(pos)
            plt.imshow(valid_images[itr2].astype(np.uint8))
            plt.axis('off')
            plt.title('Original')

            pos = 240 + 2
            plt.subplot(pos)
            # plt.imshow(valid_annotations[itr2].astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(valid_annotations[itr2].astype(
                np.uint8), cmap=ListedColormap(label_colors), norm=clothnorm)
            plt.axis('off')
            plt.title('GT')

            pos = 240 + 3
            plt.subplot(pos)
            # plt.imshow(pred[itr2].astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(pred[itr2].astype(np.uint8),
                       cmap=ListedColormap(label_colors), norm=clothnorm)
            plt.axis('off')
            plt.title('Prediction')

            # Confusion matrix for this image prediction
            crossMat = EvalMetrics.calculate_confusion_matrix(
                valid_annotations[itr2].astype(
                    np.uint8), pred[itr2].astype(
                    np.uint8), num_classes)
            crossMats.append(crossMat)

            np.savetxt(save_dir +
                       "Crossmatrix" +
                       str(itr1 *
                           flags.batch_size +
                           itr2) +
                       ".csv", crossMat, fmt='%4i', delimiter=',')

            # Save input, gt, pred, crf_pred, sum figures for this image

            """ Generate CRF """
            # 1. run CRF
            crfwithlabeloutput = denseCRF.crf_with_labels(valid_images[itr2].astype(
                np.uint8), pred[itr2].astype(np.uint8), num_classes)
            crfwithprobsoutput = denseCRF.crf_with_probs(
                valid_images[itr2].astype(np.uint8), predprob[itr2], num_classes)

            # 2. show result display
            crfwithlabelpred = crfwithlabeloutput.astype(np.uint8)
            crfwithprobspred = crfwithprobsoutput.astype(np.uint8)

            # -----------------------Save inp and masks----------------------
            Utils.save_image(valid_images[itr2].astype(np.uint8), save_dir,
                             name="inp_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(valid_annotations[itr2].astype(np.uint8), save_dir,
                             name="gt_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(pred[itr2].astype(np.uint8),
                             save_dir,
                             name="pred_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(crfwithprobspred, save_dir,
                             name="probcrf_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(crfwithlabelpred, save_dir,
                             name="labelcrf_" + str(itr1 * flags.batch_size + itr2))

            # ----------------------Save visualized masks---------------------
            Utils.save_visualized_image(valid_annotations[itr2].astype(np.uint8), save_dir,
                                        image_name="gt_" + str(itr1 * flags.batch_size + itr2), n_classes=num_classes)
            Utils.save_visualized_image(pred[itr2].astype(np.uint8),
                                        save_dir,
                                        image_name="pred_" + str(itr1 * flags.batch_size + itr2), n_classes=num_classes)
            Utils.save_visualized_image(crfwithprobspred, save_dir, image_name="probcrf_" + str(
                itr1 * flags.batch_size + itr2), n_classes=num_classes)
            Utils.save_visualized_image(crfwithlabelpred, save_dir, image_name="labelcrf_" + str(
                itr1 * flags.batch_size + itr2), n_classes=num_classes)

            # --------------------------------------------------

            # Confusion matrix for this image prediction with crf
            prob_crf_crossMat = EvalMetrics.calculate_confusion_matrix(
                valid_annotations[itr2].astype(
                    np.uint8), crfwithprobsoutput.astype(
                    np.uint8), num_classes)
            prob_crf_crossMats.append(prob_crf_crossMat)

            label_crf_crossMat = EvalMetrics.calculate_confusion_matrix(
                valid_annotations[itr2].astype(
                    np.uint8), crfwithlabeloutput.astype(
                    np.uint8), num_classes)
            label_crf_crossMats.append(label_crf_crossMat)

            np.savetxt(save_dir +
                       "prob_crf_Crossmatrix" +
                       str(itr1 *
                           flags.batch_size +
                           itr2) +
                       ".csv", prob_crf_crossMat, fmt='%4i', delimiter=',')

            np.savetxt(save_dir +
                       "label_crf_Crossmatrix" +
                       str(itr1 *
                           flags.batch_size +
                           itr2) +
                       ".csv", label_crf_crossMat, fmt='%4i', delimiter=',')

            pos = 240 + 4
            plt.subplot(pos)
            # plt.imshow(crfwithprobsoutput.astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(crfwithprobsoutput.astype(np.uint8),
                       cmap=ListedColormap(label_colors), norm=clothnorm)
            plt.axis('off')
            plt.title('Prediction + CRF (prob)')

            pos = 240 + 5
            plt.subplot(pos)
            # plt.imshow(crfwithlabeloutput.astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(crfwithlabeloutput.astype(np.uint8),
                       cmap=ListedColormap(label_colors), norm=clothnorm)
            plt.axis('off')
            plt.title('Prediction + CRF (label)')

            plt.savefig(save_dir + "resultSum_" +
                        str(itr1 * flags.batch_size + itr2))

            plt.close('all')
            print("Saved image: %d" % (itr1 * flags.batch_size + itr2))

    try:
        total_cm = np.sum(crossMats, axis=0)
        np.savetxt(
            flags.logs_dir +
            "Crossmatrix.csv",
            total_cm,
            fmt='%4i',
            delimiter=',')

        print(">>> Prediction results:")
        EvalMetrics.show_result(total_cm, num_classes)

        # Prediction with CRF
        prob_crf_total_cm = np.sum(prob_crf_crossMats, axis=0)
        np.savetxt(
            flags.logs_dir +
            "prob_CRF_Crossmatrix.csv",
            prob_crf_total_cm,
            fmt='%4i',
            delimiter=',')

        label_crf_total_cm = np.sum(label_crf_crossMats, axis=0)
        np.savetxt(
            flags.logs_dir +
            "label_CRF_Crossmatrix.csv",
            label_crf_total_cm,
            fmt='%4i',
            delimiter=',')

        print("\n")
        print(">>> Prediction results (CRF (prob)):")
        EvalMetrics.show_result(prob_crf_total_cm, num_classes)

        print("\n")
        print(">>> Prediction results (CRF (label)):")
        EvalMetrics.show_result(label_crf_total_cm, num_classes)

    except Exception as err:
        print(err)

    end = time.time()
    print("Testing time:", end - start, "seconds")


def mode_new_test(sess, flags, save_dir, validation_dataset_reader, valid_records, pred_annotation, image, annotation, keep_probability, logits, num_classes):
    print(">>>>>>>>>>>>>>>>Test mode")
    start = time.time()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    validation_dataset_reader.reset_batch_offset(0)
    probability = tf.nn.softmax(logits=logits, axis=3)

    cross_mats = list()
    crf_cross_mats = list()

    # tf_pixel_acc_list = []
    # tf_miou_list = []

    # pixel_acc_op, pixel_acc_update_op = tf.metrics.accuracy(labels=annotation, predictions=pred_annotation)
    # mean_iou_op, mean_iou_update_op = tf.metrics.mean_iou(labels=annotation, predictions=pred_annotation, num_classes=num_classes)

    for itr1 in range(validation_dataset_reader.get_num_of_records() // flags.batch_size):

        valid_images, valid_annotations = validation_dataset_reader.next_batch(
            flags.batch_size)

        predprob, pred = sess.run([probability, pred_annotation], feed_dict={image: valid_images, keep_probability: 1.0})

        # tf measures
        sess.run(tf.local_variables_initializer())
        feed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 1.0}
        # predprob, pred, _, __ = sess.run([probability, pred_annotation, pixel_acc_update_op, mean_iou_update_op], feed_dict=feed_dict)
        # tf_pixel_acc, tf_miou = sess.run([pixel_acc_op, mean_iou_op], feed_dict=feed_dict)
        # tf_pixel_acc_list.append(tf_pixel_acc)
        # tf_miou_list.append(tf_miou)

        np.set_printoptions(threshold=10)

        pred = np.squeeze(pred)
        predprob = np.squeeze(predprob)
        valid_annotations = np.squeeze(valid_annotations, axis=3)

        for itr2 in range(flags.batch_size):

            fig = plt.figure()
            pos = 240 + 1
            plt.subplot(pos)
            plt.imshow(valid_images[itr2].astype(np.uint8))
            plt.axis('off')
            plt.title('Original')

            pos = 240 + 2
            plt.subplot(pos)
            # plt.imshow(valid_annotations[itr2].astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(valid_annotations[itr2].astype(
                np.uint8), cmap=ListedColormap(label_colors), norm=clothnorm)
            plt.axis('off')
            plt.title('GT')

            pos = 240 + 3
            plt.subplot(pos)
            # plt.imshow(pred[itr2].astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(pred[itr2].astype(np.uint8),
                       cmap=ListedColormap(label_colors), norm=clothnorm)
            plt.axis('off')
            plt.title('Prediction')

            # Confusion matrix for this image prediction
            crossMat = EvalMetrics.calculate_confusion_matrix(
                valid_annotations[itr2].astype(
                    np.uint8), pred[itr2].astype(
                    np.uint8), num_classes)
            cross_mats.append(crossMat)

            np.savetxt(save_dir +
                       "Crossmatrix" +
                       str(itr1 *
                           flags.batch_size +
                           itr2) +
                       ".csv", crossMat, fmt='%4i', delimiter=',')

            # Save input, gt, pred, crf_pred, sum figures for this image

            """ Generate CRF """
            # 1. run CRF
            crfwithprobsoutput = denseCRF.crf_with_probs(
                valid_images[itr2].astype(np.uint8), predprob[itr2], num_classes)

            # 2. show result display
            crfwithprobspred = crfwithprobsoutput.astype(np.uint8)

            # -----------------------Save inp and masks----------------------
            Utils.save_image(valid_images[itr2].astype(np.uint8), save_dir,
                             name="inp_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(valid_annotations[itr2].astype(np.uint8), save_dir,
                             name="gt_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(pred[itr2].astype(np.uint8),
                             save_dir,
                             name="pred_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(crfwithprobspred, save_dir,
                             name="crf_" + str(itr1 * flags.batch_size + itr2))

            # ----------------------Save visualized masks---------------------
            Utils.save_visualized_image(valid_annotations[itr2].astype(np.uint8), save_dir,
                                        image_name="gt_" + str(itr1 * flags.batch_size + itr2), n_classes=num_classes)
            Utils.save_visualized_image(pred[itr2].astype(np.uint8),
                                        save_dir,
                                        image_name="pred_" + str(itr1 * flags.batch_size + itr2), n_classes=num_classes)
            Utils.save_visualized_image(crfwithprobspred, save_dir, image_name="crf_" + str(
                itr1 * flags.batch_size + itr2), n_classes=num_classes)

            # --------------------------------------------------

            # Confusion matrix for this image prediction with crf
            prob_crf_crossMat = EvalMetrics.calculate_confusion_matrix(
                valid_annotations[itr2].astype(
                    np.uint8), crfwithprobsoutput.astype(
                    np.uint8), num_classes)
            crf_cross_mats.append(prob_crf_crossMat)

            np.savetxt(save_dir +
                       "prob_crf_Crossmatrix" +
                       str(itr1 *
                           flags.batch_size +
                           itr2) +
                       ".csv", prob_crf_crossMat, fmt='%4i', delimiter=',')

            pos = 240 + 4
            plt.subplot(pos)
            # plt.imshow(crfwithprobsoutput.astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(crfwithprobsoutput.astype(np.uint8),
                       cmap=ListedColormap(label_colors), norm=clothnorm)
            plt.axis('off')
            plt.title('Prediction + CRF')

            plt.savefig(save_dir + "resultSum_" +
                        str(itr1 * flags.batch_size + itr2))

            plt.close('all')
            print("Saved image: %d" % (itr1 * flags.batch_size + itr2))

    try:
        total_cm = np.sum(cross_mats, axis=0)
        np.savetxt(
            flags.logs_dir +
            "Crossmatrix.csv",
            total_cm,
            fmt='%4i',
            delimiter=',')

        # print("\n>>> Prediction results (TF functions):")
        # print("Pixel acc:", np.nanmean(tf_pixel_acc_list))
        # print("mean IoU:", np.nanmean(tf_miou_list))

        print("\n>>> Prediction results:")
        EvalMetrics.calculate_eval_metrics_from_confusion_matrix(total_cm, num_classes)

        # Prediction with CRF
        crf_total_cm = np.sum(crf_cross_mats, axis=0)
        np.savetxt(
            flags.logs_dir +
            "CRF_Crossmatrix.csv",
            crf_total_cm,
            fmt='%4i',
            delimiter=',')

        print("\n")
        print("\n>>> Prediction results (CRF):")
        EvalMetrics.calculate_eval_metrics_from_confusion_matrix(crf_total_cm, num_classes)

    except Exception as err:
        print(err)

    end = time.time()
    print("Testing time:", end - start, "seconds")
