import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")


def main(argv=None):
    # save list of error to file
    pred_eUnet = list()
    pred_eFCN = list()
    with open(FLAGS.logs_dir + 'FCN/pred_e.txt', 'r') as file:
        for line in file:
            pred_eFCN.append(int(line))
    with open(FLAGS.logs_dir + 'UN/pred_e.txt', 'r') as file:
        for line in file:
            pred_eUnet.append(int(line))
    # compare
    num_unet_good = 0
    num_FCN_good = 0
    sum_error_unet = 0
    sum_error_FCN = 0
    sum_better_error_unet = 0
    sum_better_error_FCN = 0
    for i in range(len(pred_eUnet)):
        sum_error_unet = sum_error_unet + pred_eUnet[i]
        sum_error_FCN = sum_error_FCN + pred_eFCN[i]
        if (pred_eUnet[i] >= pred_eFCN[i]):
            print(i)
            num_unet_good = num_unet_good + 1
        else:
            num_FCN_good = num_FCN_good + 1
        if (pred_eUnet[i] >= pred_eFCN[i]):
            sum_better_error_unet = sum_better_error_unet + \
                (pred_eUnet[i] - pred_eFCN[i])
        else:
            sum_better_error_FCN = sum_better_error_FCN - \
                (pred_eUnet[i] - pred_eFCN[i])
    print("num_unet_good:", num_unet_good)
    print("num_FCN_good:", num_FCN_good)
    print("sum_error_unet:", sum_error_unet)
    print("sum_error_FCN:", sum_error_FCN)
    print("sum_better_error_unet:", sum_better_error_unet)
    print("sum_better_error_FCN:", sum_better_error_FCN)
    print("sum_better_error_unet/num_unet_good:",
          sum_better_error_unet/float(num_unet_good))
    print("sum_better_error_FCN/num_FCN_good:",
          sum_better_error_FCN/float(num_FCN_good))


if __name__ == "__main__":
    main()
