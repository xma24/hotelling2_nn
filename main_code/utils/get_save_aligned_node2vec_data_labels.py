import numpy as np
from .align_node2vec_results import align_node2vec_results
from .get_aligned_labels import get_aligned_labels


def get_save_aligned_node2vec_data_labels(args):
    aligned_data = align_node2vec_results(args)
    aligned_data_reshape = np.reshape(aligned_data, (aligned_data.shape[0], -1))
    # print("aligned_data_reshape: ", aligned_data_reshape.shape)

    # #### save the data into txt file
    embedding_save_file = args.data_dir + args.node2vec_prefix + str(args.repeat_number) + "_data_" + str(
        args.expr_index) + ".txt"
    np.savetxt(embedding_save_file, aligned_data_reshape)

    # #### get and save corresponding labels
    unc_raw_labels = np.loadtxt(args.node2vec_labels_path)
    aligned_labels_ret = get_aligned_labels(args, unc_raw_labels)
    unc_aligned_labels_save_file = args.data_dir + args.node2vec_prefix + str(
        args.repeat_number) + "_aligned_labels_" + str(args.expr_index) + ".txt"
    np.savetxt(unc_aligned_labels_save_file, aligned_labels_ret)

    return aligned_data_reshape, aligned_labels_ret




# if __name__ == "__main__":
#
#     from hotell2_nn.main.a5_main_motor_hotelling_expr_2g import parameter_setting
#
#     experiment_index = 2202
#     cuda_index = 0
#     repeat_number = 1
#     learning_rate = 0.001
#     output_dimension = 50
#     interval = 5
#     epoch_number = 20000
#     l1_reg = 0.001
#
#     args = parameter_setting(cuda_index, repeat_number, learning_rate, output_dimension, interval, epoch_number, l1_reg,
#                              experiment_index)
#
#     data, labels = get_save_aligned_node2vec_data_labels(args)
#     print("data[0]: ", data[0])
