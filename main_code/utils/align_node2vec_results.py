import numpy as np
from .matrix_node_sort import matrix_node_sort


def align_node2vec_results(args):
    """
    1. 27 graphs and each graph includes 50 repeats.
    """

    # file_names_list = list_filenames(file_folder)
    # file_number = len(file_names_list)

    init_matrix_with_nodeindex = np.loadtxt(args.node2vec_dir + args.node2vec_prefix + "g_" + str(0) + "re_" + str(0) + ".txt",
                                            skiprows=1)
    init_matrix_update = init_matrix_with_nodeindex[:, 1:]
    unc_node2vec_matrix = np.zeros(init_matrix_update.shape, dtype=np.float32)
    unc_node2vec_matrix_ext = np.expand_dims(unc_node2vec_matrix, axis=0)

    for graph_i in range(args.graph_number):
        for repeat_i in range(args.repeat_number):
            file_name_i = args.node2vec_dir + args.node2vec_prefix + "g_" + str(graph_i) + "re_" + str(repeat_i) + ".txt"
            unsorted_matrix_ret = np.loadtxt(file_name_i, skiprows=1)

            sorted_matrix_ret = matrix_node_sort(unsorted_matrix_ret)  # (148, 50)
            sorted_matrix_ret_ext = np.expand_dims(sorted_matrix_ret, axis=0)
            unc_node2vec_matrix_ext = np.append(unc_node2vec_matrix_ext, sorted_matrix_ret_ext, axis=0)

    unc_node2vec_matrix_update = unc_node2vec_matrix_ext[1:]

    return unc_node2vec_matrix_update


# if __name__ == "__main__":
#     # unsorted_matrix_file = "../data/node2vec_embedding_dataset/unc_brain_node2vec_checking_g_0re_0.txt"
#     # unsorted_matrix = np.loadtxt(unsorted_matrix_file, skiprows=1)
#     # print("unsorted_matrix.shape: ", unsorted_matrix.shape)
#     # sorted_matrix_ret = matrix_node_sort(unsorted_matrix)
#     # print("sorted_matrix_ret.shape: ", sorted_matrix_ret)
#
#     from hotell2_nn.main.parameter_setting_expr_1 import parameter_setting
#
#     experiment_index = 2202
#     cuda_index = 0
#     repeat_number = 1
#     learning_rate = 0.001
#     output_dimension = 6
#     interval = 5
#     epoch_number = 20000
#     l1_reg = 0.001
#
#     args = parameter_setting(cuda_index, repeat_number, learning_rate, output_dimension, interval, epoch_number, l1_reg,
#                              experiment_index)
#
#     unc_node2vec = align_node2vec_results(args)
#     print("unc_node2vec[0]: ", unc_node2vec[0])


