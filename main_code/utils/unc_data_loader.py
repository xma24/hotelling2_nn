import numpy as np

from .get_save_aligned_node2vec_data_labels import get_save_aligned_node2vec_data_labels
# from load_unc_node2vec_data import load_unc_node2vec_data
from .get_group_dict_data import get_group_dict_data


def unc_data_loader(args):
    # node_number_in = args.node_number
    # node2vec_embedding_dim_in = args.feature_dimension
    # graph_number_in = 506
    # repeat_number_in = args.repeat_number

    # unc_data_file_tosave = "../data/" + args.node2vec_prefix + "data_g_" + str(
    #     args.graph_number) + "re_" + str(args.repeat_number) + "expr_" + str(args.expr_index) + ".txt"
    # unc_labels_file_tosave = "../data/" + args.node2vec_prefix + "labels_g_" + str(
    #     args.graph_number) + "re_" + str(args.repeat_number) + "expr_" + str(args.expr_index) + ".txt"

    unc_node2vec_data_ret, unc_node2vec_labels_ret = get_save_aligned_node2vec_data_labels(args)

    # unc_node2vec_data_ret, unc_node2vec_labels_ret = load_unc_node2vec_data(unc_data_file_tosave,
    #                                                                         unc_labels_file_tosave)
    # print("unc_node2vec_data_ret.shape: ", unc_node2vec_data_ret.shape)
    # print("unc_node2vec_labels_ret.shape: ", unc_node2vec_labels_ret.shape)

    # for repeat_index in range(repeat_number):
    #     group_dataret = get_group_data(unc_node2vec_data_ret[:, repeat_index * node2vec_embedding_dim_in:(repeat_index + 1) * node2vec_embedding_dim_in], unc_node2vec_labels_ret)
    #     print("group_dataret: ", group_dataret)
    #
    #     print("group_dataret len: ", len(group_dataret))

    group_data = get_group_dict_data(args, unc_node2vec_data_ret.astype(np.float32), unc_node2vec_labels_ret.astype(int))

    return group_data


# if __name__ == "__main__":
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
#     group_data_ret = unc_data_loader(args)
#
#     for key, value in group_data_ret.items():
#         print("value: ", value.shape)
#
#     print("group_data_ret.shape: ", len(group_data_ret))
