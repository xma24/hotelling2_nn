import numpy as np


def get_group_dict_data(args, data, labels):
    group_list = np.unique(labels)
    # print("group_list: ", group_list)

    group_data = {}

    for group_i in range(len(group_list)):
        each_group_data = np.zeros((1, data.shape[1]))
        for graph_i in range(data.shape[0]):
            if labels[graph_i] == group_list[group_i]:
                each_group_data = np.append(each_group_data, data[graph_i].reshape((1, -1)), axis=0)

        each_group_data_update = each_group_data[1:]
        group_data[group_i] = each_group_data_update.reshape((-1, args.node_number, args.feature_dimension)).astype(
            np.float32)

    return group_data


# if __name__ == "__main__":
#     from hotell2_nn.utils.unc_data_loader import unc_data_loader
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
#     unc_node2vec_data, unc_node2vec_labels = unc_data_loader(args)
#     print("unc_node2vec_data.shape: ", unc_node2vec_data.shape)
#     print("unc_node2vec_labels.shape: ", unc_node2vec_labels.shape)
#
#     group_data_ret = get_group_dict_data(args, unc_node2vec_data, unc_node2vec_labels)
#     print("group_data_ret: ", group_data_ret[0].shape)
