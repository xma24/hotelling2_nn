from .pairwise_dist_torch import pairwise_dist_torch


def cdist_kNN_graph_dict_generator_torch(transformed_dict_torch):
    cdist_kNN_graph = {}
    for key, value in transformed_dict_torch.items():
        each_group_data = value

        each_group_cdist_torch = pairwise_dist_torch(each_group_data, each_group_data)
        # print("each_group_cdist_torch: ", each_group_cdist_torch.shape)

        cdist_kNN_graph[key] = each_group_cdist_torch

    return cdist_kNN_graph


# if __name__ == "__main__":
#     from hotell2_nn.main.parameter_setting_expr_4 import parameter_setting
#     from hotell2_nn.utils.unc_data_loader import unc_data_loader
#     import torch
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
#     print("group_data_ret.shape: ", group_data_ret)
#
#     group_data_torch = {}
#
#     for key, value in group_data_ret.items():
#         value_torch = torch.from_numpy(value)
#         group_data_torch[key] = value_torch
#
#     cdist_kNN_graph_dict_generator_torch(group_data_torch)
