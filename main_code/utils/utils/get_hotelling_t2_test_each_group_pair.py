import torch
import numpy as np
from .hotell2_torch import hotell2_torch


def get_hotelling_t2_test_each_group_pair(args, group_data_dict):
    group_number = len(group_data_dict)

    # group_info = torch.from_numpy(np.zeros((1, 4), dtype=np.float32)).to(args.device)

    t2_test_matrix = torch.from_numpy(np.zeros((1, args.node_number + 4), dtype=np.float32)).to(args.device)

    for group_i in range(group_number - 1):
        for group_j in range(group_i + 1, group_number):
            t2_matri_temp = torch.from_numpy(np.zeros((1, args.node_number + 4), dtype=np.float32)).to(args.device)
            t2_matri_temp[0, args.node_number] = group_i

            t2_matri_temp[0, args.node_number + 2] = group_j

            for node_i in range(args.node_number):
                # print("group_data_dict[group_i] shape: ", group_data_dict[group_i].shape)
                group_i_data = group_data_dict[group_i][:, node_i, :]
                group_j_data = group_data_dict[group_j][:, node_i, :]

                t2_matri_temp[0, args.node_number + 1] = group_data_dict[group_i].shape[0]
                t2_matri_temp[0, args.node_number + 3] = group_data_dict[group_j].shape[0]

                group_i_data_trans = torch.transpose(group_i_data, 0, 1)
                group_j_data_trans = torch.transpose(group_j_data, 0, 1)

                t2_score = hotell2_torch(group_i_data_trans, group_j_data_trans)
                # t2 = spm1d.stats.ttest2(group_data_dict[group_i], group_data_dict[group_j])

                # t2_score = t2.inference(alpha=0.05).p_set
                # print("t2_score: ", t2_score)
                t2_matri_temp[0, node_i] = t2_score
            t2_test_matrix = torch.cat((t2_test_matrix, t2_matri_temp), dim=0)

    t2_test_matrix_update = t2_test_matrix[1:]

    return t2_test_matrix_update




# if __name__ == "__main__":
#     from hotell2_nn.utils.get_save_aligned_node2vec_data_labels import \
#         get_save_aligned_node2vec_data_labels
#     from hotell2_nn.utils.load_unc_node2vec_data import load_unc_node2vec_data
#     from hotell2_nn.utils.get_group_data import get_group_data
#     from hotell2_nn.main.parameter_setting import parameter_setting
#
#     node_number_in = 148
#     node2vec_embedding_dim_in = 6
#     graph_number_in = 506
#     repeat_number_in = 1
#
#     unc_data_file_tosave = "./updated_node2vec_embedding_dim_6_repeated_embedding" + "g_" + str(
#         graph_number_in) + "re_" + str(repeat_number_in) + ".txt"
#     print("unc_data_file_tosave: ", unc_data_file_tosave)
#     unc_labels_file_tosave = "./updated_node2vec_embedding_dim_6_unc_aligned_labels" + "g_" + str(
#         graph_number_in) + "re_" + str(repeat_number_in) + ".txt"
#
#     get_save_aligned_node2vec_data_labels("../data_files/updated_node2vec_embedding_dim_6_whole/", unc_data_file_tosave,
#                                           unc_labels_file_tosave,
#                                           graph_number_in, repeat_number_in)
#
#     unc_node2vec_data_ret, unc_node2vec_labels_ret = load_unc_node2vec_data(unc_data_file_tosave,
#                                                                             unc_labels_file_tosave)
#     print("unc_node2vec_data_ret.shape: ", unc_node2vec_data_ret.shape)
#     print("unc_node2vec_labels_ret.shape: ", unc_node2vec_labels_ret.shape)
#
#     group_dataret = get_group_data(unc_node2vec_data_ret, unc_node2vec_labels_ret)
#     print("group_dataret: ", group_dataret)
#
#     args = parameter_setting(1, 1, 0.001)
#
#     for key, value in group_dataret.items():
#         value = torch.from_numpy(value.reshape((-1, args.node_number, args.feature_dimension))).to(args.device)
#         # print("value.shape: ", value.shape)
#         group_dataret[key] = value
#
#     # t2_test_matrix_ret = get_hotelling_t2_test_each_group_pair(args, group_dataret)
#     # print("t2_test_matrix_ret: ", t2_test_matrix_ret)
#     # print("t2_test_matrix_ret.shape: ", t2_test_matrix_ret.shape)
#
#     updated_t2_test_matrix_ret = get_hotelling_t2_test_each_group_pair_update(args, group_dataret)
#     print("updated_t2_test_matrix_ret: ", updated_t2_test_matrix_ret)
#     print("updated_t2_test_matrix_ret.shape: ", updated_t2_test_matrix_ret.shape)
#
#
#
#     # for node_i in range(1):
#     #     print("node_i: ", node_i)
#     #
#     #     group_dataret = get_group_data(
#     #         unc_node2vec_data_ret[:, node_i * node2vec_embedding_dim_in:(node_i + 1) * node2vec_embedding_dim_in],
#     #         unc_node2vec_labels_ret)
#     #     # print("group_dataret: ", group_dataret)
#     #
#     #     print("group_dataret len: ", len(group_dataret))
#     #
#     #     t_test_matrix_ret = get_hotelling_t2_test_each_group_pair(group_dataret)
#     #     print("t_test_matrix_ret: ")
#     #     print(t_test_matrix_ret)
#     #     print("t_test_matrix_ret.shape: ", t_test_matrix_ret.shape)
