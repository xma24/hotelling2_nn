import numpy as np
import torch


def graph_regularization(args, kNN_graph_input, cdist_graph_transformed):
    graph_regularization_loss = torch.from_numpy(np.zeros((1,), dtype=np.float32)).to(args.device)

    group_number = len(kNN_graph_input)

    for key, value in kNN_graph_input.items():
        each_group_kNN_graphs = torch.from_numpy(np.array(value, dtype=np.float32)).to(args.device)
        each_cdist_matrix = cdist_graph_transformed[key]

        # np.savetxt(args.data_dir + args.top_k_dir + str(key) + "each_group_kNN_graphs.txt", value[0])
        # np.savetxt(args.data_dir + args.top_k_dir + str(key) + "each_cdist_matrix.txt", each_cdist_matrix.detach().cpu().numpy()[0])
        # save_fig(value[0], args.data_dir + args.top_k_dir + str(key) + "each_group_kNN_graphs")
        # save_fig(each_cdist_matrix.detach().cpu().numpy()[0], args.data_dir + args.top_k_dir + str(key) + "each_cdist_matrix")

        # print("each_group_kNN_graphs.shape: ", each_group_kNN_graphs.shape)
        # print("each_cdist_matrix.shape: ", each_cdist_matrix.shape)
        each_group_loss_matrix = torch.mul(each_group_kNN_graphs, each_cdist_matrix)
        # np.savetxt(args.data_dir + args.top_k_dir + str(key) + "each_group_loss_matrix.txt", each_group_loss_matrix.detach().cpu().numpy()[0])
        # save_fig(each_group_loss_matrix.detach().cpu().numpy()[0], args.data_dir + args.top_k_dir + str(key) + "each_group_loss_matrix")
        # print("each_group_loss_matrix.shape: ", each_group_loss_matrix)
        each_group_loss = each_group_loss_matrix.mean()
#        print("each_group_loss: ", each_group_loss)
        graph_regularization_loss[0] += each_group_loss

    return graph_regularization_loss


# if __name__ == "__main__":
#     from hotell2_nn.main.parameter_setting_expr_4 import parameter_setting
#     from hotell2_nn.utils.kNN_graph_generator import kNN_graph_generator
#     from hotell2_nn.utils.unc_data_loader import unc_data_loader
#     from hotell2_nn.utils.cdist_kNN_graph_dict_generator_torch import cdist_kNN_graph_dict_generator_torch
#     from hotell2_nn.utils.save_fig import save_fig
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
#     knn_graph_ret = kNN_graph_generator(args)
#     print("knn_graph_ret: ", knn_graph_ret[0])
#     save_fig(knn_graph_ret[0][10], args.data_dir + args.top_k_dir + "knn_graph_ret[0][10]")
#
#     group_data_ret = unc_data_loader(args)
#     # print("group_data_ret.shape: ", group_data_ret)
#
#     group_data_torch = {}
#
#     for key, value in group_data_ret.items():
#         value_torch = torch.from_numpy(value).to(args.device)
#         group_data_torch[key] = value_torch
#
#     cdist_matrix_torch = cdist_kNN_graph_dict_generator_torch(group_data_torch)
#     print("cdist_matrix_torch: ", cdist_matrix_torch[0])
#     save_fig(cdist_matrix_torch[0].detach().cpu().numpy()[10],
#              args.data_dir + args.top_k_dir + "cdist_matrix_torch.detach().cpu().numpy()[0][10]")
#
#     gr_loss = graph_regularization(args, knn_graph_ret, cdist_matrix_torch)
#     print("gr_loss: ", gr_loss)
