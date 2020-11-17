import torch.optim as optim
from utils.unc_data_loader import unc_data_loader
import numpy as np
# from notion.e3classification.isbi2020.raw.utils.get_transformed_features import get_transformed_features
# from notion.e3classification.isbi2020.raw.utils.unc_t_value_result_plotting import unc_t_value_result_plotting
import torch
import os
from utils.get_hotelling_t2_test_each_group_pair import get_hotelling_t2_test_each_group_pair
from utils.graph_regularization import graph_regularization
from utils.cdist_kNN_graph_dict_generator_torch import cdist_kNN_graph_dict_generator_torch
from utils.unc_original_graph_loader import unc_original_graph_loader


def train(args, model):
    figure_save_dir = args.data_dir + args.ttest_figure_dir + str(args.expr_index) + "exp_" + str(
        args.output_dimension) + "rep_" + str(
        args.repeat_number) + "/"
    txt_save_dir = args.data_dir + args.ttest_scores_dir + str(args.expr_index) + "exp_" + str(
        args.output_dimension) + "rep_" + str(
        args.repeat_number) + "/"
    top_k_dir = args.data_dir + args.top_k_dir + str(args.expr_index) + "exp_" + str(
        args.output_dimension) + "rep_" + str(
        args.repeat_number) + "/"
    transformed_feature_save_dir = args.data_dir + args.transform_features_dir + str(args.expr_index) + "exp_" + str(
        args.output_dimension) + "rep_" + str(args.repeat_number) + "/"

    loss_save_dir = args.data_dir + args.loss_dir + str(args.expr_index) + "exp_" + str(
        args.output_dimension) + "rep_" + str(args.repeat_number) + "/"

    if not os.path.exists(figure_save_dir):
        os.makedirs(figure_save_dir)

    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir)

    if not os.path.exists(transformed_feature_save_dir):
        os.makedirs(transformed_feature_save_dir)

    if not os.path.exists(top_k_dir):
        os.makedirs(top_k_dir)

    if not os.path.exists(loss_save_dir):
        os.makedirs(loss_save_dir)

    input_graph = unc_original_graph_loader(args)
    model.train()
    optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate)
    multiple_group_data_ret = unc_data_loader(args)

    loss_total = []
    for epoch in range(args.epoch_number):
        print("epoch: ", epoch)

        optimizer.zero_grad()

        # #### make sure the input dict is on the gpu
        group_data2gpu = {}
        for key, value in multiple_group_data_ret.items():
            unc_data_torch = torch.from_numpy(value).to(args.device)
            group_data_cuda = unc_data_torch
            group_data2gpu[key] = group_data_cuda

        # #### make the transformation
        if epoch == 0:
            group_data_transform_ret, hotell2_group_matrix_ret, t2_node_matrix_ret, t2_node_matrix_min_ret, intra_t2_vector_ret = model(group_data2gpu, False)
        else:
            group_data_transform_ret, hotell2_group_matrix_ret, t2_node_matrix_ret, t2_node_matrix_min_ret, intra_t2_vector_ret = model(group_data2gpu, True)

        cdist_matrix_torch = cdist_kNN_graph_dict_generator_torch(group_data_transform_ret)

        graph_regularization_loss = graph_regularization(args, input_graph, cdist_matrix_torch)[0]
#        print("  graph_regularization_loss: ", graph_regularization_loss)

#        print("     t2_node_matrix_min_ret: ", t2_node_matrix_min_ret)
#        print("   intra_t2_vector_ret: ", intra_t2_vector_ret)

        ttest_loss = 1.0 / (hotell2_group_matrix_ret.mean())

        
        l1_regularization_loss = None
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == "hotell_merge_weights":
                    if l1_regularization_loss is None:
                        l1_regularization_loss = param.norm(1)
                    else:
                        l1_regularization_loss = l1_regularization_loss + param.norm(1)

        l2_reg = None
        for name, param in model.named_parameters():
            if not (name == "hotell_merge_weights"):
                if l2_reg is None:
                    l2_reg = param.norm(2)
                else:
                    l2_reg = l2_reg + param.norm(2)

        l1_regularization_loss_update = args.l1_reg * l1_regularization_loss
#        print("     l1_regularization_loss_update: ", l1_regularization_loss_update)

        l2_regularization_loss_update = args.l2_reg * l2_reg
#        print("     l2_regularization_loss_update: ", l2_regularization_loss_update)

        greg_regularization_loss_update = args.gr_reg * graph_regularization_loss
#        print("        greg_regularization_loss_update: ", greg_regularization_loss_update)

        ttest_loss += l1_regularization_loss_update + greg_regularization_loss_update + l2_regularization_loss_update
        print("ttest_loss: ", ttest_loss)
        ttest_loss.backward()
        optimizer.step()

        if epoch % args.interval == 0:
            print("Epoch: {}, Training loss: {}, ".format(epoch, ttest_loss))

            test(args, model, epoch)

            # get and save t values
            t_test_matrix_numpy = t2_node_matrix_ret.detach().cpu().numpy()
#            print("t_test_matrix_numpy.shape: ", t_test_matrix_numpy.shape)
            np.savetxt(txt_save_dir + "t_value_" + str(epoch) + ".txt", t_test_matrix_numpy)

            loss_total.append(ttest_loss)
            np.savetxt(loss_save_dir + "loss_" + str(epoch) + ".txt", np.array(loss_total).reshape((-1,)))

            # get and save transformed features

            # if epoch == 0:
            #     dimension = args.feature_dimension
            # else:
            #     dimension = args.output_dimension
            #
            # transformed_features, transformed_labels = get_transformed_features(group_data_transform_ret, dimension)
            #
            # np.savetxt(transformed_feature_save_dir + "transformed_features_" + str(epoch) + ".txt",
            #            transformed_features.reshape((transformed_features.shape[0], -1)))
            # np.savetxt(transformed_feature_save_dir + "transformed_labels_" + str(epoch) + ".txt", transformed_labels)

            # # log_ttest_matrix = log_vector(t_test_matrix_numpy.reshape((-1, 1)))
            # # save normalized t values to figures
            # normalized_ttest_matrix = t_test_matrix_numpy / t_test_matrix_numpy.max()
            # normalized_ttest_matrix_append = np.append(normalized_ttest_matrix, np.zeros((1, 21)))
            # t_test_matrix_numpy_update = normalized_ttest_matrix_append.reshape((13, 13))
            # save_fig_with_labels(t_test_matrix_numpy_update, figure_save_dir + "norm_" + str(epoch))
            #
            # save unnormalized t values to figures

            # unnormalized_t_values = t_test_matrix_numpy
            # top_k_index = np.argpartition(unnormalized_t_values.reshape((-1,)), args.top_k)[args.top_k:] + 1
            # print("       Top_K Eopch: {}".format(epoch))
            # print(top_k_index)
            # np.savetxt(top_k_dir + "top_k_index_" + str(epoch) + ".txt", top_k_index)
            #
            # # unnormalized_t_values_append = np.append(unnormalized_t_values, np.zeros((1, 21)))
            # # unnormalized_t_values_reshape = unnormalized_t_values_append.reshape((13, 13))
            # unc_t_value_result_plotting(epoch, unnormalized_t_values, figure_save_dir, 0)

            # save_fig_with_labels(unnormalized_t_values_reshape, figure_save_dir + str(epoch))


def test(args, model, epcoh_number):
    model.eval()

    transformed_t_value_save_dir = args.data_dir + args.transform_t_value_dir + str(args.expr_index) + "exp_" + str(
        args.output_dimension) + "rep_" + str(args.repeat_number) + "/"

    if not os.path.exists(transformed_t_value_save_dir):
        os.makedirs(transformed_t_value_save_dir)

    with torch.no_grad():
        # multiple_group_data_ret = unc_data_loader_2_groups(args)
        multiple_group_data_ret_test = unc_data_loader(args)

        group_data2gpu_test = {}
        for key, value in multiple_group_data_ret_test.items():
            unc_data_torch = torch.from_numpy(value).to(args.device)
            group_data_cuda = unc_data_torch
            group_data2gpu_test[key] = group_data_cuda

        # #### make the transformation
        if epcoh_number == 0:
            group_data_transform_ret, hotell2_group_matrix_ret, t2_node_matrix_ret, t2_node_matrix_min_ret, intra_t2_vector_ret = model(group_data2gpu_test, False)
        else:
            group_data_transform_ret, hotell2_group_matrix_ret, t2_node_matrix_ret, t2_node_matrix_min_ret, intra_t2_vector_ret = model(group_data2gpu_test, True)

        hotelling_t2_test_ret = get_hotelling_t2_test_each_group_pair(args, group_data_transform_ret)

        hotelling_t2_test_np = hotelling_t2_test_ret.detach().cpu().numpy()

        np.savetxt(transformed_t_value_save_dir + "transformed_t_value_" + str(epcoh_number) + ".txt",
                   hotelling_t2_test_np)
