import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils.hotell2_torch import hotell2_torch


class ModelStructure(nn.Module):
    def __init__(self, args):
        super(ModelStructure, self).__init__()
        self.hidden_dimension = args.hidden_dimension
        self.hidden_layer2_dimension = 2000
        self.node_number = args.node_number
        self.feature_dimension = args.feature_dimension
        self.output_dimension = args.output_dimension

        self.Linear1 = nn.Linear(args.feature_dimension, args.hidden_dimension)
        self.dense1_bn = nn.BatchNorm1d(args.hidden_dimension)
        self.Linear2 = nn.Linear(args.hidden_dimension, 2000)
        self.dense2_bn = nn.BatchNorm1d(2000)
        self.Linear3 = nn.Linear(2000, args.hidden_dimension)
        self.dense3_bn = nn.BatchNorm1d(args.hidden_dimension)
        self.Linear4 = nn.Linear(args.hidden_dimension, args.output_dimension)
        self.dense4_bn = nn.BatchNorm1d(args.output_dimension)
        self.node_number = args.node_number
        self.args = args

        self.hotell_merge_weights = torch.nn.Parameter(torch.randn(args.node_number, args.node_number))
        torch.nn.init.xavier_uniform(self.hotell_merge_weights)
        self.hotell_merge_weights.requires_grad = True

    def forward(self, x_dict, transform_flag):
        group_data_transform = {}

        for key, value in x_dict.items():
            group_data_tranform = self.each_forward(value, transform_flag)
#            print("key: {}, transformed_data_shape: {} ".format(key, group_data_tranform.shape))
#            print("self.hotell_merge_weights: ", self.hotell_merge_weights.view(1, -1))
            group_data_transform[key] = group_data_tranform

        hotell2_group_matrix, t2_node_matrix, t2_node_matrix_min, intra_t2_vector = self.hotell2_merge(group_data_transform, transform_flag)

        return group_data_transform, hotell2_group_matrix, t2_node_matrix, t2_node_matrix_min, intra_t2_vector

    def each_forward(self, x, transform_flag):

        if transform_flag:
            x = x.view(-1, self.feature_dimension)
            x = self.Linear1(x)
#            print("x: ", x.shape)
            x = F.relu(self.dense1_bn(x))
#            print("x layer1: ", x.shape)
            x = F.relu(self.dense2_bn(self.Linear2(x)))
#            print("x layer2: ", x.shape)
            x = F.relu(self.dense3_bn(self.Linear3(x)))
#            print("x layer3: ", x.shape)
            x = F.leaky_relu(self.dense4_bn(self.Linear4(x)))
#            print("x layer4: ", x)
            x_transform = x.view(-1, self.node_number, self.output_dimension)
            # print("x_transform: ", x)
        else:
            x_transform = x

        return x_transform

    def hotell2_merge(self, group_data_dict, transform_flag):

        group_number = len(group_data_dict)

        t2_raw_matrix = torch.from_numpy(np.zeros((1, self.node_number), dtype=np.float32)).to(
            self.args.device)

        t2_group_matrix = torch.from_numpy(np.zeros((1, 1), dtype=np.float32)).to(
            self.args.device)

        for group_i in range(group_number - 1):
            for group_j in range(group_i + 1, group_number):

                each_group_pair_t2_vector = torch.from_numpy(np.zeros((1, self.node_number), dtype=np.float32)).to(
                    self.args.device)

                for node_i in range(self.node_number):
                    group_i_data = group_data_dict[group_i][:, node_i, :]
                    group_j_data = group_data_dict[group_j][:, node_i, :]

                    group_i_data_trans = torch.transpose(group_i_data, 0, 1)
                    group_j_data_trans = torch.transpose(group_j_data, 0, 1)

                    # t2_score = each_hotelling_t2_test_torch(group_i_data_trans, group_j_data_trans)
                    t2_score = F.relu(hotell2_torch(group_i_data_trans, group_j_data_trans))
                    # print("t2_score: ", t2_score)

                    # t2_score = t2.inference(alpha=0.05).p_set
                    # print("t2_score: ", t2_score)
                    each_group_pair_t2_vector[0, node_i] = t2_score

                each_group_pair_t2_vector = each_group_pair_t2_vector.view((1, self.node_number))

                if transform_flag:

                    # sparse_weigths = torch.mul(self.hotell_merge_weights, self.hotell_merge_weights)
                    #
                    # transformed_each_group_pair_t2 = F.relu(torch.mm(each_group_pair_t2_vector,
                    #                                                  torch.mm(sparse_weigths,
                    #                                                           sparse_weigths.transpose(0,
                    #                                                                                               1))))
                    # transformed_each_group_pair_t2 = F.relu(each_group_pair_t2_vector * self.hotell_merge_weights)
                    # print("transformed_each_group_pair_t2: ", transformed_each_group_pair_t2)

                    transformed_each_group_pair_t2 = F.relu(
                        torch.mm(each_group_pair_t2_vector, self.hotell_merge_weights))
                else:
                    transformed_each_group_pair_t2 = each_group_pair_t2_vector
                    # print("not transformed_each_group_pair_t2: ", transformed_each_group_pair_t2)

                t2_raw_matrix = torch.cat((t2_raw_matrix, each_group_pair_t2_vector), dim=0)
                # print("t2_raw_matrix.shape: ", t2_raw_matrix.shape)

                t2_group_matrix = torch.cat((t2_group_matrix, transformed_each_group_pair_t2.sum().view(1, 1)), dim=1)

        t2_node_matrix = torch.sum(t2_raw_matrix[1:, :], dim=0)   # #### (1, 148)
        t2_node_matrix_min, _ = torch.min(t2_raw_matrix[1:, :], dim=0)
        # print("t2_node_matrix.shape: ", t2_node_matrix.shape)
        t2_group_matrix_update = t2_group_matrix[0, 1:]
#        print("t2_group_matrix_update: ", t2_group_matrix_update)

        each_group_t2_mean = torch.mean(t2_raw_matrix[1:, :], dim=0)
        each_group_t2_diff = torch.mul(torch.abs(t2_raw_matrix[1:, :] - each_group_t2_mean), each_group_t2_mean)
        intra_t2_vector = torch.sum(each_group_t2_diff, dim=0)

        return t2_group_matrix_update, t2_node_matrix, t2_node_matrix_min, intra_t2_vector
