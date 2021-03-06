import numpy as np
#from ..two_groups_hotelling_nn_parameter_setting import parameter_setting


def unc_original_graph_loader_2g(args):
    group_number = 2
    group_index_dict = {0: [0, 3], 1: [1, 2]}

    unc_raw_data_labels_file = args.data_dir + "unc_data_labels_aligned.txt"
    unc_raw_data_labels = np.loadtxt(unc_raw_data_labels_file)
    # print("unc_raw_data_labels.shape: ", unc_raw_data_labels.shape)

    unc_raw_data = unc_raw_data_labels[:, :-1]
    unc_labels = unc_raw_data_labels[:, -1]

    unc_raw_data_norm = unc_raw_data / unc_raw_data.max()
    # print("unc_raw_data_norm: ", unc_raw_data_norm)

    unc_orignal_graph_dict = {}

    unique_labels_array = np.array(np.unique(unc_labels)).reshape((1, -1))
    # print("unique_labels_array: ", unique_labels_array)

    for uni_labels_i in range(group_number):
        group_graph = np.zeros((1, args.node_number, args.node_number), dtype=np.float32)

        for graph_i in range(unc_raw_data_norm.shape[0]):
            if unc_labels[graph_i] in group_index_dict[uni_labels_i]:
                each_graph = unc_raw_data_norm[graph_i].reshape((args.node_number, args.node_number))
                each_graph_ext = np.expand_dims(each_graph, axis=0)
                for repeat_index in range(args.repeat_number):
                    group_graph = np.append(group_graph, each_graph_ext, axis=0)

        group_graph_update = group_graph[1:]
        # print("group_graph_update.shape: ", group_graph_update.shape)
        # print("group_graph_update[0]: ", group_graph_update[0])
        unc_orignal_graph_dict[uni_labels_i] = group_graph_update

    return unc_orignal_graph_dict


#if __name__ == "__main__":
#    experiment_index = 100000
#    cuda_index = 0
#    repeat_number = 1
#    learning_rate = 0.001
#    output_dimension = 6
#    interval = 5
#    epoch_number = 20000
#    l1_reg = 0.001
#
#    args = parameter_setting(cuda_index, repeat_number, learning_rate, output_dimension, interval, epoch_number, l1_reg,
#                             experiment_index)
#    unc_original_graph_loader_2g(args)
