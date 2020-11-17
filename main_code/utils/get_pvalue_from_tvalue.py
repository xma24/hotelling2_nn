import numpy as np
import scipy.stats
from .plotting_results import plotting_results
from .get_hotelling_tvalue_matrix_whole import get_hotelling_tvalue_matrix_whole


def get_pvalue_from_tvalue(args, tvalue_matrix_whole):
    tvalue_matrix = tvalue_matrix_whole[:, :, :-4]
#    print("tvalue_matrix.shape: ", tvalue_matrix.shape)
    group_pair_info = tvalue_matrix_whole[:, :, -4:]
#    print("group_pair_info.shape: ", group_pair_info.shape)

    p_value_number_matrix = np.zeros((tvalue_matrix.shape[1], tvalue_matrix.shape[0]))
    p_value_min_matrix = np.zeros((tvalue_matrix.shape[1], tvalue_matrix.shape[0]))
    shared_accepted_len_array = np.zeros((1, 1))

    for index in range(tvalue_matrix.shape[0]):
        print(" ")
        print("Epoch: {}".format(index * args.interval))

        accepted_index_all = []
        for group_pair_index in range(tvalue_matrix.shape[1]):
            tvalue_vector = tvalue_matrix[index, group_pair_index, :]
            group_a_name = group_pair_info[index, group_pair_index, 0]
            group_a_len = group_pair_info[index, group_pair_index, 1]
            group_b_name = group_pair_info[index, group_pair_index, 2]
            group_b_len = group_pair_info[index, group_pair_index, 3]

            dof_a = group_a_len - 1
            dof_b = group_b_len - 1
            p = args.output_dimension

            f_dist = scipy.stats.f(p, dof_a + dof_b - p - 1)

            hotell2_t2_p_value = 1 - f_dist.cdf(tvalue_vector)
            # print("hotell2_t2_p_value: ", hotell2_t2_p_value)
            hotell2_t2_p_value_min = np.log10(np.min(hotell2_t2_p_value))

            hotel2_boolean = np.where(hotell2_t2_p_value < 0.05 / 148)
            # hotel2_boolean = np.where(hotell2_t2_p_value < 0.005 / 148)

            hotell2_accepted_index = np.array([x+1 for x in hotel2_boolean]).reshape((1, -1))
            # print("hotell2_accepted_index: ", hotell2_accepted_index)

            hotel2_values = hotell2_t2_p_value[hotel2_boolean]
            # print("hotel2_values: ", hotel2_values)
            # print("hotel2_values.shape: ", hotel2_values)

            sorted_index = np.argsort(hotel2_values)
            # print("sorted_index.shape: ", sorted_index)

            sorted_accepted_hotell2_index = []
            sorted_hotel2_values = []

            for i in range(len(sorted_index)):
                # print("i: ", i)
                new_index = hotell2_accepted_index[0, sorted_index[i]]
                # print("new_index: ", new_index)
                sorted_accepted_hotell2_index.append(new_index)
                sorted_hotel2_values.append(hotel2_values[sorted_index[i]])

            print("group_a: {}, group_b: {}, p_min: {}, sorted_accepted_hotell2_index: {} ".format(group_a_name, group_b_name,
                                                                                    hotell2_t2_p_value_min,
                                                                                    sorted_accepted_hotell2_index))
            print("sorted_hotel2_values: ", sorted_hotel2_values)

            accepted_index_all.extend(sorted_accepted_hotell2_index)

            accpted_p_values_len = hotel2_boolean[0].shape[0]
            p_value_number_matrix[group_pair_index, index] = accpted_p_values_len
            p_value_min_matrix[group_pair_index, index] = hotell2_t2_p_value_min

        accepted_index_all_array = np.array(accepted_index_all).reshape((-1, ))
        unique_index = np.unique(accepted_index_all_array)
        print("unique_index: ", unique_index)

        shared_accepted_index = []
        for i in unique_index:
            # print("i: ", i)
            equal_boolean = np.where(accepted_index_all_array == i)
            # print("len(equal_boolean): ", len(equal_boolean[0]))
            if len(equal_boolean[0]) == 10:
                shared_accepted_index.append(i)

        print("final shared_accepted_index: ", shared_accepted_index)

        shared_accepted_index_len = len(shared_accepted_index)

        shared_accepted_len_array = np.append(shared_accepted_len_array, np.array(shared_accepted_index_len).reshape((1, 1)), axis=1)
    shared_accepted_len_array_update = shared_accepted_len_array[0, 1:].reshape((1, -1))
    p_value_number_matrix = np.append(p_value_number_matrix, shared_accepted_len_array_update, axis=0)
    plotting_results(p_value_number_matrix, args.interval,
                     args.data_dir + "p_value_number_matrix_expr_" + str(args.expr_index))
    plotting_results(p_value_min_matrix, args.interval, args.data_dir + "p_value_min_matrix_expr_" + str(args.expr_index))

    # for node_index in range(args.node_number):
    #     tvalue = tvalue_vector[node_index] / 10000000
    #     print("tvalue: ", tvalue)
    #
    #     p_value = (1 - f_dist.cdf(tvalue)) * 6 * 148
    #
    #     print("p_value: ", p_value)


if __name__ == "__main__":
    from G.graph_node2vec_ttest_nn.parameter_setting_expr_10 import parameter_setting

    experiment_index = 10
    cuda_index = 1
    repeat_number = 1
    learning_rate = 0.0001
    output_dimension = 6
    interval = 10

    args = parameter_setting(cuda_index, repeat_number, learning_rate, output_dimension, interval)

    transformed_t_value_save_dir = "../" + args.transform_t_value_dir + str(experiment_index) + "exp_" + str(
        args.feature_dimension) + "rep_" + str(args.repeat_number) + "/"
    print("transformed_t_value_save_dir: ", transformed_t_value_save_dir)

    tvalue_matrix_whole = get_hotelling_tvalue_matrix_whole(args, transformed_t_value_save_dir, "transformed_t_value_")
    print("tvalue_matrix_whole: ", tvalue_matrix_whole)

    get_pvalue_from_tvalue(args, tvalue_matrix_whole, args.output_dimension, experiment_index)
