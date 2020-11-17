import numpy as np
from .list_filenames import list_filenames


def get_hotelling_tvalue_matrix_whole(args, tvalue_folder, tvalue_name_pre):
    group_pair_number = 10
    print_interval = args.interval

    init_tvalue_filename = tvalue_folder + tvalue_name_pre + str(print_interval * 0) + ".txt"
    init_tvalue_matrix_example = np.loadtxt(init_tvalue_filename)

    if len(init_tvalue_matrix_example.shape) == 1:
        init_tvalue_matrix_example = init_tvalue_matrix_example.reshape((1, -1))

#    print("init_tvalue_matrix_example.shape: ", init_tvalue_matrix_example.shape)

    tvalue_matrix = np.zeros((1, init_tvalue_matrix_example.shape[0], init_tvalue_matrix_example.shape[1]))

    file_name_list = list_filenames(tvalue_folder)
#    print("file_name_list: ", file_name_list)
#    print("file_name_list len: ", len(file_name_list))

    for i in range(len(file_name_list)):
        each_filename = tvalue_name_pre + str(print_interval * i) + ".txt"
        # print("each_filename: ", each_filename)

        each_file_path = tvalue_folder + each_filename

        each_tvalue_matrix = np.loadtxt(each_file_path)

        if len(each_tvalue_matrix.shape) == 1:
            each_tvalue_matrix = np.reshape(each_tvalue_matrix, (1, -1))

        each_tvalue_matrix_ext = np.expand_dims(each_tvalue_matrix, axis=0)
        tvalue_matrix = np.append(tvalue_matrix, each_tvalue_matrix_ext, axis=0)

    tvalue_matrix_update = tvalue_matrix[1:]
#    print("tvalue_matrix_update.shape: ", tvalue_matrix_update.shape)

    return tvalue_matrix_update


if __name__ == "__main__":
    from G.graph_node2vec_ttest_nn.parameter_setting_expr_7 import parameter_setting

    experiment_index = 7
    cuda_index = 1
    repeat_number = 5
    learning_rate = 0.001
    output_dimension = 6

    args = parameter_setting(cuda_index, repeat_number, learning_rate, output_dimension)

    transformed_t_value_save_dir = "../" + args.transform_t_value_dir + str(experiment_index) + "exp_" + str(
        args.feature_dimension) + "rep_" + str(args.repeat_number) + "/"
    print("transformed_t_value_save_dir: ", transformed_t_value_save_dir)

    tvalue_matrix_whole = get_hotelling_tvalue_matrix_whole(args, transformed_t_value_save_dir, "transformed_t_value_")

    # transformed_t_value_save_dir + "transformed_t_value_" + str(epcoh_number) + ".txt"
