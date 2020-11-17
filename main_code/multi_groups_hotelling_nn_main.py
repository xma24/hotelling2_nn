import warnings

from multi_groups_hotelling_nn_model import ModelStructure
from multi_groups_hotelling_nn_train_test import train

warnings.filterwarnings("ignore")

from multi_groups_hotelling_nn_parameter_setting import parameter_setting
experiment_index = 10607
cuda_index = 1
repeat_number = 1
learning_rate = 0.1
output_dimension = 4
interval = 10
epoch_number = 5000
l2_reg = 0.001
l1_reg = 0.00001
gr_reg = 0.000001
args = parameter_setting(cuda_index, repeat_number, learning_rate, output_dimension, interval, epoch_number, l1_reg,
                         experiment_index, gr_reg, l2_reg)

ttest_model = ModelStructure(args).to(args.device)
train(args, ttest_model)



# #### get the results
from utils.get_hotelling_tvalue_matrix_whole import get_hotelling_tvalue_matrix_whole
from utils.get_pvalue_from_tvalue import get_pvalue_from_tvalue
transformed_t_value_save_dir = "../data/" + args.transform_t_value_dir + str(experiment_index) + "exp_" + str(
    args.output_dimension) + "rep_" + str(args.repeat_number) + "/"
print("transformed_t_value_save_dir: ", transformed_t_value_save_dir)

tvalue_matrix_whole = get_hotelling_tvalue_matrix_whole(args, transformed_t_value_save_dir, "transformed_t_value_")
print("tvalue_matrix_whole: ", tvalue_matrix_whole)

get_pvalue_from_tvalue(args, tvalue_matrix_whole)
