import warnings

from two_groups_hotelling_nn_model import ModelStructure
from two_groups_hotelling_nn_train_test import train

warnings.filterwarnings("ignore")

from two_groups_hotelling_nn_parameter_setting import parameter_setting
# experiment_index = 1
# cuda_index = 3
# repeat_number = 1
# learning_rate = 0.1
# output_dimension = 4
# interval = 10
# epoch_number = 5000
# l2_reg = 0.0001
# l1_reg = 0.000001
# gr_reg = 0.000000001

# experiment_index = 100004
# cuda_index = 3
# repeat_number = 1
# learning_rate = 0.001
# output_dimension = 4
# interval = 10
# epoch_number = 50000
# l2_reg = 0.0001
# l1_reg = 0.000001
# gr_reg = 0.00000000001

experiment_index = 7
cuda_index = 2
repeat_number = 1
learning_rate = 0.1
output_dimension = 4
interval = 10
epoch_number = 5000
l2_reg = 0.0001
l1_reg = 0.000001
gr_reg = 0.0000001

args = parameter_setting(cuda_index, repeat_number, learning_rate, output_dimension, interval, epoch_number, l1_reg,
                         experiment_index, gr_reg, l2_reg)

# #### (S) build the model
ttest_model = ModelStructure(args).to(args.device)

# #### (RS)train the model
train(args, ttest_model)

# #### (E) Evaluate the model
# #### 1. Statistical inference (training/test split is not needed)
# #### 2. Generalization capability of the model (split the date set)


