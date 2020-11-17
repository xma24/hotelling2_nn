import argparse
import torch


def parameter_setting(cuda_index, repeat_num, learning_rate, output_dimension, interval, epoch_num, l1_reg, expr_index,
                      gr_reg, l2_reg):
    parser = argparse.ArgumentParser(description="Graph T Test")

    parser.add_argument("--batch_size", default=20)
    parser.add_argument("--learning_rate", default=learning_rate)
    parser.add_argument("--epoch_number", default=epoch_num)
    parser.add_argument("--ttest_figure_dir", default="ttest_figure_dir/")
    parser.add_argument("--class_number", default=2)
    parser.add_argument("--hidden_dimension", default=64)
    parser.add_argument("--feature_dimension", default=6)
    parser.add_argument("--output_dimension", default=output_dimension)
    parser.add_argument("--ttest_scores_dir", default="ttest_scores_dir/")
    parser.add_argument("--transform_features_dir", default="transform_features_dir/")
    parser.add_argument("--node_number", default=148)
    parser.add_argument("--repeat_number", default=repeat_num)
    parser.add_argument("--top_k", default=-20)
    parser.add_argument("--top_k_dir", default="top_k_dir/")
    parser.add_argument("--transform_t_value_dir", default="transform_t_value_dir/")
    parser.add_argument("--interval", default=interval)
    parser.add_argument("--l1_reg", default=l1_reg)
    parser.add_argument("--loss_dir", default="loss_dir/")
    parser.add_argument("--node2vec_dir", default="../data/node2vec_embedding_dataset/")
    parser.add_argument("--node2vec_prefix", default="unc_brain_node2vec_checking_")
    parser.add_argument("--expr_index", default=expr_index)
    parser.add_argument("--node2vec_labels_path", default="../data/unc_raw_labels.txt")
    parser.add_argument("--unc_raw_data_labels_path", default="../data/unc_raw_data_labels.txt")
    parser.add_argument("--graph_number", default=506)
    parser.add_argument("--data_dir", default="../data/")
    parser.add_argument("--unc_data_dir", default="../data/unc_raw_data/AD-Data/")
    parser.add_argument("--unc_labels_excel_path", default="../data/unc_raw_data/DataTS.csv")
    parser.add_argument("--train_sample_frac", default=0.5)
    parser.add_argument("--gr_reg", default=gr_reg)
    parser.add_argument("--l2_reg", default=l2_reg)

    args = parser.parse_args()

    args.kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}

    args.device = torch.device("cuda:" + str(cuda_index) if torch.cuda.is_available() else "cpu")

    return args
