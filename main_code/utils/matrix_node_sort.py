import numpy as np


def matrix_node_sort(unsorted_matrix):
    """
    1. The output of node2vec is unordered. We need to sort those nodes.
    """
    feature_number = unsorted_matrix.shape[1] - 1
    node_number = unsorted_matrix.shape[0]

    sorted_matrix = np.zeros((node_number, feature_number), dtype=np.float32)

    for i in range(unsorted_matrix.shape[0]):
        node_index = int(unsorted_matrix[i, 0])
        # print("node_index: ", node_index)
        node_features = unsorted_matrix[i, 1:]

        sorted_matrix[node_index] = node_features

    return sorted_matrix


if __name__ == "__main__":
    unsorted_matrix_file = "../data/node2vec_embedding_dataset/unc_brain_node2vec_checking_g_0re_0.txt"
    unsorted_matrix = np.loadtxt(unsorted_matrix_file, skiprows=1)
    print("unsorted_matrix.shape: ", unsorted_matrix.shape)
    sorted_matrix_ret = matrix_node_sort(unsorted_matrix)
    print("sorted_matrix_ret.shape: ", sorted_matrix_ret)
