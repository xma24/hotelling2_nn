import numpy as np


def get_aligned_labels(args, unc_labels):
    aligned_labels = []

    repeat_array = np.ones((1, args.repeat_number))

    for graph_i in range(args.graph_number):
        each_graph_labels = list(int(unc_labels[graph_i]) * repeat_array)
        aligned_labels.append(each_graph_labels)

    aligned_labels_update = np.array(aligned_labels).reshape((-1, 1))

    return aligned_labels_update


if __name__ == "__main__":
    pass
