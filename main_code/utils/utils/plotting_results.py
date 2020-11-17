import matplotlib as mpl
import numpy as np
from matplotlib.pyplot import cm

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def plotting_results(inner_matrix, interval, inner_file_name):
    print("inner_matrix.shape: ", inner_matrix.shape)
    plot_point_number = 400
    inner_fig = plt.figure()
    x = np.arange(inner_matrix.shape[1]) * interval

    color = iter(cm.rainbow(np.linspace(0, 1, inner_matrix.shape[0])))

    for i in range(inner_matrix.shape[0]-1):
        c = next(color)
        plt.plot(x[:plot_point_number], inner_matrix[i][:plot_point_number], c=c, linewidth=0.5)
    # plt.title(inner_file_name)

    plt.plot(x[:plot_point_number], inner_matrix[-1][:plot_point_number], "b.", markersize=3)

    plt.xlabel("Epoch")
    plt.ylabel("Number of Significant ROIs")

    plt.legend(("AD-CN", "AD-EMCI", "AD-LMCI", "AD-SMC", "CN-EMCI", "CN-LMCI", "CN-SMC", "EMCI-LMCI", "EMCI-SMC", "LMCI-SMC"), loc='upper right')

    inner_file_name_ext = inner_file_name + ".pdf"
    inner_fig.savefig(inner_file_name_ext)
    plt.close(inner_fig)


if __name__ == "__main__":
    matrix2 = np.random.uniform(0, 1, (10, 20)).astype(np.float32)

    plotting_results(matrix2, "plotting_results_testing")
