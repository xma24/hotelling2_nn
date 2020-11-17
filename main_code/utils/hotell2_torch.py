import numpy as np
import scipy.stats
import scipy.io as sio
import torch


def hotell2(group_1, group_2):
    mean_1 = np.mean(group_1, axis=1).reshape((-1, 1))
    mean_2 = np.mean(group_2, axis=1).reshape((-1, 1))

    size_n_1 = group_1.shape[1]
    size_n_2 = group_2.shape[1]

    dof_1 = group_1.shape[1] - 1
    dof_2 = group_2.shape[1] - 1
    p = group_1.shape[0]

    cov_1 = np.cov(group_1)
    # print("cov_1: ", cov_1)

    cov_2 = np.cov(group_2)
    # print("cov_2: ", cov_2)

    pooled_cov_12 = (size_n_1 * cov_1 + size_n_2 * cov_2) / (size_n_1 + size_n_2 - 2)
    # print("pooled_cov_12: ", pooled_cov_12)

    D2 = np.dot(np.dot((mean_1 - mean_2).T, np.linalg.pinv(pooled_cov_12)), (mean_1 - mean_2))
    # print("D2: ", D2)

    hotell2_t2 = ((size_n_1 * size_n_2) / (size_n_1 + size_n_2)) * D2

    # print("hotell2_t2: ", hotell2_t2)

    hotell2_t2_update = hotell2_t2 * (size_n_1 + size_n_2 - p - 1) / ((size_n_1 + size_n_2 - 2) * p)
    # print("hotell2_t2_update: ", hotell2_t2_update)

    # f_dist = scipy.stats.f(p, dof_1 + dof_2 - p - 1)
    #
    # hotell2_t2_p_value = 1 - f_dist.cdf(hotell2_t2_update)
    # print("hotell2_t2_p_value: ", hotell2_t2_p_value)

    return hotell2_t2_update


def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def pinv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    # b_inv, _ = torch.gesv(eye, b_mat)
    b_inv, _ = torch.solve(eye, b_mat)
    return b_inv


def hotell2_torch_old(group_1_torch, group_2_torch):
    mean_1 = torch.mean(group_1_torch, dim=1, keepdim=True).view((-1, 1))
    mean_2 = torch.mean(group_2_torch, dim=1, keepdim=True).view((-1, 1))
    # print("mean_1: ", mean_1)
    # print("mean_1.shape: ", mean_1.shape)

    dof_1 = group_1_torch.shape[1] - 1
    dof_2 = group_2_torch.shape[1] - 1
    p = group_1_torch.shape[0]

    cov_1 = cov(group_1_torch)
    # print("cov_1: ", cov_1)

    cov_2 = cov(group_2_torch)
    # print("cov_2: ", cov_2)

    pooled_cov_12 = (dof_1 * cov_1 + dof_2 * cov_2) / (dof_1 + dof_2 - 2)
    # print("pooled_cov_12: ", pooled_cov_12)

    hotell2_t2 = (dof_1 * dof_2) / (dof_1 + dof_2) * torch.mm(
        torch.mm((mean_1 - mean_2).transpose(0, 1), pinv(pooled_cov_12)),
        (mean_1 - mean_2))

    # print("hotell2_t2: ", hotell2_t2)

    hotell2_t2_update = hotell2_t2 * (dof_1 + dof_2 - p - 1) / ((dof_1 + dof_2 - 2) * p)
    # print("hotell2_t2_update: ", hotell2_t2_update)

    # f_dist = scipy.stats.f(p, dof_1 + dof_2 - p - 1)
    #
    # hotell2_t2_p_value = 1 - f_dist.cdf(hotell2_t2_update)
    # print("hotell2_t2_p_value: ", hotell2_t2_p_value)

    # return hotell2_t2_p_value

    return hotell2_t2_update



def hotell2_torch(group_1_torch, group_2_torch):
    mean_1 = torch.mean(group_1_torch, dim=1, keepdim=True).view((-1, 1))
    mean_2 = torch.mean(group_2_torch, dim=1, keepdim=True).view((-1, 1))
    # print("mean_1: ", mean_1)
    # print("mean_1.shape: ", mean_1.shape)

    dof_1 = group_1_torch.shape[1]
    dof_2 = group_2_torch.shape[1]
    p = group_1_torch.shape[0]

    cov_1 = cov(group_1_torch)
    # print("cov_1: ", cov_1)

    cov_2 = cov(group_2_torch)
    # print("cov_2: ", cov_2)

    pooled_cov_12 = (dof_1 * cov_1 + dof_2 * cov_2) / (dof_1 + dof_2 - 2)
    # print("pooled_cov_12: ", pooled_cov_12)

    hotell2_t2 = (dof_1 * dof_2) / (dof_1 + dof_2) * torch.mm(
        torch.mm((mean_1 - mean_2).transpose(0, 1), pinv(pooled_cov_12)),
        (mean_1 - mean_2))

    # print("hotell2_t2: ", hotell2_t2)

    hotell2_t2_update = hotell2_t2 * (dof_1 + dof_2 - p - 1) / ((dof_1 + dof_2 - 2) * p)
    # print("hotell2_t2_update: ", hotell2_t2_update)

    # f_dist = scipy.stats.f(p, dof_1 + dof_2 - p - 1)
    #
    # hotell2_t2_p_value = 1 - f_dist.cdf(hotell2_t2_update)
    # print("hotell2_t2_p_value: ", hotell2_t2_p_value)

    # return hotell2_t2_p_value

    return hotell2_t2_update



if __name__ == "__main__":
    mean_a = [0.1, 0.4]
    cov_a = [[1, 0], [0, 100]]  # diagonal covariance
    group_a = np.random.multivariate_normal(mean_a, cov_a, 1000).T

    mean_b = [0, 0]
    cov_b = [[1, 0], [0, 100]]  # diagonal covariance
    group_b = np.random.multivariate_normal(mean_b, cov_b, 1500).T
    #

    # group_a = np.array([[115, 98, 107, 90, 85, 80, 100, 105, 95, 70, 85, 78, 89, 100, 90, 65, 80]]).reshape(-1, 1)
    # group_b = np.array([[108, 105, 98, 92, 95, 81, 105, 95, 98, 80, 68, 82, 78, 85, 95, 62, 70]]).reshape(-1, 1)


    sio.savemat("group_a.mat", {'group_a': group_a})
    sio.savemat("group_b.mat", {'group_b': group_b})

    real_mean_a = np.mean(group_a, axis=1).reshape((-1, 1))
    # print("mean_a: ", mean_a)
    print("real_mean_a: ", real_mean_a)

    real_mean_b = np.mean(group_b, axis=1).reshape((-1, 1))
    # print("mean_b: ", mean_b)
    print("real_mean_b: ", real_mean_b)

    dof_a = group_a.shape[1] - 1
    print("dof_a: ", dof_a)
    dof_b = group_b.shape[1] - 1
    p = group_a.shape[0]

    real_cov_a_checking = np.dot((group_a - real_mean_a), (group_a - real_mean_a).T) / dof_a
    print("real_cov_a_checking: ", real_cov_a_checking)

    real_cov_a = np.cov(group_a, ddof=dof_a) / dof_a
    print("real_cov_a: ", real_cov_a)

    real_cov_b = np.cov(group_b, ddof=dof_b) / dof_b
    print("real_cov_b: ", real_cov_b)

    pooled_cov = (dof_a * real_cov_a + dof_b * real_cov_b) / (dof_a + dof_b - 2)
    print("pooled_cov: ", pooled_cov)

    T2 = (dof_a * dof_b) / (dof_a + dof_b) * np.dot(np.dot((real_mean_a - real_mean_b).T, np.linalg.inv(pooled_cov)),
                                                    (real_mean_a - real_mean_b))
    print("T2: ", T2)

    T2_update = T2 * (dof_a + dof_b - p - 1) / ((dof_a + dof_b - 2) * p)
    print("T2_update: ", T2_update)

    f_dist = scipy.stats.f(p, dof_a + dof_b - p - 1)

    p_value = 1 - f_dist.cdf(T2_update)
    print("p_value: ", p_value)

    t2_ret = hotell2(group_a, group_b)
    print("t2_ret: ", t2_ret)

    group_a_torch = torch.from_numpy(group_a)
    group_b_torch = torch.from_numpy(group_b)

    t2_torch_ret = hotell2_torch(group_a_torch, group_b_torch)
    print("t2_torch_ret: ", t2_torch_ret)
