from .unc_data_loader import unc_data_loader
import numpy as np


def unc_data_loader_2_groups(args):
    group_data_ret = unc_data_loader(args)
    # print("group_data_ret.shape: ", group_data_ret)

    group_AD = group_data_ret[0]
    group_LMCI = group_data_ret[3]

    group_CN = group_data_ret[1]
    group_EMCI = group_data_ret[2]

    group_AD_update = np.append(group_AD, group_LMCI, axis=0)
    group_CN_update = np.append(group_CN, group_EMCI, axis=0)

    group_data_2_groups = {}
    group_data_2_groups[0] = group_AD_update
    group_data_2_groups[1] = group_CN_update

    return group_data_2_groups




