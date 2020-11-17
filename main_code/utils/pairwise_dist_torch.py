import torch
import torch.nn.functional as F


def pairwise_dist_torch(xyz1, xyz2):
    r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)  # (B, M, N)
    r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)  # (B, M, N)
    mul = torch.matmul(xyz2, xyz1.permute(0, 2, 1))  # (B,M,N)
    dist = F.relu(r_xyz2 - 2 * mul + r_xyz1.permute(0, 2, 1))  # (B,M,N)
    return dist
