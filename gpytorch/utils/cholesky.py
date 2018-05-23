import torch


def batch_potrf(mat):
    potrf_list = []
    for i in range(mat.size(0)):
        potrf_list.append(mat[i].potrf().unsqueeze(0))
    return torch.cat(potrf_list, 0)


def batch_potrs(mat, chol):
    potrs_list = []
    for i in range(mat.size(0)):
        potrs_list.append(torch.potrs(mat[i], chol[i]).unsqueeze(0))
    return torch.cat(potrs_list, 0)
