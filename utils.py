import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target, weight=None, reduction='mean'):
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)

def generate_masks(data_y, minClassTrain, ratio_val2train):
    n_cls = data_y.max().item() + 1

    n_Alldata = []  # num of train in each class
    for i in range(n_cls):
        data_num = (data_y == i).sum()
        n_Alldata.append(int(data_num.item()))
    # print("In all nodes, class number:", n_Alldata)  # [264, 590, 668, 701, 596, 508]

    num_train_sample = []
    Tmin = min(n_Alldata)
    for i in range(len(n_Alldata)):
        Tnum_sample = int(minClassTrain * n_Alldata[i] / Tmin)
        num_train_sample.append(Tnum_sample)
    print(num_train_sample)  # [20, 44, 50, 53, 45, 38]

    train_mask = torch.zeros(len(data_y), dtype=torch.bool)
    val_mask = torch.zeros(len(data_y), dtype=torch.bool)
    test_mask = torch.zeros(len(data_y), dtype=torch.bool)

    for class_label, num_samples in zip(range(len(num_train_sample)), num_train_sample):
        class_indices = (data_y == class_label).nonzero().view(-1)
        shuffled_indices = torch.randperm(len(class_indices))

        # Divide the sampled indices into train, val, and test sets
        train_indices = class_indices[shuffled_indices[:num_samples]]
        val_indices = class_indices[shuffled_indices[num_samples:num_samples * (ratio_val2train+1)]]
        # test_indices = class_indices[shuffled_indices[num_samples * (ratio_val2train+1):num_samples * int(2*(ratio_val2train+1))]]
        test_indices = class_indices[shuffled_indices[num_samples * (ratio_val2train+1):]]

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
    print(torch.sum(train_mask), torch.sum(val_mask), torch.sum(test_mask))
    return train_mask, val_mask, test_mask


def keep_all_data(edge_index, label, n_data, n_cls, ratio, train_mask):
    """
    just keep all training data
    :param edge_index:
    :param label:
    :param n_data:
    :param n_cls:
    :param ratio:
    :param train_mask:
    :return:
    """
    class_num_list = n_data
    data_train_mask = train_mask

    index_list = torch.arange(len(train_mask))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    train_node_mask = train_mask

    row, col = edge_index[0], edge_index[1]
    row_mask = train_mask[row]
    col_mask = train_mask[col]
    edge_mask = row_mask & col_mask
    # print(torch.sum(train_mask), torch.sum(row_mask), torch.sum(col_mask), torch.sum(edge_mask))  # tensor(250) tensor(414) tensor(410) tensor(51)

    return class_num_list, data_train_mask, idx_info, train_node_mask, edge_mask
