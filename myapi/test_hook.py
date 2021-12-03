import torch
import torch.nn.functional as F


def logit_activation(seg_logit, use_sigmoid=False):
    """
    :param seg_logit: feature map without activation function
    :param use_sigmoid: whether to use sigmoid
    :return: activation feature map
    """

    if not use_sigmoid:
        output = F.softmax(seg_logit, dim=1)
    else:
        output = torch.sigmoid(seg_logit)

    return output


if __name__ == '__main__':
    pass
