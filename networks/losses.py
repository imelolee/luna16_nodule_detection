import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BCEWithWeights(nn.Module):

    def __init__(self) -> None:
        super(BCEWithWeights, self).__init__()
      

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size, num_class = input.size(0), input.size(1)

        # Weighted cross entropy for imbalance class distribution
        weight = torch.ones(num_class).cuda()
        total = len(target)
        for i in range(num_class):
            num_pos = float((target == i).sum())
            num_pos = max(num_pos, 1)
            weight[i] = total / num_pos

        weight = weight / weight.sum()
        rcnn_cls_loss = F.cross_entropy(input, target, weight=weight, size_average=True)
   
        return rcnn_cls_loss
