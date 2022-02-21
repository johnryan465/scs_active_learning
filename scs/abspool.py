from functools import partial
import torch
import torch.nn as nn

class AbsPool(nn.Module):
    def __init__(self, pooling_module=None, *args, **kwargs):
        super(AbsPool, self).__init__()
        self.pooling_layer = pooling_module(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_pool = self.pooling_layer(x)
        neg_pool = self.pooling_layer(-x)
        abs_pool = torch.where(pos_pool >= neg_pool, pos_pool, -neg_pool)
        return abs_pool


MaxAbsPool1d = partial(AbsPool, nn.MaxPool1d)
MaxAbsPool2d = partial(AbsPool, nn.MaxPool2d)
MaxAbsPool3d = partial(AbsPool, nn.MaxPool3d)