import torch
import torch.nn as nn


class CustomPool(nn.Module):
    def __init__(self, mode: str, dim: tuple):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CustomPool, self).__init__()
        self.mode = mode
        self.dim = dim


    def forward(self, x: torch.Tensor):
        """
        In the forward function we accept a Tensor of input data and we return
        a tensor with mean pooling applied on `mean_dimension`
        """

        out = None
        if self.mode == 'mean':
            out = x.mean(self.dim)
        elif self.mode == 'max':
            out = torch.amax(x, self.dim)
        elif self.mode == 'min':
            out = torch.amin(x, self.dim)

        return out