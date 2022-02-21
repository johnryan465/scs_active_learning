import torch.nn.functional as F
import torch
import torch.nn as nn


class SharpenedCosineSimilarity(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=0,
        eps=1e-12,
    ):
        super(SharpenedCosineSimilarity, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = int(padding)

        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.xavier_uniform_(w)
        # TODO: this could be initialized as
        # (out_channels, in_channel, kernel_size, kernel_size)
        # right off the bat, but we leave it in this format to retain compat
        # with the einsum implementation
        self.weight = nn.Parameter(
            w, requires_grad=True)

        self.p_scale = 10
        p_init = 2**.5 * self.p_scale
        self.register_parameter("p", nn.Parameter(torch.empty(out_channels)))
        nn.init.constant_(self.p, p_init)

        self.q_scale = 100
        self.register_parameter("q", nn.Parameter(torch.empty(1)))
        nn.init.constant_(self.q, 10)

    def forward(self, x):
        # reshaping for compatibility with the einsum-based implementation
        w = self.weight.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        w_norm = torch.linalg.vector_norm(
            w,
            dim=(1, 2, 3),
            keepdim=True,
        )

        q_sqr = (self.q / self.q_scale) ** 2

        # a small difference: we add eps outside of the norm
        # instead of inside in order to reuse the performant
        # code of torch.linalg.vector_norm
        w_normed = w / ((w_norm + self.eps) + q_sqr)

        x_norm_squared = F.avg_pool2d(
            x ** 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            divisor_override=1, # we actually want sum_pool
        ).sum(dim=1, keepdim=True)

        y_denorm = F.conv2d(
            x,
            w_normed,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )

        y = y_denorm / ((x_norm_squared + self.eps).sqrt() + q_sqr)

        sign = torch.sigmoid(y)

        y = torch.abs(y) + self.eps
        p_sqr = (self.p / self.p_scale) ** 2
        y = y.pow(p_sqr.reshape(1, -1, 1, 1))
        return sign * y