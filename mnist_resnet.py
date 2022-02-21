from due.layers import spectral_norm_conv, spectral_norm_fc
from due.layers.spectral_batchnorm import SpectralBatchNorm2d

from params import NNParams
import torch.nn as nn
import torch.nn.functional as F
from scs.abspool import MaxAbsPool2d

from scs.scs import SharpenedCosineSimilarity



class MNISTResNet(nn.Module):
    """
    Feature Extractor
    """
    def __init__(self, nn_params: NNParams):
        super().__init__()
        input_channels = 1
        input_size = 28
        num_classes = nn_params.num_classes
        self.features_size = 1024

        spectral_normalization = nn_params.spectral_normalization
        coeff = nn_params.coeff
        n_power_iterations = nn_params.n_power_iterations
        batchnorm_momentum = nn_params.batchnorm_momentum

        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(
                    num_features, coeff, momentum=batchnorm_momentum
                )
            else:
                bn = nn.BatchNorm2d(num_features, momentum=batchnorm_momentum)

            return bn

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            if kernel_size == 3:
                padding = 1
            elif kernel_size == 5:
                padding = 2
            else:
                padding = 0

            conv = SharpenedCosineSimilarity(in_c, out_c, kernel_size, stride, padding)

            if not spectral_normalization:
                return conv

            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                input_dim = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations
                )

            return wrapped_conv

        self.conv1 = wrapped_conv(input_size, input_channels, 32, 5, 1)
        self.bn1 = wrapped_bn(32)

        # 28x28 -> 14x14
        self.conv2 = wrapped_conv(input_size, 32, 64, 5, 1)
        self.shortcut2 = wrapped_conv(input_size, 32, 64, 1, 1)
        self.bn2 = wrapped_bn(64)

        # 14x14 -> 7x7
        self.conv3 = wrapped_conv(input_size, 64, 64, 3, 1)
        self.shortcut3 = wrapped_conv(input_size, 64, 64, 1, 1)
        self.bn3 = wrapped_bn(64)

        self.pool = MaxAbsPool2d(kernel_size=7, stride=7, ceil_mode=True)

        self.num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(self.features_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x) + x
        out1 = F.relu(self.bn1(x))
        out2 = F.relu(self.bn2(self.conv2(out1) + self.shortcut2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2) + self.shortcut3(out2)))

        out = self.pool(out3)
        out = out.flatten(1)
        # print(out.shape)

        if self.num_classes is not None:
            out = self.linear(out)
        return out