import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init


class M2MRF_Module(nn.Module):
    def __init__(self,
                 scale_factor,
                 encode_channels,
                 fc_channels,
                 size,
                 groups=1):
        super(M2MRF_Module, self).__init__()

        self.scale_factor = scale_factor

        self.encode_channels = encode_channels
        self.fc_channels = fc_channels

        self.size = size
        self.groups = groups

        self.unfold_params = dict(kernel_size=self.size,
                                  dilation=1, padding=0, stride=self.size)
        self.fold_params = dict(kernel_size=int(self.size * self.scale_factor),
                                dilation=1, padding=0, stride=int(self.size * scale_factor))
        self.sample_fc = nn.Conv1d(
            self.size * self.size * self.encode_channels,
            self.fc_channels,
            groups=self.groups,
            kernel_size=1)
        self.sample_fc1 = nn.Conv1d(
            self.fc_channels,
            int(self.size * self.size * self.scale_factor * self.scale_factor * self.encode_channels),
            groups=self.groups,
            kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        n, c, h, w = x.shape
        x = nn.Unfold(**self.unfold_params)(x)
        x = x.view(n, c * self.size * self.size, -1)
        x = self.sample_fc(x)
        x = self.sample_fc1(x)
        x = nn.Fold((int(h * self.scale_factor), int(w * self.scale_factor)), **self.fold_params)(x)
        return x


class M2MRF(nn.Module):
    def __init__(self,
                 scale_factor,
                 in_channels,
                 out_channels,
                 patch_size=8,
                 encode_channels_rate=4,
                 fc_channels_rate=64,
                 version=0,
                 groups=1):
        super(M2MRF, self).__init__()

        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.size = patch_size
        self.patch_size = patch_size
        self.version = version

        if encode_channels_rate is not None:
            self.encode_channels = int(in_channels / encode_channels_rate)
        else:
            raise NotImplementedError

        if fc_channels_rate is not None:
            self.fc_channels = int(self.size * self.size * self.encode_channels / fc_channels_rate)
        else:
            self.fc_channels = self.encode_channels

        self.sample_encode_conv = nn.Conv2d(self.in_channels, self.encode_channels, kernel_size=1, stride=1, padding=0)
        self.sample = M2MRF_Module(self.scale_factor, self.encode_channels, self.fc_channels,
                                   size=self.size, groups=self.groups)
        self.sample_decode_conv = nn.Conv2d(self.encode_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                xavier_init(m, distribution='uniform')

    def pad_input(self, x):
        b, c, h, w = x.shape
        fold_h, fold_w = h, w

        if h % self.patch_size > 0:
            fold_h = h + (self.patch_size - h % self.patch_size)
        if w % self.patch_size > 0:
            fold_w = w + (self.patch_size - w % self.patch_size)
        x = F.pad(x, [0, fold_w - w, 0, fold_h - h], mode='constant', value=0)

        out_h = max(int(h * self.scale_factor), 1)
        out_w = max(int(w * self.scale_factor), 1)

        return x, (out_h, out_w)

    def forward(self, x):
        x, out_shape = self.pad_input(x)

        x = self.sample_encode_conv(x)
        x = self.sample(x)
        x = self.sample_decode_conv(x)

        x = x[:, :, :out_shape[0], :out_shape[1]]
        return x
