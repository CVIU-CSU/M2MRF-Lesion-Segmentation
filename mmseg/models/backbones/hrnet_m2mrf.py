import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.models.utils.m2mrf import M2MRF
from mmseg.ops import Upsample, resize
from mmseg.utils import get_root_logger
from .resnet import BasicBlock, Bottleneck
from ..builder import BACKBONES


class HRModule_M2MRF(nn.Module):
    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),

                 m2mrf_encode_channels_rate=None,
                 m2mrf_fc_channels_rate=None,
                 m2mrf_patch_size=(8, 8),
                 m2mrf_groups=1,
                 m2mrf_version=0,
                 m2mrf_cascade_down=False,
                 m2mrf_onestep_down=False,
                 m2mrf_cascade_up=False,
                 m2mrf_onestep_up=False
                 ):
        self.m2mrf_encode_channels_rate = m2mrf_encode_channels_rate
        self.m2mrf_fc_channels_rate = m2mrf_fc_channels_rate
        self.m2mrf_patch_size = m2mrf_patch_size
        self.m2mrf_groups = m2mrf_groups
        self.m2mrf_version = m2mrf_version

        self.m2mrf_cascade_down = m2mrf_cascade_down
        self.m2mrf_onestep_down = m2mrf_onestep_down
        self.m2mrf_cascade_up = m2mrf_cascade_up
        self.m2mrf_onestep_up = m2mrf_onestep_up

        super(HRModule_M2MRF, self).__init__()
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        """Check branches configuration."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_BLOCKS(' \
                        f'{len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_CHANNELS(' \
                        f'{len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS(' \
                        f'{len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """Build one branch."""
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, num_channels[branch_index] *
                                 block.expansion)[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Build multiple branch."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    if self.m2mrf_onestep_up:
                        fuse_layer.append(
                            nn.Sequential(
                                build_conv_layer(
                                    self.conv_cfg,
                                    in_channels[j],
                                    in_channels[i],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False),
                                build_norm_layer(self.norm_cfg, in_channels[i])[1],
                                M2MRF(scale_factor=2 ** (j - i),
                                      in_channels=in_channels[i],
                                      out_channels=in_channels[i],
                                      patch_size=self.m2mrf_patch_size[1],
                                      encode_channels_rate=self.m2mrf_encode_channels_rate,
                                      fc_channels_rate=self.m2mrf_fc_channels_rate,
                                      version=self.m2mrf_version,
                                      groups=self.m2mrf_groups)
                            ))
                    elif self.m2mrf_cascade_up:
                        fuse_layer_item = nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1])
                        for it in range(j - i):
                            fuse_layer_item.add_module(
                                f'sample_{it}',
                                M2MRF(scale_factor=2,
                                      in_channels=in_channels[i],
                                      out_channels=in_channels[i],
                                      patch_size=self.m2mrf_patch_size[1],
                                      encode_channels_rate=self.m2mrf_encode_channels_rate,
                                      fc_channels_rate=self.m2mrf_fc_channels_rate,
                                      version=self.m2mrf_version,
                                      groups=self.m2mrf_groups)
                            )
                        fuse_layer.append(fuse_layer_item)
                    else:
                        fuse_layer.append(
                            nn.Sequential(
                                build_conv_layer(
                                    self.conv_cfg,
                                    in_channels[j],
                                    in_channels[i],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False),
                                build_norm_layer(self.norm_cfg, in_channels[i])[1],
                                # we set align_corners=False for HRNet
                                Upsample(
                                    scale_factor=2 ** (j - i),
                                    mode='bilinear',
                                    align_corners=False)
                            ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    if self.m2mrf_onestep_down:
                        conv_downsamples.append(
                            nn.Sequential(
                                M2MRF(scale_factor=0.5 ** (i - j),
                                      in_channels=in_channels[j],
                                      out_channels=in_channels[i],
                                      patch_size=self.m2mrf_patch_size[0],
                                      encode_channels_rate=self.m2mrf_encode_channels_rate,
                                      fc_channels_rate=self.m2mrf_fc_channels_rate,
                                      version=self.m2mrf_version,
                                      groups=self.m2mrf_groups),
                                build_norm_layer(self.norm_cfg, in_channels[i])[1]
                            )
                        )
                    elif self.m2mrf_cascade_down:
                        for k in range(i - j):
                            if k == i - j - 1:
                                conv_downsamples.append(
                                    nn.Sequential(
                                        M2MRF(scale_factor=0.5,
                                              in_channels=in_channels[j],
                                              out_channels=in_channels[i],
                                              patch_size=self.m2mrf_patch_size[0],
                                              encode_channels_rate=self.m2mrf_encode_channels_rate,
                                              fc_channels_rate=self.m2mrf_fc_channels_rate,
                                              version=self.m2mrf_version,
                                              groups=self.m2mrf_groups),
                                        build_norm_layer(self.norm_cfg, in_channels[i])[1]))
                            else:
                                conv_downsamples.append(
                                    nn.Sequential(
                                        M2MRF(
                                            scale_factor=0.5,
                                            in_channels=in_channels[j],
                                            out_channels=in_channels[j],
                                            patch_size=self.m2mrf_patch_size[0],
                                            encode_channels_rate=self.m2mrf_encode_channels_rate,
                                            fc_channels_rate=self.m2mrf_fc_channels_rate,
                                            version=self.m2mrf_version,
                                            groups=self.m2mrf_groups),
                                        build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                        nn.ReLU(inplace=False)))
                    else:
                        for k in range(i - j):
                            if k == i - j - 1:
                                conv_downsamples.append(
                                    nn.Sequential(
                                        build_conv_layer(
                                            self.conv_cfg,
                                            in_channels[j],
                                            in_channels[i],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=False),
                                        build_norm_layer(self.norm_cfg,
                                                         in_channels[i])[1]))
                            else:
                                conv_downsamples.append(
                                    nn.Sequential(
                                        build_conv_layer(
                                            self.conv_cfg,
                                            in_channels[j],
                                            in_channels[j],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=False),
                                        build_norm_layer(self.norm_cfg,
                                                         in_channels[j])[1],
                                        nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    y = y + resize(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode='bilinear',
                        align_corners=False)
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


@BACKBONES.register_module()
class HRNet_M2MRF(nn.Module):
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,

                 m2mrf_patch_size=(8, 8),
                 m2mrf_encode_channels_rate=4,
                 m2mrf_fc_channels_rate=64,
                 m2mrf_version=0,
                 m2mrf_groups=1,
                 m2mrf_cascade_down_list=(False, False, False),
                 m2mrf_onestep_down_list=(False, False, False),
                 m2mrf_cascade_up_list=(False, False, False),
                 m2mrf_onestep_up_list=(False, False, False)
                 ):
        super(HRNet_M2MRF, self).__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        # M2MRF config:
        self.m2mrf_patch_size = m2mrf_patch_size
        self.m2mrf_encode_channels_rate = m2mrf_encode_channels_rate
        self.m2mrf_fc_channels_rate = m2mrf_fc_channels_rate
        self.m2mrf_version = m2mrf_version
        self.m2mrf_groups = m2mrf_groups
        self.m2mrf_cascade_down_list = m2mrf_cascade_down_list
        self.m2mrf_onestep_down_list = m2mrf_onestep_down_list
        self.m2mrf_cascade_up_list = m2mrf_cascade_up_list
        self.m2mrf_onestep_up_list = m2mrf_onestep_up_list

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels], num_channels, stage_index=0)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels, stage_index=0)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels, stage_index=1)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels, stage_index=1)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels, stage_index=2)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, stage_index=2)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer, stage_index=0):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels

                    if self.m2mrf_cascade_down_list[stage_index] or self.m2mrf_onestep_down_list[stage_index]:
                        conv_downsamples.append(
                            nn.Sequential(
                                M2MRF(scale_factor=0.5,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      patch_size=self.m2mrf_patch_size[0],
                                      encode_channels_rate=self.m2mrf_encode_channels_rate,
                                      fc_channels_rate=self.m2mrf_fc_channels_rate,
                                      version=self.m2mrf_version,
                                      groups=self.m2mrf_groups),
                                build_norm_layer(self.norm_cfg, out_channels)[1],
                                nn.ReLU(inplace=True)))
                    else:
                        conv_downsamples.append(
                            nn.Sequential(
                                build_conv_layer(
                                    self.conv_cfg,
                                    in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False),
                                build_norm_layer(self.norm_cfg, out_channels)[1],
                                nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """Make each layer."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])
        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True, stage_index=0):
        """Make each stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]
        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule_M2MRF(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    m2mrf_patch_size=self.m2mrf_patch_size,
                    m2mrf_encode_channels_rate=self.m2mrf_encode_channels_rate,
                    m2mrf_fc_channels_rate=self.m2mrf_fc_channels_rate,
                    m2mrf_version=self.m2mrf_version,
                    m2mrf_groups=self.m2mrf_groups,
                    m2mrf_cascade_down=self.m2mrf_cascade_down_list[stage_index],
                    m2mrf_onestep_down=self.m2mrf_onestep_down_list[stage_index],
                    m2mrf_cascade_up=self.m2mrf_cascade_up_list[stage_index],
                    m2mrf_onestep_up=self.m2mrf_onestep_up_list[stage_index]
                ))

        return nn.Sequential(*hr_modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode whill keeping the normalization
        layer freezed."""
        super(HRNet_M2MRF, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class HRNet_M2MRF_A(HRNet_M2MRF):
    def __init__(self, extra, **kwargs):
        super().__init__(extra,
                         m2mrf_onestep_down_list=(True, True, True),
                         m2mrf_onestep_up_list=(True, True, True),
                         **kwargs)


@BACKBONES.register_module()
class HRNet_M2MRF_B(HRNet_M2MRF):
    def __init__(self, extra, **kwargs):
        super().__init__(extra,
                         m2mrf_onestep_down_list=(True, True, True),
                         m2mrf_cascade_up_list=(True, True, True),
                         **kwargs)


@BACKBONES.register_module()
class HRNet_M2MRF_C(HRNet_M2MRF):
    def __init__(self, extra, **kwargs):
        super().__init__(extra,
                         m2mrf_cascade_down_list=(True, True, True),
                         m2mrf_onestep_up_list=(True, True, True),
                         **kwargs)


@BACKBONES.register_module()
class HRNet_M2MRF_D(HRNet_M2MRF):
    def __init__(self, extra, **kwargs):
        super().__init__(extra,
                         m2mrf_cascade_down_list=(True, True, True),
                         m2mrf_cascade_up_list=(True, True, True),
                         **kwargs)
