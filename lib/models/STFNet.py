from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo

from lib.models.DCNv2.dcn_v2 import DCN
# from DCNv2.dcn_v2 import DCN


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = out[:,:,0:residual.shape[2],0:residual.shape[3]]

        out += residual

        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(5):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
        # self.conv = nn.Conv2d(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1)


    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            # print(len(channels))
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            #layers[i] = project(layers[i])
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out

"""mask""" 
class genMask(nn.Module):
    def __init__(self, in_channel):    
        super(genMask, self).__init__()

        self.conv1x1_down_channel = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1, stride=1, padding=0)
        # 该卷积核被不同时间帧共享  为减少参数 
        self.conv1 = nn.Conv2d(in_channel // 8, in_channel // 8, kernel_size=3, stride=1, padding=1, groups=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel // 2, in_channel, kernel_size=3, stride=1, padding=1, groups=1),
            nn.Sigmoid()
        )

    def forward(self, feat_3d):
        # (N, C//8, H, W)
        feats_down_channel = [0]*5
        hms = []
        for i in range(5):
            xi = feat_3d[:, :, i]
            feats_down_channel[i] = self.conv1x1_down_channel(xi)
            if i > 0:
                hms.append(self.conv1(feats_down_channel[i]) - feats_down_channel[i - 1])

        # (N, C//2, H, W)
        hms = torch.cat(hms, dim=1)

        return self.conv2(hms)

"""3d提取部分修改"""
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Convblock(nn.Module):
    def __init__(self, in_channel, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, 7, 1, 3, groups=in_channel)
        # self.dwconv = DeformConv(in_channel, in_channel)
        self.norm = nn.LayerNorm(in_channel)
        self.pwconv1 = nn.Linear(in_channel, 4*in_channel)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4*in_channel, in_channel)
        self.gama = nn.Parameter(layer_scale_init_value * torch.ones((in_channel)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1) # [b,c,h,w] -> [b,h,w,c]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gama is not None:
            x = self.gama * x
        x = x.permute(0,3,1,2) # [b,h,w,c] -> [b,c,h,w]
        x = residual + self.drop_path(x)
        return x
    
class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm3d(out_channels,eps=0.001, momentum=0.1, affine=True)# value found in tensorflow # default pytorch value
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class baseNet3D(nn.Module):

    def __init__(self, channel_input, channels):
        super(baseNet3D, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv3d(in_channels=channel_input, out_channels=channels[0], kernel_size=(1, 1, 5), stride=(1, 1, 1), padding=(0, 0, 2)),
            BasicConv3d(in_channels=channels[0], out_channels=channels[0], kernel_size=(1, 5, 1), stride=(1, 1, 1), padding=(0, 2, 0)),
            BasicConv3d(in_channels=channels[0], out_channels=channels[0], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )

        self.branch2 = nn.Sequential(
            BasicConv3d(in_channels=channels[0], out_channels=channels[1], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(in_channels=channels[1], out_channels=channels[1], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(in_channels=channels[1], out_channels=channels[1], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channels=channels[1], out_channels=channels[2], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(in_channels=channels[2], out_channels=channels[2], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(in_channels=channels[2], out_channels=channels[2], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        
        self.maxpool = nn.MaxPool3d([1, 2, 2])
        self.maxpool1 = nn.MaxPool3d([3, 1, 1])
        
        self.conv_block1 = Convblock(channels[0])
        self.conv_block2 = Convblock(channels[1])
        self.conv_block3 = Convblock(channels[2])
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.mix1 = nn.Conv2d(channels[2]+channels[1], channels[1], 3, 1, 1)
        self.mix2 = nn.Conv2d(channels[1]+channels[0], channels[0], 3, 1, 1)
        
        self.hm1 = genMask(channels[0])
        self.hm2 = genMask(channels[1])
        self.hm3 = genMask(channels[2])


    def forward(self, x):
        layers = []
        hms = []
        # x = x.unsqueeze(1)
        x = self.branch1(x)
        hm = self.hm1(x)
        hms.append(hm)
        layers.append(self.conv_block1(hm*self.maxpool1(x).squeeze(2)))
        x = self.maxpool(x)

        x = self.branch2(x)
        hm = self.hm2(x)
        hms.append(hm)
        layers.append(self.conv_block2(hm*self.maxpool1(x).squeeze(2)))
        x = self.maxpool(x)

        x = self.branch3(x)
        hm = self.hm3(x)
        hms.append(hm)
        layers.append(self.conv_block3(hm*self.maxpool1(x).squeeze(2)))
        layers[1] = self.mix1(torch.cat([self.upsample(layers[2]), layers[1]], dim=1))
        layers[0] = self.mix2(torch.cat([self.upsample(layers[1]), layers[0]], dim=1))
            
        return layers, hms

'''修改后的STFF'''

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "sigmoid":
        module = nn.Sigmoid()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class CBA(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class STFF(nn.Module):
    # channel1大图通道    channel2 小图通道   
    def __init__(self, channel, out_channel):    
        super(STFF, self).__init__()

        self.conv1 = nn.Sequential(
            CBA(channel, channel//2, 1, 1, act='relu'),
            CBA(channel//2, channel, 1, 1)
        )
        self.conv2 = nn.Sequential(
            CBA(channel, channel//2, 1, 1, act='relu'),
            CBA(channel//2, channel, 1, 1),
        )

        self.df_conv1 = DeformConv(channel * 2, channel)
        self.df_conv2 = DeformConv(channel * 2, channel)
        self.df_conv3 = DeformConv(channel * 2, out_channel)
        
    def forward(self, feat1, feat2):

        feat1_conved = self.conv1(feat1)
        feat2_conved = self.conv2(feat2)
        feat1_fused = self.df_conv1(torch.cat([feat1_conved, feat2], 1))
        feat2_fused = self.df_conv2(torch.cat([feat1, feat2_conved], 1))
        
        return self.df_conv3(torch.cat([feat1_fused,feat2_fused],1))
    
class ISTFF(nn.Module):
    def __init__(self, channel, m_channel, out_channel):
        super().__init__()
        self.mix12 = STFF(channel, channel)
        self.conv3 = nn.Sequential(
            nn.Conv2d(m_channel, channel, 3, 2, 1),
            CBA(channel, channel//2, 1, 1, act='relu'),
            CBA(channel//2, channel, 1, 1, act='sigmoid'),
            DeformConv(channel, channel)
        )
        self.mix123 = DeformConv(channel*2, out_channel)
        
    def forward(self, feat1, feat2, mix_feat):
        mix12 = self.mix12(feat1, feat2)
        mix_feat = self.conv3(mix_feat)
             
        return self.mix123(torch.cat([mix12, mix_feat], dim=1))

class DLASeg(nn.Module):
    def __init__(self, heads, final_kernel,
                  head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        self.first_level = 0  # int(np.log2(down_ratio))
        self.last_level = 3  # last_level
        self.base = dla34(pretrained=True)
        # channels = [32, 64, 128, 256, 512, 1024]#self.base.channels
        channels = [16, 32, 64,128,256]  # self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        channel_input = 3
        channels3d = [16, 32, 64]
        self.base3d = baseNet3D(channel_input, channels3d)

        channelsFuse = [16, 16, 32, 64]

        self.first_level = 1  # int(np.log2(down_ratio))
        self.last_level = 4  # last_level

        if out_channel == 0:
            out_channel = channelsFuse[self.first_level]

        self.ida_up = IDAUp(out_channel, channelsFuse[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])
        for i in range(3):
            if i == 0:
                self.__setattr__("FF_%d" % (i), STFF(channelsFuse[i+1], channelsFuse[i+1]))
            else: self.__setattr__("FF_%d" % (i), ISTFF(channelsFuse[i+1], channelsFuse[i], channelsFuse[i+1]))
            

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channelsFuse[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channelsFuse[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):

        xx = x[:, :, 0, :, :]
        layersspatial = self.base(xx)

        """[[b, 16, 512, 512],[b, 32, 256, 256],[b,64, 128, 128]]"""
        layers1 = self.dla_up(layersspatial)

        layerstemporal, hms = self.base3d(x)

        layers = []

        for ii in range(3):
            """[[b, 32, 512, 512],[b, 64, 256, 256],[b, 128, 128, 128]]"""
            if ii == 0:
                layers.append(self.__getattr__("FF_%d" % ii)(layers1[ii]*hms[ii], layerstemporal[ii]))
            else: 
                layers.append(self.__getattr__("FF_%d" % ii)(layers1[ii]*hms[ii], layerstemporal[ii], layers[ii-1]))

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(layers[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]


def STFNet(heads, head_conv=128):
    model = DLASeg(heads,final_kernel=1,
                 head_conv=head_conv)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    # start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

if __name__ == '__main__':
    net = STFNet({'hm': 1, 'wh': 2, 'reg': 2}, 256).cuda()
    input  = torch.randn(1,3,5,512,512)
    output = net(input.cuda())



