# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
from operations import *  ## 所有注意力模块
from genotypes import *   ## 所有注意力模块的名字
import os
import logging
import sys


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        if self.downsample is not None:
            residual = self.downsample(x)
        y = y + residual
        y = self.relu(y)
        return y


# Auto_attention network

# corresponding to channel attension in the paper
class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()   ## b: batch_size, c: channel number
        y = self.avg_pool(x).view(b, c)  ## view(b, c)作用与resize一样，把tensor resize到(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

## 所有注意力路径的组合, op表示某个注意力路径，ops所有路径的集合，w表示路径的权重
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()  ## 定义一个列表来存储op
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(
                    op,
                    nn.BatchNorm2d(C, affine=False),
                )
            self.ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))

## 最后组合起来的整个注意力模块, steps=2 or 4
class Attention(nn.Module):
    def __init__(self, steps, C):
        super(Attention, self).__init__()
        self.steps = steps
        self.C = C                    ## 特征通道数
        self.ops = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.C_in = self.C // self.steps   ## 特征按通道进行切割
        self.C_out = self.C
        for i in range(self.steps):   ## 确定多少条路径, op表示一条路径的所有注意力，ops表示所有路径的注意力模块
            for j in range(1+i):
                stride = 1
                op = MixedOp(self.C_in, stride)
                self.ops.append(op)
        ## 对应论文U0到Uk特征进行组合回去，对应论文公式（3）
        self.channel_back = nn.Sequential(
            nn.Conv2d(self.C_in*(self.steps+1), self.C, kernel_size=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.C, self.C, kernel_size=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(self.C),
        )
        self.se = SE(self.C_in, reduction=4)
        self.se2 = SE(self.C_in*self.steps, reduction=16)

    def forward(self, s0, weights):  ## s0表示特征输入， weights表示上传到ops的权重
        C = s0.shape[1]
        length = C // self.steps
        spx = torch.split(s0, length, 1)  ## 按通道进行切分
        spx_sum = sum(spx)           ## 对个分割的特征进行累加，注意这与论文（3）描述的不一致
        spx_sum = self.se(spx_sum)    ## 把累加的特征通过SE注意机制模块
        offset = 0
        states = [spx[0]]
        for i in range(self.steps):  ## 对应论文公式（2）
            states[0] = spx[i]  ## 对F 进行覆盖
            s = sum(self.ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)   ## states = [Fk-1, U0,U1..Uk-1]

        node_concat = torch.cat(states[-self.steps:], dim=1)  ## states[-self.steps:] = [U0...Uk-1]
        node_concat = torch.cat((node_concat,spx_sum), dim=1)  ## [U0...Uk-1, spx_sum]

        attention_out = self.channel_back(node_concat) + s0
        attention_out = self.se2(attention_out)

        return attention_out


class AttentionBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, steps):
        super(AttentionBasicBlock, self).__init__()
        self.steps = steps
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride
        self.attention = Attention(self.steps, planes)

    def forward(self, x, weights):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out, weights)
        out = out + residual
        out = self.relu(out)
        return out


# multi-scale neural architecture search
class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.downsample is not None:
            residual = self.downsample(x)
        y = y + residual
        y = self.relu(y)
        return y

## Parallel Module
class HorizontalModule(nn.Module):
    def __init__(self, num_inchannels, num_blocks=None):
        super(HorizontalModule, self).__init__()
        self.num_branches = len(num_inchannels)
        assert self.num_branches > 1
        if num_blocks is None:
            num_blocks = [1 for _ in range(self.num_branches)]  ## num_blocks = [1, 1, 1]
        else:
            assert self.num_branches == len(num_blocks)
        self.branches = nn.ModuleList()
        for i in range(self.num_branches):
            layers = []
            for _ in range(num_blocks[i]):
                layers.append(ResidualBlock(num_inchannels[i], num_inchannels[i]))  ## [ResidualBlock, ResidualBlock, ResidualBlock]
            self.branches.append(nn.Sequential(*layers))  ## [ResidualBlock, ResidualBlock, ResidualBlock]

    def forward(self, x):
        for i in range(self.num_branches):  ## self.num_branches = 3
            x[i] = self.branches[i](x[i])
        return x

## Fusion Module
class FusionModule(nn.Module):
    def __init__(self, num_inchannels, muti_scale_output=True):
        super(FusionModule, self).__init__()
        self.num_branches = len(num_inchannels)
        self.muti_scale_output = muti_scale_output
        assert self.num_branches > 1

        self.relu = nn.ReLU(inplace=True)
        self.fuse_layers = nn.ModuleList()
        for i in range(self.num_branches if self.muti_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    # upsample
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    #downsample
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i-j-1:  ## 从1x 到0.25x的情况
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)
                            ))
                        else:  ## 从1x 到0.25x 和 1x 到0.5x 的情况
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            self.fuse_layers.append(nn.ModuleList(fuse_layer))

    def forward(self, x):
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])  ## 第一条分支输入导致的各分支的输出
            for j in range(1, self.num_branches):
                if i == j:         ## 判断输入分支与输出分支是否在同个分支上
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class TransitionModule(nn.Module):
    def __init__(self, num_channels_pre, num_channels_cur):
        super(TransitionModule, self).__init__()
        self.num_branches_pre = len(num_channels_pre)
        self.num_branches_cur = len(num_channels_cur)
        self.transition_layers = nn.ModuleList()
        for i in range(self.num_branches_cur):
            if i <self.num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    self.transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.transition_layers.append(None)
            else:
                conv3x3 = []
                for j in range(i+1-self.num_branches_pre):
                    inchannels = num_channels_pre[-1]
                    outchannels = num_channels_cur[i] if j == i-self.num_branches_pre else inchannels
                    conv3x3.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                self.transition_layers.append(nn.Sequential(*conv3x3))

    def forward(self, x):
        x_list = []
        for i in range(self.num_branches_cur):
            if self.transition_layers[i] is not None:
                if i < self.num_branches_pre:
                    x_list.append(self.transition_layers[i](x[i]))
                else:
                    x_list.append(self.transition_layers[i](x[-1]))
            else:
                x_list.append(x[i])
        return x_list

## Mix Module, corresponding to Figure 2 (d) in the paper
class MixModule(nn.Module):
    def __init__(self, num_inchannels):
        super(MixModule, self).__init__()
        self.horizontal = HorizontalModule(num_inchannels)
        self.fuse = FusionModule(num_inchannels)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, weight):
        h_x = self.horizontal(x)
        f_x = self.fuse(x)

        weight = self.softmax(weight)

        x_list = []
        for i in range(len(x)):
            x_list.append(weight[0]*h_x[i]+weight[1]*f_x[i])

        return x_list


class FixedMixOp(nn.Module):
    def __init__(self, C, stride, weights):
        super(FixedMixOp, self).__init__()
        self.ops = nn.ModuleList()
        self.weights = weights
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(
                    op,
                    nn.BatchNorm2d(C, affine=False),
                )
            self.ops.append(op)

    def forward(self, x):
        return self.ops[self.weights](x)


class FixedAttention(nn.Module):
    def __init__(self, C, steps, weights):
        super(FixedAttention, self).__init__()
        self.steps = steps
        self.C = C
        self.weights = weights
        self.ops = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.C_in = self.C // self.steps
        self.C_out = self.C
        self.width = 4
        idx = 0
        for i in range(self.steps):
            for j in range(1+i):
                stride = 1
                op = FixedMixOp(self.C_in, stride, weights[idx])   ## 固定搜索到的op
                idx = idx + 1
                self.ops.append(op)
        self.channel_back = nn.Sequential(
            nn.Conv2d(self.C_in*(self.steps+1), self.C, kernel_size=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.C, self.C, kernel_size=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(self.C),
        )
        self.se = SE(self.C_in, reduction=4)
        self.se2 = SE(self.C_in*self.steps, reduction=16)

    def forward(self, s0):  ## s0表示特征输入
        C = s0.shape[1]
        length = C // self.steps  ## 对特征组分割成通道数为length的特征组
        spx = torch.split(s0, length, 1)  ## 按通道进行切分
        spx_sum = sum(spx)        ## 对个分割的特征进行累加，注意这与论文（3）描述的不一致
        spx_sum = self.se(spx_sum)    ## 把累加的特征通过SE注意机制模块
        offset = 0
        states = [spx[0]]
        for i in range(self.steps):  ## 对应论文公式（2）
            states[0] = spx[i]       ## 对F 进行覆盖
            s = sum(self.ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)   ## states = [Fk-1, U0,U1..Uk-1]

        node_concat = torch.cat(states[-self.steps:], dim=1)  ## states[-self.steps:] = [U0...Uk-1]
        node_concat = torch.cat((node_concat,spx_sum), dim=1)  ## [U0...Uk-1, spx_sum]

        attention_out = self.channel_back(node_concat) + s0
        attention_out = self.se2(attention_out)

        return attention_out



if __name__=='__main__':
    '''
    Computing the average complexity (params) of modules.
    '''
    from utils import count_parameters_in_MB
    import numpy as np
    list_channels=[32, 64, 128, 256]
    run_time_ratio = []
    params = np.zeros((2,3))
    for i in range(1, len(list_channels)):
        H = HorizontalModule(list_channels[:i+1]).cuda()
        F = FusionModule(list_channels[:i+1]).cuda()
        params[0,i-1] = count_parameters_in_MB(H) # parallel module
        params[1,i-1] = count_parameters_in_MB(F) # fusion module
    comp = np.sum(params, axis=1)/np.sum(params)
    print('params ratio: %.1f %.1f' % (comp[0], comp[1]))

















