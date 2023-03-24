# coding=utf-8
import torch
import utils
from utils import *
import numpy as np
import torch.nn as nn
from operations import *   
from genotypes import *   
from Networks import *
import os
import logging
import sys


class MixedSearchSpace(nn.Module):
    def __init__(self, in_channels=3, list_channels=[32, 64, 128, 256], list_module=[2, 4, 4, 4], steps=2, auto_module=[2, 3, 4]):
        super(MixedSearchSpace, self).__init__()
        self.steps = steps
        self.auto_module = auto_module
        self.list_channels = list_channels
        self.list_module = list_module

        self.stage1 = self.make_layer(in_channels, list_channels[0], list_module[0])
        # self.transition1 = TransitionModule([list_channels[0]*Bottleneck.expansion], list_channels[:2])
        self.transition1 = TransitionModule([3], list_channels[:2])
        self.auto1 = self.make_auto(steps, auto_module[0], list_channels)

        self.stage2 = self.make_stage(list_channels[:2], list_module[1])   ## construct a cell
        self.transition2 = TransitionModule(list_channels[:2], list_channels[:3])
        self.auto2 = self.make_auto(steps, auto_module[1], list_channels)

        self.stage3 = self.make_stage(list_channels[:3], list_channels[2])
        self.transition3 = TransitionModule(list_channels[:3], list_channels[:4])
        self.auto3 = self.make_auto(steps, auto_module[2], list_channels)

        self.stage4 = self.make_stage(list_channels[:4], list_module[3])

        self.final_fuse = FusionModule(list_channels, muti_scale_output=False)
        # self.final_fuse = FusionModule(list_channels[:3], muti_scale_output=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(list_channels[0], in_channels, 1, 1, bias=False)
        )

        self.auto_one_block = sum(1 for i in range(self.steps) for n in range(i + 1))

        self.init_architecture()

        self.Mutil_loss = utils.Multi_loss_ssim().cuda()
        # self.Mutil_loss = utils.Multi_loss_ssim_improved().cuda()
        # self.Mutil_loss = utils.One2One_loss_ssim().cuda()
        self.MSELoss = torch.nn.MSELoss().cuda()

    # construct auto_attention module, steps=2, num_module=2 or 3 or 4
    def make_auto(self, steps, num_module, channel):
        auto_module = nn.ModuleList()
        for i in range(num_module):
            auto_module.append(Attention(steps, channel[i]))
        return auto_module  ## auto_module = [Attention, ...]

    def make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes*Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*Bottleneck.expansion),
            )
        layers.append(Bottleneck(inplanes, planes, stride, downsample))
        inplanes = planes*Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))
        return nn.Sequential(*layers)

    ## construct a cell, the MixModule may be the fusion module or parallel module
    def make_stage(self, num_inchannels, num_module):  ## num_module = 4
        modules = nn.ModuleList()
        for _ in range(num_module):
            modules.append(MixModule(num_inchannels))
        return modules                         ## modules = [MixModule, MixModule, MixModule, MixModule]

    def new(self):
        model_new = SearchSpace().cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new


    def init_architecture(self):
        self.auto_param = torch.randn(self.auto_one_block * sum(self.auto_module[:]), len(PRIMITIVES), dtype=torch.float).cuda()*1e-3
        self.arch_param = torch.randn(sum(self.list_module[1:]), 2, dtype=torch.float).cuda()*1e-3
        self.auto_param.requires_grad = True
        self.arch_param.requires_grad = True

    def arch_parameters(self):
        return [self.arch_param]

    def auto_parameters(self):
        return [self.auto_param]

    
    def init_weight(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)   ## two parallel modules
        x_list = self.transition1([x])  ## transition module
        arch_idx = 0
        auto_idx = 0

        # construct a cell
        for i in range(self.list_module[1]):  # self.list_module[1] = 4 corresponding to Ni=4 in the paper
            x_list = self.stage2[i](x_list, self.arch_param[arch_idx+i])
        # construct a auto_attention module, 
        for i in range(self.auto_module[0]):  # self.auto_module[0] = 2 
            x_list[i] = self.auto1[i](x_list[i], self.auto_param[(i + auto_idx) * self.auto_one_block:(i + auto_idx + 1) * self.auto_one_block])
        # print('first layer shape:{}'.format(np.array(x_list)[1].shape))

        ## transition module, 
        x_list = self.transition2(x_list)
        auto_idx = auto_idx + self.auto_module[0]
        arch_idx = arch_idx + self.list_module[1]
        # construct a cell
        for i in range(self.list_module[2]):  # self.list_module[2] = 4 corresponding to Ni=4 in the paper
            x_list = self.stage3[i](x_list, self.arch_param[arch_idx+i])
        # construct auto_attention module
        for i in range(self.auto_module[1]):  # self.auto_module[1] = 3 
            x_list[i] = self.auto2[i](x_list[i], self.auto_param[(i + auto_idx) * self.auto_one_block:(i + auto_idx + 1) * self.auto_one_block])
        # print('second layer shape:{}'.format(np.array(x_list)[2].shape))

        ## transition module
        x_list = self.transition3(x_list)
        auto_idx = auto_idx + self.auto_module[1]
        arch_idx = arch_idx + self.list_module[2]
        # construct a cell
        for i in range(self.list_module[3]):   # self.list_module[3] = 4 corresponding to Ni=4 in the paper
            x_list = self.stage4[i](x_list, self.arch_param[arch_idx + i])
        # construct auto_attention module
        for i in range(self.auto_module[2]):   # self.auto_module[2] = 4 
            x_list[i] = self.auto3[i](x_list[i], self.auto_param[(i + auto_idx) * self.auto_one_block:(i + auto_idx + 1) * self.auto_one_block])
        # print('third layer shape:{}'.format(np.array(x_list)[3].shape))

        ## fusion module
        x = self.final_fuse(x_list)[0]
        x = self.final_conv(x)
        return x

    
    def Multi_Loss(self, input, target, num, epoch):
        logits = self(input)
        loss = self.Mutil_loss(logits, target, num, epoch)
        # arch_weights = torch.nn.functional.softmax(self.arch_param, dim=1)
        # auto_weights = torch.nn.functional.softmax(self.auto_param, dim=1)
        # arch_regular_loss = -arch_weights*torch.log10(arch_weights)-(1-arch_weights)*torch.log10(1-arch_weights)
        # auto_regular_loss = -auto_weights*torch.log10(auto_weights)-(1-auto_weights)*torch.log10(1-auto_weights)

        # return loss + arch_regular_loss.mean()*0.01 + auto_regular_loss.mean()*0.01
        return loss

    def loss(self, input, target, num, epoch):
        logits = self(input)
        # mse_loss = self.MSELoss(logits, target)
        loss = self.Mutil_loss(logits, target, num, epoch)
        arch_weights = torch.nn.functional.softmax(self.arch_param, dim=1)
        auto_weights = torch.nn.functional.softmax(self.auto_param, dim=1)

        # regular_loss
        arch_regular_loss = -arch_weights * torch.log10(arch_weights) - (1 - arch_weights) * torch.log10(
            1 - arch_weights)
        auto_regular_loss = -auto_weights * torch.log10(auto_weights) - (1 - auto_weights) * torch.log10(
            1 - auto_weights)
        # latency loss: computing the average complexity (params num) of modules by running `python search_space.py + python operations`
        # latency_loss = torch.mean(arch_weights[:,0]*0.7)+torch.mean(arch_weights[:,1]*0.3) \
        #                 + torch.mean(auto_weights[:,4]*0.5)+torch.mean(auto_weights[:,5]*0.2)+torch.mean(auto_weights[:,6]*0.1)+torch.mean(auto_weights[:,7]*0.1)+torch.mean(auto_weights[:,8]*0.1)

        return loss + arch_regular_loss.mean()*0.01 + auto_regular_loss.mean()*0.01 #+ latency_loss*0.1
        # return loss + arch_regular_loss.mean() * 0.01 + auto_regular_loss.mean() * 0.01 + latency_loss*0.01


class MixedAutoNet(nn.Module):
    def __init__(self, ckt_path, in_channels=3, list_channels=[32, 64, 128, 256], list_module=[2, 4, 4, 4], steps=2, auto_module=[2, 3, 4]):
        super(MixedAutoNet, self).__init__()
        self.list_channels = list_channels
        self.auto_module = auto_module
        self.steps = steps
        self.list_modules = list_module
        arch = torch.load(ckt_path)['arch_param'][0]
        auto = torch.load(ckt_path)['auto_param'][0]
        softmax = nn.Softmax(dim=1)
        arch_soft = softmax(arch)
        auto_soft = softmax(auto)

        self.auto_one_block = sum(1 for i in range(self.steps) for n in range(i + 1))

        arch_hard = torch.argmax(arch_soft, dim=1)
        auto_hard = torch.argmax(auto_soft, dim=1)

        self.stage1 = self.make_layer(in_channels, list_channels[0], list_module[0])

        self.transition1 = TransitionModule([list_channels[0]*Bottleneck.expansion], list_channels[:2])
        idx = 0
        self.stage2 = self.make_stage(arch_hard[idx:list_module[1]], list_channels[:2], list_module[1])
        self.auto1 = self.make_auto(steps, auto_module[0], list_channels)

        self.transition2 = TransitionModule(list_channels[:2], list_channels[:3])
        idx = idx + list_module[1]
        self.stage3 = self.make_stage(arch_hard[idx:idx+list_module[2]], list_channels[:3], list_module[2])
        self.auto2 = self.make_auto(steps, auto_module[1], list_channels)

        self.transition3 = TransitionModule(list_channels[:3], list_channels[:4])
        idx = idx + list_module[2]
        self.stage4 = self.make_stage(arch_hard[idx:idx+list_module[3]], list_channels[:4], list_module[3])
        self.auto3 = self.make_auto(steps, auto_module[2], list_channels)

        self.final_fuse = FusionModule(list_channels, muti_scale_output=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(list_channels[0], in_channels, 1, 1, bias=False),
        )

        # self.init_weights()
        self.init_auto_param(auto_hard)

    def make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*Bottleneck.expansion),
            )
        layers.append(Bottleneck(inplanes, planes, stride, downsample))
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))
        return nn.Sequential(*layers)

    def init_auto_param(self, auto_hard):
        num = len(auto_hard)
        self.auto_param = torch.randn(self.auto_one_block * sum(self.auto_module[:]), len(PRIMITIVES), dtype=torch.float).cuda()*0
        for i in range(num):
            self.auto_param[i][auto_hard[i]] = 1

    def make_auto(self, steps, num_module, channel):
        auto_module = nn.ModuleList()
        for i in range(num_module):
            auto_module.append(Attention(steps, channel[i]))
        return auto_module

    def make_stage(self, arch, num_inchannels, num_module):
        module = []
        for i in range(num_module):
            if arch[i] == 0:
                module.append(HorizontalModule(num_inchannels))
            elif arch[i] == 1:
                module.append(FusionModule(num_inchannels))
        return nn.Sequential(*module)

    def init_weights(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x_list = self.transition1([x])
        x_list = self.stage2(x_list)
        auto_idx = 0
        for i in range(self.auto_module[0]):
            x_list[i] = self.auto1[i](x_list[i], self.auto_param[(i + auto_idx) * self.auto_one_block:(i + auto_idx + 1) * self.auto_one_block])

        auto_idx = auto_idx + self.auto_module[0]
        x_list = self.transition2(x_list)
        x_list = self.stage3(x_list)
        for i in range(self.auto_module[1]):
            x_list[i] = self.auto2[i](x_list[i], self.auto_param[(i + auto_idx) * self.auto_one_block:(i + auto_idx + 1) * self.auto_one_block])

        auto_idx = auto_idx + self.auto_module[1]
        x_list = self.transition3(x_list)
        x_list = self.stage4(x_list)
        for i in range(self.auto_module[2]):
            x_list[i] = self.auto3[i](x_list[i], self.auto_param[(i + auto_idx) * self.auto_one_block:(i + auto_idx + 1) * self.auto_one_block])

        x = self.final_fuse(x_list)[0]
        x = self.final_conv(x)
        return x


class FixedMixedAutoNet(nn.Module):
    def __init__(self, ckt_path, in_channels=3, list_channels=[32, 64, 128, 256], list_module=[2, 4, 4, 4], steps=2, auto_module=[2, 3, 4]):
        super(FixedMixedAutoNet, self).__init__()
        self.list_channels = list_channels
        self.auto_module = auto_module
        self.steps = steps
        self.list_modules = list_module
        arch = torch.load(ckt_path)['arch_param']  
        auto = torch.load(ckt_path)['auto_param']  
        softmax = nn.Softmax(dim=1)
        arch_soft = softmax(arch)
        auto_soft = softmax(auto)

        self.auto_one_block = sum(1 for i in range(self.steps) for n in range(i + 1))

        arch_hard = torch.argmax(arch_soft, dim=1) 
        auto_hard = torch.argmax(auto_soft, dim=1) 

        self.stage1 = self.make_layer(in_channels, list_channels[0], list_module[0])

        # self.transition1 = TransitionModule([list_channels[0]*Bottleneck.expansion], list_channels[:2])
        self.transition1 = TransitionModule([3], list_channels[:2])
        idx = 0
        idx_auto = 0
        self.stage2 = self.make_stage(arch_hard[idx:list_module[1]], list_channels[:2], list_module[1])
        self.auto1 = self.make_auto(steps, auto_module[0], list_channels, auto_hard[idx_auto:idx_auto+self.auto_one_block])

        self.transition2 = TransitionModule(list_channels[:2], list_channels[:3])
        idx = idx + list_module[1]
        idx_auto = idx_auto + self.auto_one_block
        self.stage3 = self.make_stage(arch_hard[idx:idx+list_module[2]], list_channels[:3], list_module[2])
        self.auto2 = self.make_auto(steps, auto_module[1], list_channels, auto_hard[idx_auto:idx_auto+self.auto_one_block])

        self.transition3 = TransitionModule(list_channels[:3], list_channels[:4])
        idx = idx + list_module[2]
        idx_auto = idx_auto + self.auto_one_block
        self.stage4 = self.make_stage(arch_hard[idx:idx+list_module[3]], list_channels[:4], list_module[3])
        self.auto3 = self.make_auto(steps, auto_module[2], list_channels, auto_hard[idx_auto:idx_auto+self.auto_one_block])

        self.final_fuse = FusionModule(list_channels, muti_scale_output=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(list_channels[0], in_channels, 1, 1, bias=False),
        )

        # self.init_weights()

    def make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*Bottleneck.expansion),
            )
        layers.append(Bottleneck(inplanes, planes, stride, downsample))  
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))   
        return nn.Sequential(*layers)

    def make_auto(self, steps, num_module, channel, weights):
        auto_module = nn.ModuleList()
        for i in range(num_module):
            auto_module.append(FixedAttention(channel[i], steps, weights))
        return auto_module

    def make_stage(self, arch, num_inchannels, num_module):
        module = []
        for i in range(num_module):
            if arch[i] == 0:
                module.append(HorizontalModule(num_inchannels))
            elif arch[i] == 1:
                module.append(FusionModule(num_inchannels))
        return nn.Sequential(*module)

    def init_weights(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # parallel module
        # x = self.stage1(x)
        # transition module
        x_list = self.transition1([x])
        # construct a cell
        x_list = self.stage2(x_list)
        auto_idx = 0
        # construct auto_attention
        for i in range(self.auto_module[0]):
            x_list[i] = self.auto1[i](x_list[i])

        auto_idx = auto_idx + self.auto_module[0]
        # transition module
        x_list = self.transition2(x_list)

        # construct a cell
        x_list = self.stage3(x_list)
        # construct auto_attention
        for i in range(self.auto_module[1]):
            x_list[i] = self.auto2[i](x_list[i])

        auto_idx = auto_idx + self.auto_module[1]   ##
        # transition module
        x_list = self.transition3(x_list)           ##
        # construct a cell
        x_list = self.stage4(x_list)                ##
        # construct auto_attention
        for i in range(self.auto_module[2]):        ##
            x_list[i] = self.auto3[i](x_list[i])    ##

        x = self.final_fuse(x_list)[0]
        x = self.final_conv(x)
        return x


class SingleMixedSearchSpace(nn.Module):
    def __init__(self, in_channels=3, list_channels=[32, 64, 128, 256], list_module=[2, 4, 4, 4], steps=2):
        super(SingleMixedSearchSpace, self).__init__()
        self.steps = steps
        self.list_channels = list_channels
        self.list_module = list_module
        self.stage1 = self.make_layer(in_channels, list_channels[0], list_module[0])
        self.transition1 = TransitionModule([list_channels[0]*Bottleneck.expansion], list_channels[:2])
        self.auto1 = self.make_auto(steps, list_channels[0])

        self.stage2 = self.make_stage(list_channels[:2], list_module[1])
        self.transition2 = TransitionModule(list_channels[:2], list_channels[:3])
        self.auto2 = self.make_auto(steps, list_channels[0])

        self.stage3 = self.make_stage(list_channels[:3], list_channels[2])
        self.transition3 = TransitionModule(list_channels[:3], list_channels[:4])
        self.auto3 = self.make_auto(steps, list_channels[0])

        self.stage4 = self.make_stage(list_channels[:4], list_module[3])

        self.final_fuse = FusionModule(list_channels, muti_scale_output=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(list_channels[0], in_channels, 1, 1, bias=False)
        )

        self.auto_one_block = sum(1 for i in range(self.steps) for n in range(i + 1))

        self.init_architecture()

        self.MSEloss = torch.nn.MSELoss().cuda()

    def make_auto(self, steps, channel):
        auto_module = nn.ModuleList()
        auto_module.append(Attention(steps, channel))
        return auto_module

    def make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes*Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*Bottleneck.expansion),
            )
        layers.append(Bottleneck(inplanes, planes, stride, downsample))
        inplanes = planes*Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))
        return nn.Sequential(*layers)

    def make_stage(self, num_inchannels, num_module):
        modules = nn.ModuleList()
        for _ in range(num_module):
            modules.append(MixModule(num_inchannels))
        return modules

    def new(self):
        model_new = SearchSpace().cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def init_architecture(self):
        self.auto_param = torch.randn(self.auto_one_block * 3, len(PRIMITIVES), dtype=torch.float).cuda()*1e-3
        self.arch_param = torch.randn(sum(self.list_module[1:]), 2, dtype=torch.float).cuda()*1e-3
        self.auto_param.requires_grad = True
        self.arch_param.requires_grad = True

    def arch_parameters(self):
        return [self.arch_param]

    def auto_parameters(self):
        return [self.auto_param]

    def init_weight(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)

        x_list = self.transition1([x])
        arch_idx = 0

        for i in range(self.list_module[1]):
            x_list = self.stage2[i](x_list, self.arch_param[arch_idx+i])
        x_list[0] = self.auto1(x_list[0], self.auto_param[0:self.auto_one_block])

        x_list = self.transition2(x_list)
        arch_idx = arch_idx + self.list_module[1]
        for i in range(self.list_module[2]):
            x_list = self.stage3[i](x_list, self.arch_param[arch_idx+i])
        x_list[0] = self.auto2(x_list[0], self.auto_param[self.auto_one_block:2 * self.auto_one_block])

        x_list = self.transition3(x_list)
        arch_idx = arch_idx + self.list_module[2]
        for i in range(self.list_module[3]):
            x_list = self.stage4[i](x_list, self.arch_param[arch_idx + i])
        x_list[0] = self.auto3(x_list[0], self.auto_param[2 * self.auto_one_block:3 * self.auto_one_block])

        x = self.final_fuse(x_list)[0]
        x = self.final_conv(x)
        return x

    def loss(self, input, target):
        logits = self(input)
        mse_loss = self.MSEloss(logits, target)
        arch_weights = torch.nn.functional.softmax(self.arch_param, dim=1)
        regular_loss = -arch_weights*torch.log10(arch_weights)-(1-arch_weights)*torch.log10(1-arch_weights)

        return mse_loss + regular_loss.mean()*0.01



if __name__ == '__main__':
    model = FixedMixedAutoNet()
