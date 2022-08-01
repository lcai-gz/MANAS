import os
import shutil
import torch.nn as nn
import torch
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
from pytorch_ssim import ssim
import torchvision.models as models


MSELoss = torch.nn.MSELoss()


def compute_psnr(pred, tar):
    assert pred.shape == tar.shape
    pred = pred.numpy()
    tar = tar.numpy()
    pred = pred.transpose(0, 2, 3, 1)
    tar = tar.transpose(0, 2, 3, 1)
    psnr = 0
    for i in range(pred.shape[0]):
        psnr += PSNR(tar[i], pred[i])
    return psnr/pred.shape[0]


class AvgrageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if 'auxiliary' not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.bar')
        shutil.copyfile(filename, best_filename)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


tvloss = TVLoss()


class Multi_loss(nn.Module):
    def __init__(self):
        super(Multi_loss, self).__init__()

    def forward(self, output, target, batch_size, epoch):
        loss = 0
        if epoch < 5:
            for i in range(batch_size):
                loss += MSELoss(output[i * 3], target[i]) + MSELoss(output[i * 3 + 1], target[i]) + MSELoss(output[i * 3 + 2], target[i]) + \
                        0.0001 * tvloss(output[i * 3:i * 3 + 3])
        else:
            for i in range(batch_size):
                loss += MSELoss(output[i * 3], target[i]) + MSELoss(output[i * 3 + 1], target[i]) + MSELoss(output[i * 3 + 2], target[i]) + \
                        0.0001 * tvloss(output[i * 3:i * 3 + 3]) + \
                        MSELoss(output[i * 3], output[i * 3 + 1]) + MSELoss(output[i * 3], output[i * 3 + 2]) + MSELoss(output[i * 3 + 1], output[i * 3 + 2])
        return loss / batch_size


def data_transform(data, batch_size, steps):
    count = steps*batch_size  ## 记录已经遍历过的图像
    input, target = data[count]  ## 指向下一个batch_size的开始
    for i in range(batch_size-1):
        input_next, target_next = data[count+i+1]  ## 加载还没遍历的数据
        input = torch.cat((input, input_next), dim=0)
        target = torch.cat((target, target_next), dim=0)
    return input, target  ## 返回一个batch


class Multi_loss_ssim(nn.Module):
    def __init__(self):
        super(Multi_loss_ssim, self).__init__()

    def forward(self, output, target, batch_size, epoch):
        loss = 0
        if epoch < 5:
            for i in range(batch_size):
                loss += MSELoss(output[i * 3], target[i]) + MSELoss(output[i * 3 + 1], target[i]) + MSELoss(output[i * 3 + 2], target[i]) + \
                        0.0001 * tvloss(output[i * 3:i * 3 + 3]) + \
                        (1-ssim(output[i * 3].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 1].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 2].unsqueeze(dim=0), target[i].unsqueeze(dim=0)))
        else:
            for i in range(batch_size):
                loss += MSELoss(output[i * 3], target[i]) + MSELoss(output[i * 3 + 1], target[i]) + MSELoss(output[i * 3 + 2], target[i]) + \
                        0.0001 * tvloss(output[i * 3:i * 3 + 3]) + \
                        MSELoss(output[i * 3], output[i * 3 + 1]) + MSELoss(output[i * 3], output[i * 3 + 2]) + MSELoss(output[i * 3 + 1], output[i * 3 + 2]) + \
                        (1-ssim(output[i * 3].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 1].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 2].unsqueeze(dim=0), target[i].unsqueeze(dim=0)))

        return loss / batch_size


class Multi_loss_ssim_improved(nn.Module):
    def __init__(self):
        super(Multi_loss_ssim_improved, self).__init__()

    def forward(self, output, target, batch_size, epoch):
        loss = 0
        if epoch < 5:   ##
            for i in range(batch_size):
                loss += MSELoss(output[i * 3], target[i]) + MSELoss(output[i * 3 + 1], target[i]) + MSELoss(output[i * 3 + 2], target[i]) + \
                        0.0001 * tvloss(output[i * 3:i * 3 + 3]) + \
                        (1-ssim(output[i * 3].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 1].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 2].unsqueeze(dim=0), target[i].unsqueeze(dim=0)))
        # elif epoch >= 5 and epoch <= 20:  ##
        #     for i in range(batch_size):
        #         loss += MSELoss(output[i * 3], target[i]) + MSELoss(output[i * 3 + 1], target[i]) + MSELoss(output[i * 3 + 2], target[i]) + \
        #                 0.0001 * tvloss(output[i * 3:i * 3 + 3]) + \
        #                 0.1 * MSELoss(output[i * 3], output[i * 3 + 1]) + 0.1 * MSELoss(output[i * 3], output[i * 3 + 2]) + 0.1 * MSELoss(output[i * 3 + 1], output[i * 3 + 2]) + \
        #                 (1-ssim(output[i * 3].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 1].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 2].unsqueeze(dim=0), target[i].unsqueeze(dim=0)))
        else:
            for i in range(batch_size):
                loss += MSELoss(output[i * 3], target[i]) + MSELoss(output[i * 3 + 1], target[i]) + MSELoss(output[i * 3 + 2], target[i]) + \
                        0.0001 * tvloss(output[i * 3:i * 3 + 3]) + \
                        MSELoss(output[i * 3], output[i * 3 + 1]) + MSELoss(output[i * 3], output[i * 3 + 2]) + MSELoss(output[i * 3 + 1], output[i * 3 + 2]) + \
                        (1-ssim(output[i * 3].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 1].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 2].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + \
                        (1-ssim(output[i * 3].unsqueeze(dim=0), output[i * 3 + 1].unsqueeze(dim=0))) + (1-ssim(output[i * 3].unsqueeze(dim=0), output[i * 3 + 2].unsqueeze(dim=0))) + (1-ssim(output[i * 3 + 1].unsqueeze(dim=0), output[i * 3 + 2].unsqueeze(dim=0)))


        return loss / batch_size

class One2One_loss_ssim(nn.Module):
    def __init__(self):
        super(One2One_loss_ssim, self).__init__()

    def forward(self, output, target, batch_size, epoch):
        loss = 0

        for i in range(batch_size):
            loss += MSELoss(output[i * 3], target[i]) + MSELoss(output[i * 3 + 1], target[i]) + MSELoss(output[i * 3 + 2], target[i]) + \
                    0.0001 * tvloss(output[i * 3:i * 3 + 3]) + \
                    (1 - ssim(output[i * 3].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1 - ssim(output[i * 3 + 1].unsqueeze(dim=0), target[i].unsqueeze(dim=0))) + (1 - ssim(output[i * 3 + 2].unsqueeze(dim=0), target[i].unsqueeze(dim=0)))

        return loss / batch_size

if __name__ == '__main__':
    # model = models.vgg16()
    # weights = count_parameters_in_MB(model)
    # print(weights)
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.size())
    pred = torch.randn((4, 3, 64, 64)).cuda().cpu()
    tar = torch.randn((4, 3, 64, 64)).cuda().cpu()
    # psnr = compute_psnr(pred, tar)
    # print(psnr)

    ssim = ssim(pred, tar)
    print(ssim)