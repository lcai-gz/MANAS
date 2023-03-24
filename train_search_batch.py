import time
import torch
import numpy as np
import os
import logging
import math
import torch.nn as nn
import utils
import argparse
import sys
import glob
from architect import Architect
from visdom import Visdom
from Models import MixedSearchSpace
from dataset import Rain800, Muti_image_batch
import pytorch_ssim

parser = argparse.ArgumentParser("Deraining")
parser.add_argument('--data', type=str, default='./datasets/Rain1200/', help='location of dataset')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')   ##
parser.add_argument('--unrolled', type=bool, default=False, help='use one-step unrolled validation loss')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=16, help='data batch size')     ##
parser.add_argument('--patch_size', type=int, default=64, help='size of image')      ##
parser.add_argument('--learning_rate', type=int, default=0.002, help='init learning rate') ##
parser.add_argument('--learning_rate_min', type=int, default=0.0001, help='min learning rate')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--momentum', type=int, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=int, default=3e-4, help='weight decay')
parser.add_argument('--save', type=str, default='EXP-64', help='experiment name')  ##
parser.add_argument('--gpu', type=str, default='1', help='gpu device ids')  ##
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.save = 'search-{}-{}-{}'.format(args.save, 'steps2-batch16-ssim', time.strftime("%Y%m%d-%H%M%S"))  ##
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# Muti_loss = utils.Multi_loss_ssim().cuda()
tvloss = utils.TVLoss().cuda()

# viz = Visdom(env='Train Search')


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info("args = %s", args)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = MixedSearchSpace(steps=2)  ##
    model.cuda()
    architect = Architect(model, args)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    train_queue = Muti_image_batch('./datasets/Rain1200/DID-MDN-training/', args.patch_size)
    valid_data = Rain800(args.data+'DID-MDN-test/', 100, args.patch_size)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, pin_memory=True)

    best_psnr = 0
    best_psnr_epoch = 0
    best_ssim = 0
    best_ssim_epoch = 0
    best_loss = float('inf')
    best_loss_epoch = 0

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d/%d lr %e', epoch+1, args.epochs, lr)
        for steps in range(1000):   ##
            print('--steps:%d--' % steps)
            input, target = utils.data_transform(train_queue, 16, steps)  ##
            input = input.cuda()
            target = target.cuda()
            if (epoch + 1) > 10:         ##
                model.eval()
                input_search_1, target_search_1 = next(iter(valid_queue))
                input_search_2, target_search_2 = next(iter(valid_queue))
                input_search_3, target_search_3 = next(iter(valid_queue))
                input_search = torch.cat((input_search_1, input_search_2, input_search_3), dim=0)
                target_search = torch.cat((target_search_1, target_search_2, target_search_3), dim=0)
                input_search = input_search.cuda()
                target_search = target_search.cuda()
                architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)


            model.train()
            optimizer.zero_grad()
            loss = model.Multi_Loss(input, target, 16, epoch)  ##
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        psnr, ssim, loss = infer(valid_queue, model)
        # p = torch.from_numpy(np.array(psnr))
        # p = p.unsqueeze(0)
        # e = np.array(epoch + 1)
        # e = torch.from_numpy(e)
        # e = e.unsqueeze(0)
        # s = ssim.unsqueeze(0)
        # l = loss.unsqueeze(0)
        # viz.line(X=e, Y=p, win='PSNR', update='append', opts={'title': 'PSNR',
        #                                                       'xlabel': 'epoch',
        #                                                       'ylabel': 'PSNR'})
        # viz.line(X=e, Y=s, win='SSIM', update='append', opts={'title': 'SSIM',
        #                                                       'xlabel': 'epoch',
        #                                                       'ylabel': 'SSIM'})
        # viz.line(X=e, Y=l, win='LOSS', update='append', opts={'title': 'LOSS',
        #                                                       'xlabel': 'epoch',
        #                                                       'ylabel': 'LOSS'})

        if psnr > best_psnr and not math.isinf(psnr):
            torch.save({'state_dict': model.state_dict(),
                        'arch_param': model.arch_parameters()[0],
                        'auto_param': model.auto_parameters()[0],
                        }, os.path.join(args.save, 'best_psnr_weights.pt'))
            best_psnr_epoch = epoch + 1
            best_psnr = psnr
        if ssim > best_ssim:
            torch.save({'state_dict': model.state_dict(),
                        'arch_param': model.arch_parameters()[0],
                        'auto_param': model.auto_parameters()[0],
                        }, os.path.join(args.save, 'best_ssim_weights.pt'))
            best_ssim_epoch = epoch + 1
            best_ssim = ssim
        if loss < best_loss:
            torch.save({'state_dict': model.state_dict(),
                        'arch_param': model.arch_parameters()[0],
                        'auto_param': model.auto_parameters()[0],
                        }, os.path.join(args.save, 'best_loss_weights.pt'))
            best_loss_epoch =epoch + 1
            best_loss = loss

        scheduler.step()
        logging.info('psnr:%6f ssim:%6f loss:%6f -- best_psnr:%6f best_ssim:%6f best_loss:%6f', psnr, ssim, loss, best_psnr, best_ssim, best_loss)
        logging.info('arch_parameters:%s', torch.argmax(model.arch_parameters()[0], dim=1))
        logging.info('auto_parameters:%s', torch.argmax(model.auto_parameters()[0], dim=1))
    logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)', best_loss, best_loss_epoch, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)
    torch.save({'state_dict': model.state_dict(),
                'arch_param': model.arch_parameters(),
                'auto_param': model.auto_parameters(),
                }, os.path.join(args.save, 'last_weights.pt'))
MSELoss = torch.nn.MSELoss().cuda()


def infer(valid_queue, model):
    psnr = utils.AvgrageMeter()
    ssim = utils.AvgrageMeter()
    loss = utils.AvgrageMeter()

    model.eval()

    with torch.no_grad():
        for _, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()
            logits = model(input)

            l = MSELoss(logits, target)
            s = pytorch_ssim.ssim(logits, target)
            p = utils.compute_psnr(logits.cpu(), target.cpu())
            n = input.size(0)

            psnr.update(p, n)
            ssim.update(s, n)
            loss.update(l, n)

    return psnr.avg, ssim.avg, loss.avg


if __name__ == '__main__':
    main()