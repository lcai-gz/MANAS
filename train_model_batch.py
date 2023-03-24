import time
from torch.utils.data import DataLoader
import torch
import sys
import logging
import os
import utils
import argparse
import glob
import math
import torch.nn as nn
from dataset import Rain800, Muti_image_batch
import pytorch_ssim
from Models import FixedMixedAutoNet

parser = argparse.ArgumentParser("Deraining Model")
parser.add_argument('--data', type=str, default='./datasets/Rain1200/', help='location of data')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batach size')  ##
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate') ##
parser.add_argument('--save', type=str, default='EXP', help='experiment name')  ##
parser.add_argument('--patch_size', type=int, default=64, help='image size')  ##
parser.add_argument('--gpu', type=str, default='1', help='gpu device ids')  ##
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--ckt_path', type=str, default='search-EXP-64-lamda_comp=0.01-lamda_arch=0.01-steps2-half-batch16-ssim-20220412-144322/last_weights.pt', help='checkpoint path of search')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.save = 'eval-{}-{}-{}'.format(args.save, 'step2-half-64-batch32-lamda_comp=0.01-lamda_arch=0.01-ssim', time.strftime("%Y%m%d-%H%M%S"))  ##
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
MSELoss = torch.nn.MSELoss().cuda()
Muti_loss_ssim = utils.Multi_loss_ssim().cuda()

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info("args = %s", args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = FixedMixedAutoNet(ckt_path=args.ckt_path, steps=2)  ##
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    train_queue = Muti_image_batch('./datasets/Rain1200/DID-MDN-training/', args.patch_size)
    valid_data = Rain800(args.data+'DID-MDN-test/', 100, args.patch_size)
    valid_queue = DataLoader(valid_data, batch_size=args.batch_size, pin_memory=True)

    best_psnr = 0
    best_psnr_epoch = 0
    best_ssim = 0
    best_ssim_epoch = 0
    best_loss = float('inf')
    best_loss_epoch = 0

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d/%d lr %e', epoch+1, args.epochs, lr)
        for steps in range(500):  ##
            print('--steps:%d--' % steps)
            input, target = utils.data_transform(train_queue, 32, steps)  ##
            input = input.cuda()
            target = target.cuda()
            model.train()
            optimizer.zero_grad()
            output = model(input)
            loss = Muti_loss_ssim(output, target, 32, epoch)  ##
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        psnr, ssim, loss = infer(valid_queue, model)
        if psnr > best_psnr and not math.isinf(psnr):
            torch.save(model, os.path.join(args.save, 'best_psnr_weights.pt'))
            best_psnr_epoch = epoch + 1
            best_psnr = psnr
        if ssim > best_ssim:
            torch.save(model, os.path.join(args.save, 'best_ssim_weights.pt'))
            best_ssim_epoch = epoch + 1
            best_ssim = ssim
        if loss < best_loss:
            torch.save(model, os.path.join(args.save, 'best_loss_weights.pt'))
            best_loss_epoch = epoch + 1
            best_loss = loss
        scheduler.step()
        logging.info('psnr:%6f ssim:%6f loss:%6f -- best_psnr:%6f best_ssim:%6f best_loss:%6f', psnr, ssim, loss, best_psnr, best_ssim, best_loss)

    logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)', best_loss,
                 best_loss_epoch, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)
    torch.save(model, os.path.join(args.save, 'last_weights.pt'))


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
            s = pytorch_ssim.ssim(torch.clamp(logits, 0, 1), target)
            p = utils.compute_psnr(logits.cpu(), target.cpu())
            n = input.size(0)

            psnr.update(p, n)
            ssim.update(s, n)
            loss.update(l, n)

    return psnr.avg, ssim.avg, loss.avg

if __name__ == '__main__':
    main()

