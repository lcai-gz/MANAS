import torch
import os
import time
import logging
import PIL as Image
import pytorch_ssim
import utils
from dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
from Models import FixedMixedAutoNet
import os
import argparse

parser = argparse.ArgumentParser('Show deraining results ')
parser.add_argument('--path', type=str, default='./datasets/Rain1200/DID-MDN-test/', help='path of test samples')
parser.add_argument('--result_path', type=str, default='./datasets/results/derain/', help='path of result images')
parser.add_argument('--gpu', type=str, default='1', help='gpu device ids')
parser.add_argument('--ckt_path', type=str, default='eval-EXP/best_loss_weights.pt', help='checkpoint path of model')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

model = torch.load(args.ckt_path)
model.cuda()
model.eval()
# samples = Samples(args.path)
samples = Test(args.path)
inputs = DataLoader(dataset=samples, batch_size=1, num_workers=4, pin_memory=True)
transform = transforms.ToPILImage()


def main():
    psnr = utils.AvgrageMeter()
    ssim = utils.AvgrageMeter()
    avg_time = 0
    with torch.no_grad():
        for i, (target, image, image_name) in enumerate(inputs):
            print('num:',i)
            image = image.cuda()
            target = target.cuda()
            start_time = time.time()
            output = model(image)
            avg_time = avg_time + time.time() - start_time
            print('avg_time: %.5f' % (avg_time / (i + 1)))
            s = pytorch_ssim.ssim(torch.clamp(output, 0, 1), target)
            p = utils.compute_psnr(output.cpu(), target.cpu())
            n = image.size(0)
            psnr.update(p, n)
            ssim.update(s.cpu(), n)
            image_save = image[0].cpu()
            image_save = transform(image_save)
            image_save.save(os.path.join('./datasets/results/samples', image_name[0]))
            target_save = target[0].cpu()
            target_save = transform(target_save)
            target_save.save(os.path.join('./datasets/results/gt', image_name[0]))
            output = torch.clamp(output[0], 0, 1)
            result = transform(output.cpu())
            result.save(os.path.join(args.result_path, image_name[0]))
    print('psnr:', psnr.avg, 'ssim:', ssim.avg)
    resultfile = open('result_full.txt', 'w')
    resultfile.write('psnr:')
    resultfile.write(str(float(psnr.avg)))
    resultfile.write('\n')
    resultfile.write('ssim:')
    resultfile.write(str(float(ssim.avg.cpu())))
    resultfile.write('\n')
    resultfile.close()



if __name__ == '__main__':
    main()



