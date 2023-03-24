import torch
import os
import PIL as Image
import pytorch_ssim
import utils
from dataset_lcai import *
from torch.utils.data import DataLoader
from torchvision import transforms
from Models import FixedMixedAutoNet
import os
import argparse

parser = argparse.ArgumentParser('Show deraining results ')
parser.add_argument('--path', type=str, default='./datasets/Real_Internet/', help='path of test samples')
parser.add_argument('--result_path', type=str, default='./datasets/results/', help='path of result images')
parser.add_argument('--gpu', type=str, default='2', help='gpu device ids')
parser.add_argument('--ckt_path', type=str, default='eval-EXP-step2-64-batch32-ssim-20211209-095925/best_loss_weights.pt', help='checkpoint path of model')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

model = torch.load(args.ckt_path)
model.cuda()
model.eval()
samples = Test_real(args.path)
inputs = DataLoader(dataset=samples, batch_size=1, num_workers=4, pin_memory=True)
transform = transforms.ToPILImage()

def main():
    with torch.no_grad():
        for i, (image, image_name) in enumerate(inputs):
            print(image_name)
            image = image.cuda()
            output = model(image)
            image_save = image[0].cpu()
            image_save = transform(image_save)
            image_save.save(os.path.join('./datasets/samples/', image_name[0]))
            output = torch.clamp(output[0], 0, 1)
            result = transform(output.cpu())
            result.save(os.path.join(args.result_path, image_name[0]))

if __name__ == '__main__':
    main()
