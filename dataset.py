import torch
import random
import os
import numpy as np
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class Rain800(Dataset):
    def __init__(self, pth, length, patch_size):   
        super(Rain800, self).__init__()
        self.rain_image = []
        self.clear_image = []
        self.path_size = patch_size
        self.length = length
        self.num = 0
        for i in glob.glob(pth + '*.jpg'):
            img = np.array(Image.open(i))
            assert img.shape[1] % 2 == 0
            self.rain_image.append(img[:,:img.shape[1]//2,:])
            self.clear_image.append(img[:,img.shape[1]//2:,:])
            self.num += 1

        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num

    def data_augmentation(self, image, mode):
        if mode == 0:
            out = image
        elif mode == 1:
            out = np.flipud(image)
        elif mode == 2:
            out = np.rot90(image)
        elif mode == 3:
            out = np.rot90(image)
            out = np.flipud(out)
        elif mode == 4:
            out = np.rot90(image, k=2)
        elif mode == 5:
            out = np.rot90(image, k=2)
            out = np.flipud(out)
        elif mode == 6:
            out = np.rot90(image, k=3)
        elif mode == 7:
            out = np.rot90(image, k=3)
            out = np.flipud(out)
        else:
            raise Exception('Invalid choice of image transformation')
        return out

    def random_augmentation(self, *args):
        out = []
        flag = random.randint(1,7)
        for data in args:
            out.append(self.data_augmentation(data, flag).copy())
        return out

    def __getitem__(self, index):
        idx = random.randint(0, self.num-1)
        H = self.clear_image[idx].shape[0]
        W = self.clear_image[idx].shape[1]
        int_H = random.randint(0, H - self.path_size)
        int_W = random.randint(0, W - self.path_size)
        clear_image = self.clear_image[idx][int_H:int_H+self.path_size,int_W:int_W+self.path_size,:]  ## randomly cropped
        rain_image = self.rain_image[idx][int_H:int_H+self.path_size,int_W:int_W+self.path_size,:]
        # aug_list = self.random_augmentation(rain_image, clear_image)
        return self.transform(rain_image), self.transform(clear_image)


class Rain100(Dataset):
    def __init__(self, pth, name, length, patch_size):
        super(Rain100, self).__init__()
        self.root_dir = os.path.join(pth, name)
        self.clear_path = os.path.join(self.root_dir, 'norain')
        self.rain_path = os.path.join(self.root_dir, 'rain/X2')
        self.clear_images = os.listdir(self.clear_path)
        self.rain_images = os.listdir(self.rain_path)
        self.patch_size = patch_size
        self.num = len(self.clear_images)
        self.length = length
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num

    def data_augmentation(self, image, mode):
        if mode == 0:
            out = image
        elif mode == 1:
            out = np.flipud(image)
        elif mode == 2:
            out = np.rot90(image)
        elif mode == 3:
            out = np.rot90(image)
            out = np.flipud(out)
        elif mode == 4:
            out = np.rot90(image, k=2)
        elif mode == 5:
            out = np.rot90(image, k=2)
            out = np.flipud(out)
        elif mode == 6:
            out = np.rot90(image, k=3)
        elif mode == 7:
            out = np.rot90(image, k=3)
            out = np.flipud(out)
        else:
            raise Exception('Invalid choice of image transformation')
        return out

    def random_augmentation(self, *args):
        out = []
        flag = random.randint(1,7)
        for data in args:
            out.append(self.data_augmentation(data, flag).copy())
        return out

    def __getitem__(self, item):
        clear_image_name = self.clear_images[item]
        clear_image_path = os.path.join(self.clear_path, clear_image_name)
        (filename, extension) = os.path.splitext(clear_image_name)
        rain_image_path = os.path.join(self.rain_path, filename + 'x2' + extension)
        clear_image = np.array(Image.open(clear_image_path))
        rain_image = np.array(Image.open(rain_image_path))
        H = clear_image.shape[0]
        W = clear_image.shape[1]
        int_H = random.randint(0, H - self.patch_size)
        int_W = random.randint(0, W - self.patch_size)
        rain_patch = rain_image[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size,:]
        clear_patch = clear_image[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size,:]
        out_list = self.random_augmentation(rain_patch, clear_patch)
        return self.transform(out_list[0]), self.transform(out_list[1])


class Test(Dataset):
    def __init__(self, path):
        super(Test, self).__init__()
        self.path = path
        self.pairs = os.listdir(self.path)
        self.num = len(self.pairs)
        self.transfrom = transforms.ToTensor()

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        pair_name = self.pairs[item]
        pair_path = os.path.join(self.path, pair_name)
        pairs = np.array(Image.open(pair_path).convert('RGB'))
        H = pairs.shape[0] // 8 * 8
        W = pairs.shape[1] // 2
        patch = W // 8 * 8
        input = pairs[:H, W:W+patch, :]
        target = pairs[:H, :patch, :]
        input = self.transfrom(input)
        target = self.transfrom(target)
        # return target, input, pair_name
        return input, target, pair_name

class PreNet_test(Dataset):
    def __init__(self, path, length):
        super(PreNet_test, self).__init__()
        self.path = path
        self.ground_truth_path = os.path.join(self.path, 'gt')
        self.derained_image_path = os.path.join(self.path, 'recovery_images')
        # self.ground_truth = []
        # self.derained_image = []
        # self.get_image()
        self.pairs = os.listdir(self.ground_truth_path)
        self.num = length
        self.transfrom = transforms.ToTensor()

    def __len__(self):
        return self.num

    # def get_image(self):
    #     for i in os.listdir(self.ground_truth_path):
    #         gt_image_path = os.path.join(self.ground_truth_path, i)
    #         derained_image_path = os.path.join(self.derained_image_path, i)
    #         gt_image = np.array(Image.open(gt_image_path).convert('RGB'))
    #         derained_image = np.array(Image.open(derained_image_path).convert('RGB'))
    #         self.ground_truth.append(gt_image)
    #         self.derained_image.append(derained_image)

    def __getitem__(self, item):
        image_name = self.pairs[item]
        gt_path = os.path.join(self.ground_truth_path, image_name)
        gt_image = np.array(Image.open(gt_path).convert('RGB'))
        derain_path = os.path.join(self.derained_image_path, image_name)
        derain_image = np.array(Image.open(derain_path).convert('RGB'))
        derain = self.transfrom(derain_image)
        target = self.transfrom(gt_image)
        return derain, target, image_name

class Muti_image(Dataset):
    def __init__(self, path, patch_size):
        super(Muti_image, self).__init__()
        self.path = path
        self.patch_size = patch_size
        self.pairs = os.listdir(os.path.join(self.path, 'Rain_Heavy/data_augment/train2018new'))
        self.transform = transforms.ToTensor()
        self.num = len(self.pairs)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        pairs_name = self.pairs[item]
        pairs_Heavy_path = os.path.join(self.path, 'Rain_Heavy/data_augment/train2018new/'+pairs_name)
        pairs_Medium_path = os.path.join(self.path, 'Rain_Medium/data_augment/train2018new/'+pairs_name)
        pairs_Light_path = os.path.join(self.path, 'Rain_Light/data_augment/train2018new/'+pairs_name)
        pairs_Light = np.array(Image.open(pairs_Light_path).convert('RGB'))
        pairs_Heavy = np.array(Image.open(pairs_Heavy_path).convert('RGB'))
        pairs_Medium = np.array(Image.open(pairs_Medium_path).convert('RGB'))
        assert pairs_Light.shape[1] % 2 == 0
        input_Light = pairs_Light[:, :pairs_Light.shape[1]//2, :]
        clear = pairs_Light[:, pairs_Light.shape[1]//2:, :]
        input_Medium = pairs_Medium[:, :pairs_Light.shape[1]//2, :]
        input_Heavy = pairs_Heavy[:, :pairs_Light.shape[1]//2, :]
        H = input_Light.shape[0]
        W = input_Light.shape[1]
        int_H = random.randint(0, H-self.patch_size)
        int_W = random.randint(0, W-self.patch_size)
        input_Light = input_Light[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size, :]
        input_Medium = input_Medium[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size, :]
        input_Heavy = input_Heavy[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size, :]
        clear = clear[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size, :]
        out_list = self.random_augmentation(input_Light, input_Medium, input_Heavy, clear)
        return self.transform(out_list[0]), self.transform(out_list[1]), self.transform(out_list[2]), self.transform(out_list[3])

    def data_augmentation(self, image, mode):
        if mode == 0:
            out = image
        elif mode == 1:
            out = np.flipud(image)
        elif mode == 2:
            out = np.rot90(image)
        elif mode == 3:
            out = np.rot90(image)
            out = np.flipud(out)
        elif mode == 4:
            out = np.rot90(image, k=2)
        elif mode == 5:
            out = np.rot90(image, k=2)
            out = np.flipud(out)
        elif mode == 6:
            out = np.rot90(image, k=3)
        elif mode == 7:
            out = np.rot90(image, k=3)
            out = np.flipud(out)
        else:
            raise Exception('Invalid choice of image transformation')
        return out

    def random_augmentation(self, *args):
        out = []
        flag = random.randint(1,7)
        for data in args:
            out.append(self.data_augmentation(data, flag).copy())
        return out

class Muti_image_batch(Dataset):
    def __init__(self, path, patch_size):
        super(Muti_image_batch, self).__init__()
        self.path = path
        self.patch_size = patch_size
        self.pairs = os.listdir(os.path.join(self.path, 'Rain_Heavy/data_augment/train2018new'))
        self.transform = transforms.ToTensor()
        self.num = len(self.pairs)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        pairs_name = self.pairs[item]
        pairs_Heavy_path = os.path.join(self.path, 'Rain_Heavy/data_augment/train2018new/'+pairs_name)
        pairs_Medium_path = os.path.join(self.path, 'Rain_Medium/data_augment/train2018new/'+pairs_name)
        pairs_Light_path = os.path.join(self.path, 'Rain_Light/data_augment/train2018new/'+pairs_name)
        pairs_Light = np.array(Image.open(pairs_Light_path).convert('RGB'))
        pairs_Heavy = np.array(Image.open(pairs_Heavy_path).convert('RGB'))
        pairs_Medium = np.array(Image.open(pairs_Medium_path).convert('RGB'))
        assert pairs_Light.shape[1] % 2 == 0
        input_Light = pairs_Light[:, :pairs_Light.shape[1]//2, :]
        clear = pairs_Light[:, pairs_Light.shape[1]//2:, :]
        input_Medium = pairs_Medium[:, :pairs_Light.shape[1]//2, :]
        input_Heavy = pairs_Heavy[:, :pairs_Light.shape[1]//2, :]
        H = input_Light.shape[0]
        W = input_Light.shape[1]
        int_H = random.randint(0, H-self.patch_size)
        int_W = random.randint(0, W-self.patch_size)
        input_Light = input_Light[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size, :]
        input_Medium = input_Medium[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size, :]
        input_Heavy = input_Heavy[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size, :]
        clear = clear[int_H:int_H+self.patch_size, int_W:int_W+self.patch_size, :]
        # out_list = self.random_augmentation(input_Light, input_Medium, input_Heavy, clear)
        # input_L = self.transform(out_list[0]).unsqueeze(0)
        # input_M = self.transform(out_list[1]).unsqueeze(0)
        # input_H = self.transform(out_list[2]).unsqueeze(0)
        # output =  self.transform(out_list[3]).unsqueeze(0)
        input_L = self.transform(input_Light).unsqueeze(0)
        input_M = self.transform(input_Medium).unsqueeze(0)
        input_H = self.transform(input_Heavy).unsqueeze(0)
        output = self.transform(clear).unsqueeze(0)
        input = torch.cat((input_L, input_M, input_H), dim=0)
        return input, output

    def data_augmentation(self, image, mode):
        if mode == 0:
            out = image
        elif mode == 1:
            out = np.flipud(image)
        elif mode == 2:
            out = np.rot90(image)
        elif mode == 3:
            out = np.rot90(image)
            out = np.flipud(out)
        elif mode == 4:
            out = np.rot90(image, k=2)
        elif mode == 5:
            out = np.rot90(image, k=2)
            out = np.flipud(out)
        elif mode == 6:
            out = np.rot90(image, k=3)
        elif mode == 7:
            out = np.rot90(image, k=3)
            out = np.flipud(out)
        else:
            raise Exception('Invalid choice of image transformation')
        return out

    def random_augmentation(self, *args):
        out = []
        flag = random.randint(1,7)
        for data in args:
            out.append(self.data_augmentation(data, flag).copy())
        return out


class Rain12600(Dataset):
    def __init__(self, path, patch_size):
        super(Rain12600, self).__init__()




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    samples = Rain800('./datasets/Rain1200/train/', 4, 128)
    for i in range(2):
        k, v = samples[i]
        print(k.shape, v.shape, k.dtype, v.dtype, k.mean(), v.mean())
        p = transforms.ToPILImage()(k)
        p.save('train_input_' + str(i) + '.jpg')
        p = transforms.ToPILImage()(v)
        p.save('train_target_' + str(i) + '.jpg')
    # train_queue = torch.utils.data.DataLoader(samples, batch_size=4, pin_memory=True)
    # print(len(train_queue))
    # for step, (input, target) in enumerate(train_queue):
    #     print(input.shape, target.shape)
    #     inp = input[0]
    #     out = target[0]
    #     inp = transforms.ToPILImage()(inp)
    #     out = transforms.ToPILImage()(out)
    #     inp.save('input.jpg')
    #     out.save('output.jpg')
    # samples = Rain100('./datasets/Rain100/Rain100H', 'rain_data_train_Heavy', 4, 256)
    # for i in range(4):
    #     k, v = samples[i]
    #     print(k.shape, v.shape, v.dtype, k.mean(), v.mean())
    #     p = transforms.ToPILImage()(k)
    #     p.save('train_input_' + str(i) + '.jpg')
    #     p = transforms.ToPILImage()(v)
    #     p.save('train_target_' + str(i) + '.jpg')
    # train_queue = torch.utils.data.DataLoader(samples, batch_size=4, pin_memory=True)
    # print(len(train_queue))
    # train_data = Rain800('./datasets/Rain800/'+'DID-MDN-training/Rain_Medium/train2018new/', 8, 128)
    # train_queue = torch.utils.data.DataLoader(train_data, batch_size=4, num_workers=4, pin_memory=True)
    # print(len(train_queue))
    # valid_data = Rain800('./datasets/Rain800/'+'DID-MDN-test/', 8, 128)
    # valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=4, num_workers=4, pin_memory=True)
    # print(len(valid_queue))
    # Samples = Test('./datasets/Rain800/DID-MDN-test')
    # for i in range(2):
    #     i, o, n = Samples[i]
    #     print(i.shape, o.shape, i.dtype, o.dtype, i.mean(), o.mean())
    #     p = transforms.ToPILImage()(i)
    #     p.save('input_' + n)
    #     p = transforms.ToPILImage()(o)
    #     p.save('target_' + n)
    # samples = Muti_image_batch('./datasets/Rain1200/DID-MDN-training/', 128)
    # for i in range(2):
    #     input, output = samples[i]
    #     input1, output1 = samples[i+1]
    #     input = torch.cat((input, input1), dim=0)
    #     print(input[0:3].shape)
    #     print(torch.all(input[3:6].eq(input1)))
    #     print(input.shape, input[0].shape, output.shape)







