import sys, os
import cv2 as cv
import numpy as np
import time
import torchvision, torch
from PIL import Image
import random
from torch.utils.data import DataLoader, Dataset


class CYCLEDataset(DataLoader):
    def __init__(self, dataset_dir, dirs, dsize=(256, 256), mode='test', randomstate=None):
        self.dataset_dir = dataset_dir
        self.dirs = dirs
        self.train_lists_a = os.listdir(r'%s/%s' % (dataset_dir, dirs[0]))  # List of paths to all images in domain a
        self.train_lists_b = os.listdir(r'%s/%s' % (dataset_dir, dirs[1]))  # List of paths to all images in domain b
        self.dsize = dsize  # Scale to between (h,w), where h is height and w is width
        self.mode = mode  # mode='train' means using image enhancement, mode='test' means not using image enhancement
        self.randomstate = randomstate  # This parameter controls whether or not an image should be specified for the test. If the default is None, then no image is specified, and the image will be randomly selected for each run; if it is a constant, then the image will be the same one every time.
        self.train_transform = torchvision.transforms.Compose([  # Image enhancement done during training
            torchvision.transforms.Resize((int(dsize[0] * 1.2), int(dsize[1] * 1.2))),  # Zoom to specified height and width (h,w)
            # torchvision.transforms.Resize((dsize[0], dsize[1])),
            torchvision.transforms.RandomRotation(degrees=(-15, 15)),  # random rotation
            torchvision.transforms.RandomCrop((dsize[0], dsize[1])),  # Crop to specified height and width (h,w)
            # torchvision.transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.2), shear=(-0.1, 0.1),
            #                                     fill=0),  # Randomized scaling of images
            torchvision.transforms.RandomHorizontalFlip(),  # Random Horizontal Flip
            # torchvision.transforms.ColorJitter(brightness=(0.7, 1.2),  # Brightness (0.5,1.5) is equivalent to 0.5, i.e. a change between 0.5 and 1.5 times the original image. Same for the following
            #                                    contrast=0.1,  
            #                                    saturation=0.1,  
            #                                    hue=0),  
            # torchvision.transforms.RandomAdjustSharpness(0.3),

            torchvision.transforms.ToTensor(),  # Normalized to 0 to 1
            # torchvision.transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.1, 0.1)),  # random masking
        ])
        self.test_transform = torchvision.transforms.Compose([  # Image enhancement done during testing
            torchvision.transforms.Resize((dsize[0], dsize[1])),  # Zoom to specified height and width (h,w)
            torchvision.transforms.ToTensor(),  # Normalized to 0 to 1
        ])

    def __getitem__(self, index):
        items = []
        x = Image.open('%s/%s/%s' % (self.dataset_dir, self.dirs[0], self.train_lists_a[index])).convert('RGB')
        if self.mode == 'train':
            x = self.train_transform(x)
        else:
            x = self.test_transform(x)
        items.append(x)
        if self.randomstate is None:
            indexb = random.randint(0, len(self.train_lists_b) - 1)
        else:
            random.seed(self.randomstate)  # Selection of fixed images, mainly for testing purposes
            indexb = random.randint(0, len(self.train_lists_b) - 1)
            random.seed(None)  # It was changed back to random selection to prevent the subsequent randomization from being affected
        # print(indexb)
        y = Image.open('%s/%s/%s' % (self.dataset_dir, self.dirs[1], self.train_lists_b[indexb])).convert('RGB')
        if self.mode == 'train':
            y = self.train_transform(y)
        else:
            y = self.test_transform(y)
        items.append(y)

        # print(x.shape, x.dtype, x.min(), x.max(), y.shape, y.dtype, y.min(), y.max())
        # cv.imshow('x', np.flip((x.permute(1, 2, 0).numpy() * 255),-1).astype('uint8'))
        # cv.imshow('y', np.flip((y.permute(1, 2, 0).numpy() * 255),-1).astype('uint8'))
        # cv.waitKey()
        return items  # 输出imgs为0-1

    def __len__(self):
        return len(self.train_lists_a)

    def imread(self, path, outRGB=True):
        img = cv.imread(path)
        if outRGB:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB map
        return img

    def load_img(self, img_path, normalize=True, resize=None, outRGB=True):  # This function loads the specified image and normalizes it.
        img = self.imread(img_path, outRGB)
        if resize is not None:
            img = cv.resize(img, self.dsize).astype('float32')
        if normalize:
            img = img / 127.5 - 1.
        img = img[np.newaxis, :, :, :]
        return img

    def load_imgs(self, dir_path, batch_size=None):  # Input the path of the image, output the -1 to 1 image of (b,c,h,w)
        image_names = os.listdir(dir_path)
        image_list = []
        for image_name in image_names:
            img = self.load_img(os.path.join(dir_path, image_name))
            image_list.append(img)
        img = np.concatenate(image_list, 0).astype('float32')  # Convert to float32
        img = np.transpose(img, axes=[0, 3, 1, 2])  # (b,c,h,w)
        if batch_size is not None:
            np.random.seed(0)  # Selection of fixed images, mainly for testing purposes
            ids = np.random.randint(0, len(img), size=batch_size)
            img = img[ids]  # Randomly select a batch if desired.
            np.random.seed(None)  # It was changed back to random selection to prevent the subsequent randomization from being affected
        return img  # (b,c,h,w)


def main():  # This is just debugging, and reporting errors does not affect the normal operation of the program.
    dataset = CYCLEDataset('./datasets/repair_tomato_256', ['aaa', 'bbb'],
                           dsize=(256, 256), mode='train', randomstate=0)
    train_db = DataLoader(dataset, batch_size=20, shuffle=False)
    items = next(iter(train_db))
    print(len(items))
    print(torch.cat(items, dim=0).shape)  # (b,c,h,w)

    # Check out the images here
    for i in range(len(items[0])):
        img_A, img_B = items[0], items[1]
        print(img_A.min(), img_A.max())
        print(img_B.min(), img_B.max())
        cv.imshow('img_A', np.flip((img_A.detach().cpu().numpy().transpose(0, 2, 3, 1)[i]), -1))
        cv.imshow('img_B', np.flip((img_B.detach().cpu().numpy().transpose(0, 2, 3, 1)[i]), -1))
        cv.waitKey()


if __name__ == '__main__':
    main()
