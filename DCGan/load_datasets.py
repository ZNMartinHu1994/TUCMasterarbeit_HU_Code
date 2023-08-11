import sys, os
import cv2 as cv
import numpy as np
import time, torchvision
from torch.utils.data import Dataset


class DataLoader():
    def __init__(self, dir_path, dataset_name, norm_range=(0, 1), img_res=(64, 64)):
        '''
        :param dataset_name: The name of the datasets folder inside the datasets folder
        :param norm_range: Input images from 0-255, normalized to each pixel value within the norm_range interval (including boundaries)
        :param img_res: Shape of the image becomes (h,w) If None then no operation is performed
        '''
        self.dataset_name = dataset_name
        self.norm_range = norm_range
        self.img_res = img_res
        self.image_path = []  # First, save all the paths to the images

        ImgTypeList = ['jpg', 'JPG', 'bmp', 'png', 'jpeg', 'rgb', 'tif']
        try:
            file_names = os.listdir(dir_path)
            for dir_path, dir_names, image_names in os.walk(dir_path):
                print(dir_path, dir_names, image_names)
                for image_name in image_names:
                    if image_name.split('.')[-1] in ImgTypeList:  # The description is a picture, proceed to the next step
                        image_path = os.path.join(dir_path, image_name)
                        self.image_path.append(image_path)
            self.image_path = np.array(self.image_path)
            print(self.image_path)
        except:
            pass

    def img_normalize(self, image, norm_range=(0, 1)):
        '''
        Input images from 0-255, normalized to each pixel value within the norm_range interval (including boundaries)
        '''
        image = np.array(image).astype('float32')
        image = (norm_range[1] - norm_range[0]) * image / 255. + norm_range[0]
        return image

    def load_datasets(self, ):
        '''
        The subdirectory folder inside the parent directory is the label, and the images inside the subdirectory folder are the data for this label.
        :param dir_path:parent directory
        :return:NONE
        '''
        x_list = []
        for image_path in self.image_path:
            image = cv.imread(image_path, flags=-1)  # flags=-1 is to read in the image in full form
            if self.img_res is not None and (
                    image.shape[0] != self.img_res[1] or image.shape[1] != self.img_res[0]):
                image = cv.resize(image,
                                  dsize=(self.img_res[1], self.img_res[0]))  # If the size of the image is not the same as the size to be cropped, crop the image to the specified size
            if len(image.shape) == 2:  # Explain that it's a gray map. Skip i
                # print('k'*200)
                image = np.expand_dims(image, axis=-1)
                # if image.shape[-1]==3:
                #     image=np.repeat(image,repeats=3,axis=-1)
                # continue
            # print(image.shape)
            x_list.append(image)
        # self.x = np.array(x_list)
        self.x = np.stack(x_list, axis=0)
        if self.norm_range is not None:
            self.x = self.img_normalize(self.x, norm_range=self.norm_range)
        return self.x

    def load_batch(self, batch_size=128):  # Loading batch_size=128 once took 0.00498 seconds
        '''
        :return: x_batch, y_batch
        '''
        x_list = []
        train_ids = np.random.randint(0, len(self.image_path), size=batch_size)
        for image_path in self.image_path[train_ids]:
            image = cv.imread(image_path, flags=-1)  # flags=-1 is to read in the image in full form
            if self.img_res is not None and (
                    image.shape[0] != self.img_res[1] or image.shape[1] != self.img_res[0]):
                image = cv.resize(image,
                                  dsize=(self.img_res[1], self.img_res[0]))  # If the size of the image is not the same as the size to be cropped, crop the image to the specified size
            if len(image.shape) == 2:  # Explain that it's a gray map. Skip it
                # print('k'*200)
                image = np.expand_dims(image, axis=-1)
                # if image.shape[-1]==3:
                #     image=np.repeat(image,repeats=3,axis=-1)
                # continue
            x_list.append(image)
        x_batch = np.array(x_list)
        # x_batch = np.stack(x_list, axis=0)
        if self.norm_range is not None:
            x_batch = self.img_normalize(x_batch, norm_range=self.norm_range)
        return x_batch


class MyDataSet(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def transform(self):
        train_transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip, p is the flip probability
            # torchvision.transforms.RandomCrop(32, padding=4),  # Crop the image randomly to a width and height of 32
        ])
        return train_transforms

    def __getitem__(self, idx):
        return self.transform()(self.x[idx])


def main():
    dataset_name = '480max'  # Data set name
    dir_path = r'.\datasets\%s' % dataset_name  # Training set images
    dataloader = DataLoader(dir_path, dataset_name=dataset_name, norm_range=(0, 1), img_res=(128, 128))
    # dataloader.save_all_images(dir_path)  # (10000, 64, 64, 3)
    x = dataloader.load_datasets()
    print(x.shape, x.min(), x.max())

    for i in range(5):
        t = time.time()
        x = dataloader.load_batch(batch_size=128)
        print(x.shape, x.min(), x.max())
        print(i + 1, '用时:', time.time() - t)


if __name__ == '__main__':
    main()
