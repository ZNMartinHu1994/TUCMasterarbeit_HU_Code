import os
import torch
import time
import random
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
from load_datasets import CYCLEDataset
from model_cyclegan import generator, discriminator
import matplotlib.pyplot as plt
import cv2 as cv
import torchvision

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Troubleshooting a few bugs
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.family'] = 'FangSong'  
plt.rcParams['font.size'] = 10  # Set the size of the font to 10

'''---------------------------The following is the definition of the parameter section---------------------------------'''
#the tuning parameter only needs to be changed here, but not anywhere else

h, w, c = 128, 128, 1  # width, height, and channels of the dataset image
dsize = (56, 379)  # (width, height) The size to which the generated image will be scaled
dataset_name = 'type1'  # Name of the dataset:type1
dataset_path = r'C:/work/data/1.original_dataset/%s' % dataset_name # Dataset Path

dir_path = r'C:/work/data/1.original_dataset/%s/good' % dataset_name  # Input folder path
save_dir = r'C:/work/generated_images/2.CycleGAN/type1/test_from_dirB2A'  # Output folder path

# Define the parameters 
gpu = 0  # Which GPU to choose for computation
gen_image_range = (0, 999999)  # Range of generated images (not normally changed here)
load_model = True  # Loading Models
beta = 0.8  # Value of channel scaling

# 创建文件夹
directory = r'C:/work/Masterarbeit/checkpoint/2.CycleGAN/' + dataset_name + r'_%s_%s' % (h, w)  # Trained model paths
image_dir = r'C:/work/Masterarbeit/check_images/2.CycleGAN/' + dataset_name + r'_%s_%s' % (h, w)  # Folder where samples of model training effect images are saved

'''---------------------------Above is the definition parameter section------------------------------------------------'''

# os.makedirs(directory, exist_ok=True)
# os.makedirs(image_dir, exist_ok=True)
# Choose which gpu to use, if no gpu then use cpu
if torch.cuda.is_available() and gpu >= 0:
    device = torch.device('cuda:%s' % gpu)
    print('使用GPU:%s' % torch.cuda.get_device_name(0))
else:
    device = 'cpu'
# Load Dataset
dataloader = CYCLEDataset(dataset_path, ['bad', 'good'],
                          dsize=(h, w), mode='test', randomstate=0)  # mode='test' means no image enhancement is used
# Building Network Models
G_AB = generator(h, w, c, output_channel=c, beta=beta).to(device)
G_BA = generator(h, w, c, output_channel=c, beta=beta).to(device)
G_AB.eval(), G_BA.eval()  # Set to test state for testing
max_num = max(
    [int(model_name.split('.')[0].split('_')[-1]) for model_name in os.listdir(directory)]) if os.listdir(
    directory) else 0  # The weight names are saved as xxx_100.h5 like this, xxx is the name and 100 is the 100th epoch
# Loading Models
try:  # Load model
    G_AB.load_state_dict(torch.load(r'%s/generator_G_AB_%s.h5' % (directory, max_num), map_location=device),
                         strict=False)
    G_BA.load_state_dict(torch.load(r'%s/generator_G_BA_%s.h5' % (directory, max_num), map_location=device),
                         strict=False)
    print('Load the weights of the pre-trained model')
except:
    print('The model is not loaded and the weights do not match the network!')
    pass


def gen_images_from_dir(gen_image_range, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image_names = os.listdir(dir_path)
    try:
        image_names = sorted(image_names, key=lambda x: int(x.split('.')[0]))  # Image names are sorted from smallest to largest
    except:
        pass
    images = []  # Retrieve all images from the input folder
    image_names = image_names[gen_image_range[0]: gen_image_range[1]]  # Generate images within a specified range
    for image_name in image_names:
        image_path = os.path.join(dir_path, image_name)
        image = dataloader.load_img(image_path, normalize=False, resize=(w, h)).transpose(
            [0, 3, 1, 2]) / 255.  # Load the image to be tested
        images.append(image)
    images = torch.from_numpy(np.concatenate(images, axis=0))  # (b,c,h,w)
    if c == 1:  # If the setting channel is 1, it is converted to gray map
        images = torchvision.transforms.Grayscale(num_output_channels=1)(images)
    images = (images - 0.5) * 2  # Images are -1 to 1, applied to adversarial generative networks
    data_loader = torch.utils.data.TensorDataset(images)
    test_db = DataLoader(data_loader, batch_size=1, shuffle=False)  # Loading training sets
    gen_imgs = []
    test_imgs = []
    for test_img in test_db:  # Input each image into the model for conversion
        test_img = test_img[0]
        with torch.no_grad():  # Test Generator
            # Generate a fake image
            gen_img = G_BA(test_img.float().to(device)).detach().cpu().numpy()
        gen_img = (0.5 * gen_img + 0.5).transpose([0, 2, 3, 1])  # Image inverse normalized to 0-255 pixel values for displaying images
        test_img = (0.5 * test_img.detach().cpu().numpy() + 0.5).transpose([0, 2, 3, 1])
        gen_imgs.append(gen_img)
        test_imgs.append(test_img)
    gen_imgs = np.concatenate(gen_imgs, axis=0)  # Value range 0 to 1
    test_imgs = np.concatenate(test_imgs, axis=0)  # Value range 0 to 1
    # print(gen_imgs.shape, gen_imgs.min(), gen_imgs.max())
    for gen_img, test_img, image_name in zip(gen_imgs, test_imgs, image_names):
        gen_img = (cv.cvtColor(gen_img, cv.COLOR_RGB2BGR) * 255.).astype('uint8')  # Convert to BGR, inverse normalized to 0 to 255 for cv save image
        test_img = (cv.cvtColor(test_img, cv.COLOR_RGB2BGR) * 255.).astype('uint8')
        # save_img = np.concatenate([test_img, gen_img], axis=1)
        save_img = gen_img
        save_to = os.path.join(save_dir, image_name)  # Save Path
        save_img = cv.resize(save_img, dsize=(dsize[1], dsize[0]))
        cv.imwrite(save_to, save_img)

    # Save all test images from the folder


print('The model is being tested...')

gen_images_from_dir(gen_image_range, save_dir)
print('Completed, the generated image has been saved to:', save_dir)
