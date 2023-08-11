import os
import torch
import numpy as np
from load_datasets import DataLoader, MyDataSet
from network import Generator, Discriminator
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
batch_size = 10  # Testing batch (too large may not have enough video memory and report an error, turn it down and try again)
gen_image_num = 2000  # Number of images generated
dsize = (56, 379)  # (width, height) The size to which the generated image will be scaled
dataset_name = 'type1'  # Name of the dataset
dir_path = r'C:/work/data/1.original_dataset/%s/bad' % dataset_name # Path to the dataset
save_dir = r'C:/work/generated_images/1.DCGAN/dc_1.bad'  # Output the path to the folder where the images are generated

# Define the parameters (the ones here are generally not changed)
noise_dim = 100  # Dimension of noise (generally not changed)
load_model = True  # Loading Models
gpu = 0  # Select which GPU to compute on (put 0 here if there is only one GUP)


directory = r'C:/work/models/1.DCGAN/type1/1.bad/' # Trained model paths
image_dir = r'C:/work/check_images/1.DCGAN/1.bad/' + dataset_name + r'_%s_%s' % (h, w)  # Folder where samples of model training effect images are saved

'''---------------------------Above is the definition parameter section------------------------------------------------'''


# os.makedirs(directory, exist_ok=True)
# os.makedirs(image_dir, exist_ok=True)
image_names = os.listdir(image_dir)
generator_weights_path = os.path.join(directory, 'generator.h5')
discriminator_weights_path = os.path.join(directory, 'discriminator.h5')
# Choose which gpu to use, if no gpu then use cpu
if torch.cuda.is_available() and gpu >= 0:
    device = torch.device('cuda:%s' % gpu)
    print('使用GPU:%s' % torch.cuda.get_device_name(0))
else:
    device = 'cpu'
# Loading datasets and preprocessing
dataloader = DataLoader(dir_path, dataset_name=dataset_name, norm_range=(-1, 1), img_res=(h, w))
x_train = dataloader.load_datasets()
x_train = torch.tensor(np.transpose(x_train, axes=[0, 3, 1, 2]).astype('float32'))  # Transpose and normalize
print(x_train.shape, x_train.min(), x_train.max())
torch_train_dataset = MyDataSet(x_train)
# torch_train_dataset = torch.utils.data.TensorDataset(x_train)
train_db = torch.utils.data.DataLoader(dataset=torch_train_dataset, batch_size=batch_size, shuffle=False)
# Building Network Models
generator = Generator(c, h, w, noise_dim=noise_dim).to(device)
generator.eval()  # Set to test state for testing
# 输入数据，为随机噪声
noise_test = torch.normal(0, 1, [gen_image_num * 3, noise_dim]).to(device)
# noise_test = torch.randn([gen_image_num*3, noise_dim],).to(device)
# Input data as random noise
try:
    generator_state = torch.load(generator_weights_path, map_location=device)
    generator.load_state_dict(generator_state['state'], strict=False)  # Loading generator weights
    restore_epoch = generator_state['epoch']
    print('Load trained %s times weights' % restore_epoch)
except:
    print('The weights are not loaded!')


def gen_images_from_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    noise_test_loader = torch.utils.data.TensorDataset(noise_test)
    noise_test_db = torch.utils.data.DataLoader(noise_test_loader, batch_size=batch_size, shuffle=False)
    fake_image_list = []
    for noise in noise_test_db:
        noise = noise[0]
        fake_image = generator(noise.to(device))  # (b,c,h,w)
        mse_loss = mse(fake_image[0:batch_size].detach().cpu().numpy(), x_train[0:batch_size].detach().cpu().numpy())
        # print(mse_loss)
        if mse_loss <= 23000:
            fake_image = fake_image.detach().cpu().numpy().transpose(0, 2, 3,
                                                                     1) * 127.5 + 127.5  # (b,h,w,c) Normalized back to 0-255 in order to save the image
            fake_image = np.clip(fake_image, 0, 255).astype('uint8')  # (b,h,w,c)
            fake_image_list.append(fake_image)
        if len(fake_image_list) >= gen_image_num:
            break
    # save_to = image_dir + '/' + dataset_name + '%d.png' % epoch
    fake_images = np.concatenate(fake_image_list, axis=0)  # Value range 0 to 1
    # print(gen_imgs.shape, gen_imgs.min(), gen_imgs.max())
    for i, fake_image in enumerate(fake_images):
        save_to = os.path.join(save_dir, '%s.png' % (i + 1))
        fake_image = cv.resize(fake_image, dsize=(dsize[1], dsize[0]))
        cv.imwrite(save_to, fake_image.astype('uint8'))  # Save Image


def mse(imageA, imageB):  # The smaller the value, the better, and the smaller it is the more similar the two images are
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # Perform error normalization
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# Save all test images from folder to another folder
print('Images being generated from :%s to :%s' % (dir_path, save_dir))
gen_images_from_dir(save_dir)
print('Done')
