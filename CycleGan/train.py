import os
import torch
import time
import random
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
from load_datasets import CYCLEDataset
from model_cyclegan import generator, discriminator
import matplotlib.pyplot as plt
import cv2 as cv
from utils import save_some_samples, draw_loss
import torchvision

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Troubleshooting a few bugs
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.family'] = 'FangSong'  
plt.rcParams['font.size'] = 10  # Set the size of the font to 10

'''---------------------------The following is the definition of the parameter section---------------------------------'''
#the tuning parameter only needs to be changed here, but not anywhere else

total_iters = 20000  # Total number of iterations for training, default 20000
batch_size = 6  # Batches per training (too large may not have enough video memory and report an error, turn it down and try again)
h, w, c = 128, 128, 1  # width, height, and channels of the dataset image
dataset_name = 'type1'  # Name of the dataset:type1
dir_path = r'C:/work/data/1.original_dataset/%s' % dataset_name  # Path to the root directory of the dataset

# Define the parameters 
load_model = True  # Whether or not to load the model, breakpoints
lr = 5e-7  # Learning rate, default 5e-5
alpha = 10  # Hyperparameter, the size of the cyclic loss weight, default is 10
beta = 0.8  # Hyperparameter, the value of the model's channel scaling
save_step = 50  # Saves the intermediate result map and weights every few training sessions. If 0, the intermediate result map and weights are not saved
gpu = 0  # Select which GPU to compute on, if there is only 1 GPU, then this parameter can be filled with 0


#imgs_A_path = r'C:/work/data/1.original_dataset/%s/bad' % dataset_name  # Image A path. If the value is None, images are automatically randomly selected for display in the test set
#imgs_B_path = r'C:/work/data/1.original_dataset/%s/good' % dataset_name  # Image B path. If the value is None, images are automatically randomly selected for display in the test set
imgs_A_path = None
imgs_B_path = None

# Creating Folders
directory = r'C:/work/models/2.CycleGAN/' + dataset_name + r'_%s_%s' % (h, w)  # Trained model paths
image_dir = r'C:/work/check_images/2.CycleGAN/' + dataset_name + r'_%s_%s' % (h, w)  # Folder where samples of model training effect images are saved

'''---------------------------Above is the definition parameter section------------------------------------------------'''

os.makedirs(directory, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
image_names = os.listdir(image_dir)
# Choose which gpu to use, if no gpu then use cpu
if torch.cuda.is_available() and gpu >= 0:
    device = torch.device('cuda:%s' % gpu)
    print('Using the GPU:%s' % torch.cuda.get_device_name(0))
else:
    device = 'cpu'
# Load Dataset
train_dataloader = CYCLEDataset(dir_path, ['bad', 'good'],
                                dsize=(h, w), mode='train')  # mode='test' means no image enhancement is used
test_dataloader = CYCLEDataset(dir_path, ['bad', 'good'],
                               dsize=(h, w), mode='test', randomstate=0)  # mode='test' means no image enhancement is used
# Loading training sets
train_db = DataLoader(train_dataloader, batch_size=batch_size, shuffle=True)  # Loading training sets
test_db = DataLoader(test_dataloader, batch_size=batch_size, shuffle=False)  # Loading Test Sets
a = next(iter(train_db))
print(len(a), a[0].shape, a[1].shape, a[0].min(), a[0].max())
if imgs_A_path is None:
    imgs_A_path, _ = next(iter(test_db))
    imgs_A_path = imgs_A_path[0:1]  # (1,c,h,w)
if imgs_B_path is None:
    _, imgs_B_path = next(iter(test_db))
    imgs_B_path = imgs_B_path[0:1]  # (1,c,h,w)
# The image can be viewed here
# for i in range(len(a[0])):
#     img_A, img_B = a[0], a[1]
#     print(img_A.min(), img_A.max())
#     print(img_B.min(), img_B.max())
#     cv.imshow('img_A', np.flip(img_A.detach().cpu().numpy().transpose(0, 2, 3, 1)[i] * 255, -1).astype('uint8'))
#     cv.imshow('img_B', np.flip(img_B.detach().cpu().numpy().transpose(0, 2, 3, 1)[i] * 255, -1).astype('uint8'))
#     cv.waitKey()
# Use half-precision training (mixed-precision training, which speeds up training by a factor of two)
try:
    G_scaler = torch.cuda.amp.GradScaler()  # scaling of gradient
    D_A_scaler = torch.cuda.amp.GradScaler()
    D_B_scaler = torch.cuda.amp.GradScaler()
except:
    print('It can only be calculated on the CPU, takes too long and exits.')
    exit()
# Building Network Models
G_AB = generator(h, w, c, output_channel=c, beta=beta).to(device)
G_BA = generator(h, w, c, output_channel=c, beta=beta).to(device)
D_A = discriminator(c).to(device)
D_B = discriminator(c).to(device)


def save_checkpoint(directory, i, save_step):  # directory is the path to the folder where the weights are saved. i is the number of epochs. save_step is the number of steps to save the weights.
    torch.save(G_AB.state_dict(), r'%s/generator_G_AB_%s.h5' % (directory, i))  # Save the weights of the generator G_AB
    torch.save(G_BA.state_dict(), r'%s/generator_G_BA_%s.h5' % (directory, i))  # Save the weights of the generator G_BA
    torch.save(D_A.state_dict(), r'%s/discriminator_D_A_%s.h5' % (directory, i))  # Preserve the weight of the discriminator D_A
    torch.save(D_B.state_dict(), r'%s/discriminator_D_B_%s.h5' % (directory, i))  # Preserve the weight of the discriminator D_B
    try:
        os.remove(r'%s/generator_G_AB_%s.h5' % (directory, i - save_step))
        os.remove(r'%s/generator_G_BA_%s.h5' % (directory, i - save_step))
        os.remove(r'%s/discriminator_D_A_%s.h5' % (directory, i - save_step))
        os.remove(r'%s/discriminator_D_B_%s.h5' % (directory, i - save_step))  # Deleting the last saved weights saves space
    except:
        pass


# Training Models
def train(load_model):
    t = time.time()
    try:
        max_num = max(
            [int(model_name.split('.')[0].split('_')[-1]) for model_name in os.listdir(directory)]) if os.listdir(
            directory) else 0  # The weight name is saved as xxx_100.h5 like this, xxx is the name and 100 is the 100th epoch
    except:
        max_num = 0
    if load_model:
        # Loading Models
        try:  # Try to load the model, if the model already exists under the path, then continue the training at the breakpoint.
            G_AB.load_state_dict(torch.load(r'%s/generator_G_AB_%s.h5' % (directory, max_num), map_location=device),
                                 strict=False)
            G_BA.load_state_dict(torch.load(r'%s/generator_G_BA_%s.h5' % (directory, max_num), map_location=device),
                                 strict=False)
            D_A.load_state_dict(
                torch.load(r'%s/discriminator_D_A_%s.h5' % (directory, max_num), map_location=device), strict=False)
            D_B.load_state_dict(
                torch.load(r'%s/discriminator_D_B_%s.h5' % (directory, max_num), map_location=device), strict=False)
            print('breakpoint training')
        except:
            print('Model not loading, weights and network not matching! Retraining in progress ......')
            pass
    # Defining the Optimizer
    optimizer_G = optim.Adam(list(G_BA.parameters()) + list(G_AB.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr)
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr)
    # Define loss function
    mae_criterion = nn.L1Loss()  # L1 loss function, also called MAE loss function
    bce_criterion = nn.BCEWithLogitsLoss()  # Bicategorical cross-entropy loss function
    # Start iterative training
    g_loss_total_list = []
    d_loss_total_list = []
    for iters in range(total_iters):
        iters += 1
        G_AB.train()  # Generator set to training state
        G_BA.train()
        D_A.train()  # Discriminator set to training status
        D_B.train()

        total_G_loss = .0
        total_D_loss = .0
        real_A, real_B = next(iter(train_db))
        # for i, (real_A, real_B) in enumerate(train_db):
        real_A = real_A.to(device, dtype=torch.float32)
        real_B = real_B.to(device, dtype=torch.float32)
        if c == 1:  # If the setting channel is 1, it is converted to gray map
            real_A = torchvision.transforms.Grayscale(num_output_channels=1)(real_A)
            real_B = torchvision.transforms.Grayscale(num_output_channels=1)(real_B)
        real_A, real_B = (real_A - 0.5) * 2, (real_B - 0.5) * 2  # Images are -1 to 1, applied to adversarial generative networks

        # print(real_A.min(),real_A.max(),real_B.min(),real_B.max())
        with torch.cuda.amp.autocast():  # Mixed-precision training that doubles the speed of training
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            cycled_A = G_BA(fake_B)
            cycled_B = G_AB(fake_A)
            same_A = G_BA(real_A)
            same_B = G_AB(real_B)

            fake_A_out = D_A(fake_A)
            fake_B_out = D_B(fake_B)

            # Compute the generator loss function
            cycle_lossBA = mae_criterion(cycled_A, real_A)
            cycle_lossAB = mae_criterion(cycled_B, real_B)
            cycle_loss = cycle_lossBA + cycle_lossAB  # add up

            identity_lossBA = mae_criterion(same_A, real_A)
            identity_lossAB = mae_criterion(same_B, real_B)
            identity_loss = identity_lossBA + identity_lossAB  # add up

            G_BA_loss1 = bce_criterion(fake_A_out, torch.ones_like(fake_A_out))  # Generator: fake A is 1
            G_AB_loss1 = bce_criterion(fake_B_out, torch.ones_like(fake_B_out))  # Generator: fake B is 1

            G_loss = (G_BA_loss1 + G_AB_loss1) + (alpha * cycle_loss) + (alpha * 0.5 * identity_loss)

            optimizer_G.zero_grad()  # Gradient zeroing
            G_scaler.scale(G_loss).backward()  # scaling of gradient
            G_scaler.step(optimizer_G)  # backward propagation
            G_scaler.update()  # update
        with torch.cuda.amp.autocast():  # Mixed-precision training that doubles the speed of training
            # Discriminator A
            real_A_out = D_A(real_A)
            fake_A_out = D_A(fake_A.detach())

            D_A_loss = (bce_criterion(real_A_out, torch.ones_like(real_A_out)) + bce_criterion(fake_A_out,
                                                                                               torch.zeros_like(
                                                                                                   fake_A_out))) * 0.5

            optimizer_D_A.zero_grad()  # Gradient zeroing
            D_A_scaler.scale(D_A_loss).backward()  # scaling of gradient
            D_A_scaler.step(optimizer_D_A)  # backward propagation
            D_A_scaler.update()  # update
        with torch.cuda.amp.autocast():  # Mixed-precision training that doubles the speed of training
            # Discriminator B
            real_B_out = D_B(real_B)
            fake_B_out = D_B(fake_B.detach())

            D_B_loss = (bce_criterion(real_B_out, torch.ones_like(real_B_out)) + bce_criterion(fake_B_out,
                                                                                               torch.zeros_like(
                                                                                                   fake_B_out))) * 0.5

            optimizer_D_B.zero_grad()  # Gradient zeroing
            D_B_scaler.scale(D_B_loss).backward()  # scaling of gradient
            D_B_scaler.step(optimizer_D_B)  # backward propagation
            D_B_scaler.update()  # backward propagation

            # The total loss function is obtained
            total_G_loss += G_loss.item()
            total_D_loss += D_A_loss.item() + D_B_loss.item()
        g_loss_total_list.append(total_G_loss)  # Add the total loss function and draw the curve with
        d_loss_total_list.append(total_D_loss)
        # Save Loss Function
        with open('%s_loss.txt' % dataset_name, 'ab') as f:
            np.savetxt(f, [total_G_loss, ], delimiter='   ', newline='   ',
                       header='\niters:%d   g_loss:' % (iters + 1 + max_num), comments='', fmt='%.6f')
            np.savetxt(f, [total_D_loss, ], delimiter='   ', newline='   ', header='d_loss:', comments='', fmt='%.6f')
        # Save the weights and the current output graph
        if save_step:
            if iters % save_step == 0:  # Saved every epoch of the set value
                save_checkpoint(directory, iters + max_num, save_step)  # Preservation of weights
                # Save test image
                save_to = image_dir + r'/' + dataset_name + '_%d.png' % (iters + max_num)
                save_some_samples(G_AB, G_BA, train_dataloader, imgs_A_path, imgs_B_path, device, h, w, c, fake_num=1,
                                  save_to=save_to)  # Save Image
                print('iters:%s/%s' % (iters + max_num, total_iters),  # Number of training iterations
                      'total_loss_G:%.6f' % total_G_loss,  # Generator Loss Function
                      'total_loss_D:%.6f' % total_D_loss,  # Discriminator Loss Function
                      'time-consuming:%s s' % np.round((time.time() - t), 4))  # time-consuming
                t = time.time()
        if iters + max_num >= total_iters:  # If the number of training iterations is reached, jump out of the loop
            break
    try:
        draw_loss(max_num, total_iters, g_loss_total_list, d_loss_total_list)  # Drawing the training curve
    except:
        pass

    print('Has run through %s of iterations' % total_iters)


if __name__ == '__main__':
    train(load_model=load_model)
