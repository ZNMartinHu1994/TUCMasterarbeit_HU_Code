import sys, os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import time
from network import Generator, Discriminator
from load_datasets import DataLoader, MyDataSet
import torchvision
import torch.autograd as autograd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Troubleshooting a few bugs
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.family'] = 'FangSong'  
plt.rcParams['font.size'] = 10  # Set the size of the font to 10


def save_image(fake_image, fake_label=None, fake_num=16, save_to=''):  # X is a 4-dimensional image with batches
    plt.figure()
    _, h, w, c = fake_image.shape
    # print(fake_image.shape, fake_image.max(), fake_image.min())
    if fake_num > 1:
        for i in range(fake_num):
            plt.subplot(4, 4, i + 1)
            if c == 3:  # Convert to rgb if bgr
                plt.imshow(cv.cvtColor(fake_image[i], cv.COLOR_BGR2RGB))
            if c == 1:  # Displays a grayscale image if it is a grayscale image
                plt.imshow(np.squeeze(fake_image[i]), cmap='gray')
            if fake_label is not None:  # Pass in the label if there is one
                plt.title(fake_label[i])
            plt.axis('off')
    else:  # Save single image when fake_num=1
        cv.imwrite(save_to, fake_image[0])
    print('Output image has been saved to : %s' % save_to)
    plt.savefig(save_to)
    plt.close()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  


'''---------------------------The following is the definition of the parameter section---------------------------------'''
#the tuning parameter only needs to be changed here, but not anywhere else

h, w, c = 128, 128, 1  # width, height, and channels of the dataset image
epochs = 20000  # Number of iterations of training
batch_size = 240  # Training batch (too large may not have enough video memory and report an error, turn it down and try again)
dataset_name = 'type1'  # Name of the dataset
dir_path = r'C:/work/data/1.original_dataset/%s/bad' % dataset_name  # Path to the dataset

# Define the parameters (the ones here are generally not changed)
noise_dim = 100  # Dimension of noise (generally not changed)
g_lr = 0.0002  # Generator learning rate
d_lr = 0.0002  # Discriminator learning rate
fake_num = 16  # The number of images generated (don't change this parameter easily, otherwise many parts of the whole code will have to be changed!)
save_step = 1  # Saves the intermediate result images and weights every few epochs. (If 0, intermediate result images and weights are not saved)

# Creating Folders
print(device)
assert os.path.exists(dir_path), 'Path: %s does not exist, please check dir_path' % dir_path
directory = r'C:/work/checkpoint/1.DCGAN/5.bad/' + dataset_name + r'_%s_%s' % (h, w)  # Folder where model parameters are saved
image_dir = r'C:/work/check_images/1.DCGAN/5.bad/' + dataset_name + r'_%s_%s' % (h, w)  # Folder where samples of model training effect images are saved

'''---------------------------Above is the definition parameter section------------------------------------------------'''


os.makedirs(directory, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
image_names = os.listdir(image_dir)
generator_weights_path = os.path.join(directory, 'generator.h5')
discriminator_weights_path = os.path.join(directory, 'discriminator.h5')
# Loading datasets and preprocessing
dataloader = DataLoader(dir_path, dataset_name=dataset_name, norm_range=(-1, 1), img_res=(h, w))
x_train = dataloader.load_datasets()
x_train = torch.tensor(np.transpose(x_train, axes=[0, 3, 1, 2]).astype('float32'))  # Transpose and normalize
print(x_train.shape, x_train.min(), x_train.max())
torch_train_dataset = MyDataSet(x_train)
# torch_train_dataset = torch.utils.data.TensorDataset(x_train)
train_db = torch.utils.data.DataLoader(dataset=torch_train_dataset, batch_size=batch_size, shuffle=True)
## This comment can simply show the image to check that there is no error
# for i in range(batch_size):
#     x = next(iter(train_db))
#     x_ = cv.cvtColor(x[i].numpy().transpose(1, 2, 0), cv.COLOR_BGR2RGB)
#     plt.imshow((x_ * 127.5 + 127.5).astype('uint8'))
#     plt.show()
# print(next(iter(train_db))[0].shape)

# network building
generator = Generator(c, h, w, noise_dim=noise_dim).to(device)
discriminator = Discriminator(c, h, w).to(device)
# Building optimizers and loss functions
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))  # Setting up the Optimizer
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
g_scaler = torch.cuda.amp.GradScaler()  # Semi-precision training
d_scaler = torch.cuda.amp.GradScaler()  # Semi-precision training
adversarial_loss = nn.BCEWithLogitsLoss()  # Bicategorical cross-entropy loss function
# Setting a random seed (which allows the resulting intermediate result images to be unchanged)
# np.random.seed(1)
noise_test = torch.normal(0, 1, [fake_num, noise_dim]).to(device)
# noise_test = torch.from_numpy(np.random.normal(0, 1, [fake_num, noise_dim])).to(device)
# np.random.seed(None)
# Try to load the model, breakpoints
restore_epoch = 1
try:
    generator_state = torch.load(generator_weights_path, map_location=device)
    discriminator_state = torch.load(discriminator_weights_path, map_location=device)
    generator.load_state_dict(generator_state['state'], strict=False)  # Loading generator weights
    discriminator.load_state_dict(discriminator_state['state'], strict=False)  # Load Discriminator Weights
    restore_epoch = generator_state['epoch']
    print('Load the model trained %s of times' % restore_epoch)
except:
    print('Model not loaded, retrain!')


# Training and forecasting
def train(train_db):
    generator.train().float()
    g_total_loss = .0
    d_total_loss = .0
    for i, x in enumerate(train_db):
        x = x.float().to(device)
        if c == 1 and x.shape[1] != 1:  # If the set channel is 1, it will be converted to grayscale image
            x = torchvision.transforms.Grayscale(num_output_channels=1)(x)
        # ----------Start training generator----------
        noise = torch.normal(0, 1, [len(x), noise_dim]).to(device)  # (b,noise_dim) Generate random noise, input into generator, generate image
        with torch.cuda.amp.autocast():
            fake_x = generator(noise)  # Generator generates fake images
            fake_x_out = discriminator(fake_x)

            # print(noise.min(),noise.max())
            # print(fake_x.min(), fake_x.max())
            # print(fake_x_out.min(), fake_x_out.max())
            # print(fake_x_out.shape)
            g_loss = adversarial_loss(fake_x_out, torch.ones_like(fake_x_out))  # Calculated Losses
            g_total_loss += g_loss.item()
            # Gradient Normalized 0 Backpropagation
            g_optimizer.zero_grad()
            g_scaler.scale(g_loss).backward()
            g_scaler.step(g_optimizer)
            g_scaler.update()
        # ----------Start training the discriminator----------
        with torch.cuda.amp.autocast():
            x_out = discriminator(x)
            d_real_loss = adversarial_loss(x_out, torch.ones_like(x_out))  # Loss of real images

            fake_x_out = discriminator(fake_x.detach())
            d_fake_loss = adversarial_loss(fake_x_out, torch.zeros_like(fake_x_out))  # Loss of generated images
            d_loss = (d_real_loss + d_fake_loss) / 2  # total loss
            d_total_loss += d_loss.item()
            # Gradient Normalized 0 Backpropagation
            d_optimizer.zero_grad()
            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_optimizer)
            d_scaler.update()
    return [g_total_loss, d_total_loss]


# Training Models
t1 = time.time()
for epoch in range(epochs):
    epoch += restore_epoch
    if epoch > epochs + 1:
        print('已跑完%s次' % epochs)
        break
    g_loss, d_loss = train(train_db)
    # Saving the loss function
    with open('%s_loss.txt' % dataset_name, 'ab') as f:
        np.savetxt(f, [g_loss, ], delimiter='   ', newline='   ',
                   header='\nepoch:%d   g_loss:' % epoch, comments='', fmt='%.6f')
        np.savetxt(f, [d_loss, ], delimiter='   ', newline='   ', header='d_loss:', comments='', fmt='%.6f')
    # Saving
    if save_step:
        if epoch % save_step == 0 or epoch == 0:  # Saved every few epochs
            generator.eval().half()  # Generator is set to test model
            # Save the intermediate process image of the training
            fake_image = generator(noise_test.half().to(device))  # (b,c,h,w)
            fake_image = fake_image.detach().cpu().numpy().transpose(0, 2, 3,
                                                                     1) * 127.5 + 127.5  # (b,h,w,c) Normalized back to 0-255 in order to save the image
            fake_image = np.clip(fake_image, 0, 255).astype('uint8')  # (b,h,w,c)
            save_to = image_dir + '/' + dataset_name + '%d.png' % epoch
            save_image(fake_image, fake_label=None, fake_num=fake_num, save_to=save_to)  # Save Image
            # Save model weights
            generator_state = {'state': generator.state_dict(), 'epoch': epoch, }
            discriminator_state = {'state': discriminator.state_dict(), 'epoch': epoch, }
            torch.save(generator_state, generator_weights_path)  # Save Generator weights
            torch.save(discriminator_state, discriminator_weights_path)  # Save Generator weights
            print('epoch:%d/%d:' % (epoch, epochs),
                  'g_total_loss:%.6f' % g_loss,
                  'd_total_loss:%.6f' % d_loss, '用时:%.6f' % (time.time() - t1))
            t1 = time.time()
