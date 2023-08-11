import torch
import numpy as np
import random
import cv2 as cv
import os, pathlib
import matplotlib.pyplot as plt
import torchvision


def save_some_samples(G_AB, G_BA, dataloader, imgs_A_path, imgs_B_path, device, h, w, c, fake_num,
                      save_to=None, show=False):  # G_AB and G_BA are generators, imgs_A_path and imgs_B_path are images or image paths
    # Randomly save test images
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    G_AB.eval(), G_BA.eval()  # Generator tuned to test state
    if isinstance(imgs_A_path, str):  # Specify the image path here as the displayed image
        imgs_A = torch.from_numpy(
            dataloader.load_img(imgs_A_path, normalize=False, resize=(w, h)).transpose([0, 3, 1, 2])) / 255.  # A
        imgs_B = torch.from_numpy(
            dataloader.load_img(imgs_B_path, normalize=False, resize=(w, h)).transpose([0, 3, 1, 2])) / 255.  # B
    else:  # If you're passing in an image, just assign it directly
        imgs_A = imgs_A_path
        imgs_B = imgs_B_path
    if c == 1:  # If the setting channel is 1, it is converted to gray map
        imgs_A = torchvision.transforms.Grayscale(num_output_channels=1)(imgs_A)
        imgs_B = torchvision.transforms.Grayscale(num_output_channels=1)(imgs_B)
    imgs_A, imgs_B = (imgs_A - 0.5) * 2, (imgs_B - 0.5) * 2  # Images are -1 to 1, applied to adversarial generative networks
    # print(imgs_B.min(), imgs_B.max())
    # Generate fake image B
    fake_B = G_AB(imgs_A.float().to(device)).detach().cpu().numpy()
    # Generate fake image A
    fake_A = G_BA(imgs_B.float().to(device)).detach().cpu().numpy()

    gen_imgs = np.concatenate(
        [imgs_A, fake_B, imgs_B, fake_A])  # Stitching these images together into one big picture for displaying
    gen_imgs = (0.5 * gen_imgs + 0.5).transpose([0, 2, 3, 1])  # Image inverse normalized to 0-255 pixel values for displaying images
    # print('gen_imgs:',gen_imgs.shape, gen_imgs.min(), gen_imgs.max())
    ## The following are all drawings, which are performed in order to display the image
    plt.figure(figsize=(10, 10))  # Resolution of the whole large image = figsize*100
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        if c == 1:  # If it's a grayscale map then display a grayscale map
            plt.imshow(gen_imgs[i], cmap='gray')
        else:
            plt.imshow(gen_imgs[i])
        plt.axis('off')
        if i == 0:
            ax.set_title('Original', fontsize=25)  # Setting Title
        elif i == fake_num:
            ax.set_title('Transform', fontsize=25)
    plt.tight_layout()  # Automatically adjusts the spacing between subgraphs
    if save_to is not None:
        print(r'The output image has been saved to:%s' % (pathlib.PurePath(os.getcwd(), save_to)))
        plt.savefig(save_to)  # Save Image
        if show:
            plt.show()
    plt.close()  # Close Image
    G_AB.train()  # The generator is called back to the training state, in order to train next
    G_BA.train()


def draw_loss(start_epochs, end_epochs, g_loss_list, d_loss_list):  # 画出损失函数
    plt.plot(range(start_epochs, end_epochs), g_loss_list, label='g_loss', color='g')
    plt.plot(range(start_epochs, end_epochs), d_loss_list, label='d_loss', color='r')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xticks(range(start_epochs, end_epochs + 1, 10))
    plt.legend()
    plt.show()
