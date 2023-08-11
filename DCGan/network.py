import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class Generator(nn.Module):  # The generator consists of 3 transposed convolutional layers, where the last transposed convolutional layer is the output layer
    def __init__(self, input_c=3, input_h=128, input_w=128, noise_dim=100):
        super(Generator, self).__init__()
        self.input_c, self.input_h, self.input_w = input_c, input_h, input_w
        self.h_, self.w_ = input_h // 4, input_w // 4
        self.noise_dim = noise_dim
        out_channels = 128 * self.h_ * self.w_
        self.linear = nn.Sequential(nn.Linear(self.noise_dim, out_channels),  # Fully connected layer for upscaling the input noise
                                    # nn.BatchNorm1d(out_channels, out_channels, momentum=0.8),  # batch normalization layer
                                    nn.LeakyReLU(negative_slope=0.2)  # activation function
                                    )
        self.features = nn.Sequential(
            self._Conv_block(in_channels=128, out_channels=64, stride=2, kernel_size=3, padding=1),
            self._Conv_block(in_channels=64, out_channels=32, stride=2, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=input_c, stride=1, kernel_size=4, padding=0),
            nn.Tanh(),  # Normalized to between -1 and 1
        )

    def _Conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        feature = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding,
                kernel_size=kernel_size, bias=True),  # convolutional layer
            nn.BatchNorm2d(out_channels, out_channels, momentum=0.8),  # batch normalization layer
            nn.LeakyReLU(negative_slope=0.2)  # activation function
        )
        return feature

    def forward(self, x):  # (b,noise_dim)
        x = self.linear(x)  # (b,h/4*w/4*256)
        x = x.view(-1, 128, self.h_, self.w_)  # (b,256,h/4,w/4)
        x = self.features(x)  # (b,3,h,w)
        return x



class Discriminator(nn.Module):  # The discriminator consists of 3 convolutional layers, where the last convolutional layer is not connected to an activation function, and is finally converted to a one-dimensional vector with values between 0 and 1 using a fully connected layer
    def __init__(self, input_c=3, input_h=128, input_w=128):
        super(Discriminator, self).__init__()
        self.h_, self.w_ = input_h // 8, input_w // 8
        self.features = nn.Sequential(
            self._Conv_block(in_channels=input_c, out_channels=32, stride=2, kernel_size=3,
                             padding=1),
            self._Conv_block(in_channels=32, out_channels=64, stride=2, kernel_size=3,
                             padding=1),
            self._Conv_block(in_channels=64, out_channels=128, stride=2, kernel_size=3,
                             padding=1),
            nn.Flatten(),
            nn.Linear(self.h_ * self.w_ * 128, 1)  # The fully connected layer is used as the output layer of the discriminator, with values from 0 to 1
            # nn.Sigmoid()
        )

    def _Conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        feature = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding,
                kernel_size=kernel_size, bias=True),  # convolutional layer
            nn.BatchNorm2d(out_channels, out_channels, momentum=0.8),  # batch normalization layer
            nn.LeakyReLU(negative_slope=0.2)  # activation function
        )
        return feature

    def forward(self, x):  # (b,c,h,w)
        x = self.features(x)  # (b,1)
        return x
#
# helper conv function
# def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
#     layers = []
#     conv_layer = nn.Conv2d(in_channels, out_channels,
#                            kernel_size, stride, padding, bias=False)
#
#     # Appending the layer
#     layers.append(conv_layer)
#     # Applying the batch normalization if it's given true
#     if batch_norm:
#         layers.append(nn.BatchNorm2d(out_channels))
#     # returning the sequential container
#     return nn.Sequential(*layers)
#
#
# def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
#     layers = []
#     convt_layer = nn.ConvTranspose2d(in_channels, out_channels,
#                                      kernel_size, stride, padding, bias=False)
#
#     # Appending the above conv layer
#     layers.append(convt_layer)
#
#     if batch_norm:
#         # Applying the batch normalization if True
#         layers.append(nn.BatchNorm2d(out_channels))
#
#     # Returning the sequential container
#     return nn.Sequential(*layers)
#
#
# class Generator(nn.Module):
#
#     def __init__(self, z_size=100, conv_dim=32):
#         super(Generator, self).__init__()
#
#         self.z_size = z_size
#
#         self.conv_dim = conv_dim
#
#         # fully-connected-layer
#         self.fc = nn.Linear(z_size, self.conv_dim * 8 * 2 * 2)
#         # 2x2
#         self.dcv1 = deconv(self.conv_dim * 8, self.conv_dim * 4, 4, batch_norm=True)
#         # 4x4
#         self.dcv2 = deconv(self.conv_dim * 4, self.conv_dim * 2, 4, batch_norm=True)
#         # 8x8
#         self.dcv3 = deconv(self.conv_dim * 2, self.conv_dim, 4, batch_norm=True)
#         # 16x16
#         self.dcv4 = deconv(self.conv_dim, 3, 4, batch_norm=False)
#         # 32 x 32
#
#     def forward(self, x):
#         # Passing through fully connected layer
#         x = self.fc(x)
#         # Changing the dimension
#         x = x.view(-1, self.conv_dim * 8, 2, 2)
#         # Passing through deconv layers
#         # Applying the ReLu activation function
#         x = F.relu(self.dcv1(x))
#         x = F.relu(self.dcv2(x))
#         x = F.relu(self.dcv3(x))
#         x = F.tanh(self.dcv4(x))
#         # returning the modified image
#         return x
#
#
# class Discriminator(nn.Module):
#
#     def __init__(self, conv_dim=32):
#         super(Discriminator, self).__init__()
#
#         self.conv_dim = conv_dim
#
#         # 32 x 32
#         self.cv1 = conv(3, self.conv_dim, 4, batch_norm=False)
#         # 16 x 16
#         self.cv2 = conv(self.conv_dim, self.conv_dim * 2, 4, batch_norm=True)
#         # 4 x 4
#         self.cv3 = conv(self.conv_dim * 2, self.conv_dim * 4, 4, batch_norm=True)
#         # 2 x 2
#         self.cv4 = conv(self.conv_dim * 4, self.conv_dim * 8, 4, batch_norm=True)
#         # Fully connected Layer
#         self.fc1 = nn.Linear(self.conv_dim * 8 * 2 * 2, 1)
#
#     def forward(self, x):
#         # After passing through each layer
#         # Applying leaky relu activation function
#         x = F.leaky_relu(self.cv1(x), 0.2)
#         x = F.leaky_relu(self.cv2(x), 0.2)
#         x = F.leaky_relu(self.cv3(x), 0.2)
#         x = F.leaky_relu(self.cv4(x), 0.2)
#         # To pass throught he fully connected layer
#         # We need to flatten the image first
#         x = x.view(-1, self.conv_dim * 8 * 2 * 2)
#         # Now passing through fully-connected layer
#         x = self.fc1(x)
#         return x

if __name__ == '__main__':
    b, c, h, w = 10, 3, 64, 64
    noise_dim = 100
    noise = torch.randn([b, noise_dim])
    generator = Generator()
    fake_img = generator(noise)
    print(fake_img.shape)
    discriminator = Discriminator()
    fake_out = discriminator(fake_img)
    print(fake_out.shape)
