import torch
from torch import nn
from resnet import get_resnet


class generator(nn.Module):  # generator
    def __init__(self, input_height, input_width, input_channel, output_channel=3, beta=1.):
        super(generator, self).__init__()
        self.resnet = get_resnet(input_height, input_width, input_channel, output_channel, beta)

    def forward(self, x):
        return self.resnet(x)


class conv2d(nn.Module):
    def __init__(self, input_channel, filters, f_size=4, normalization=True):
        super(conv2d, self).__init__()
        self.normalization = normalization
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv = nn.Conv2d(input_channel, filters, f_size, stride=2)
        self.instancenormalization = nn.InstanceNorm2d(filters)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        x = self.zero_pad(inputs)
        x = self.conv(x)
        if self.normalization:
            x = self.instancenormalization(x)
        x = self.leakyrelu(x)
        return x


class discriminator(nn.Module):  # discriminator
    def __init__(self, input_channel):
        super(discriminator, self).__init__()
        set_channels = [64, 128, 256, 512]
        self.conv1 = conv2d(input_channel, set_channels[0], normalization=False)
        self.conv2 = conv2d(set_channels[0], set_channels[1])
        self.conv3 = conv2d(set_channels[1], set_channels[2])
        self.conv4 = conv2d(set_channels[2], set_channels[3])

        self.conv_out = nn.Conv2d(set_channels[3], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):  # Parentheses are examples of the shape of a specific image, denoted as (c,h,w) i.e. (number of image channels, image height, image width)
        # 64,64,64
        x = self.conv1(inputs)
        # 128,32,32
        x = self.conv2(x)
        # 256,16,16
        x = self.conv3(x)
        # 512,8,8
        x = self.conv4(x)
        # Determine whether it is valid for each pixel point
        # 64
        # 1,8,8
        validity = self.conv_out(x)  # Note: No sigmoid.
        return validity


def main():  # This is just debugging, and reporting errors does not affect the normal operation of the program.
    from torchsummary import summary
    b, c, h, w = 2, 1, 196, 160
    x = torch.ones([b, c, h, w], dtype=torch.float32).cuda()
    G = generator(h, w, c, output_channel=c).cuda()
    fake_img = G(x)
    print(fake_img.shape)
    D = discriminator(c).cuda()
    y_pred = D(fake_img)
    print(y_pred.shape)
    # summary(model, input_size=(c, h, w))  # 2,759,105


if __name__ == '__main__':
    main()
