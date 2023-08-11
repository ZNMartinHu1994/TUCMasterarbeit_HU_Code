import torch
from torch import nn

IMAGE_ORDERING = 'channels_last'


def one_side_pad(x):  # 0 Filling
    x = nn.ZeroPad2d((1, 1, 1, 1))(x)
    return x


class identity_block(nn.Module):
    def __init__(self, input_channel, filter_num):
        super(identity_block, self).__init__()
        self.conv_block1 = nn.Sequential(nn.ZeroPad2d((1, 1, 1, 1)),
                                         nn.Conv2d(input_channel, filter_num, (3, 3)),
                                         nn.InstanceNorm2d(input_channel),
                                         nn.ReLU())

        self.conv_block2 = nn.Sequential(nn.ZeroPad2d((1, 1, 1, 1)),
                                         nn.Conv2d(input_channel, filter_num, (3, 3)),
                                         nn.InstanceNorm2d(input_channel))

    def forward(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        # residual network
        x = x + inputs
        x = nn.ReLU()(x)
        return x


class get_resnet(nn.Module):
    def __init__(self, input_height, input_width, input_channel, output_channel=3, beta=1.):
        super(get_resnet, self).__init__()
        self.beta = beta
        set_channels = [int(64 * self.beta), int(128 * self.beta), int(256 * self.beta)]
        self.beta = beta
        self.conv_block1 = nn.Sequential(nn.ZeroPad2d((2, 2, 2, 2)),
                                         nn.Conv2d(input_channel, set_channels[0], (5, 5)),
                                         nn.InstanceNorm2d(set_channels[0]),
                                         nn.ReLU())
        self.conv_block2 = nn.Sequential(nn.ZeroPad2d((1, 1, 1, 1)),
                                         nn.Conv2d(set_channels[0], set_channels[1], (3, 3), stride=2),
                                         nn.InstanceNorm2d(set_channels[1]),
                                         nn.ReLU())

        self.conv_block3 = nn.Sequential(nn.ZeroPad2d((1, 1, 1, 1)),
                                         nn.Conv2d(set_channels[1], set_channels[2], (3, 3), stride=2),
                                         nn.InstanceNorm2d(set_channels[2]),
                                         nn.ReLU())
        self.identity_block1 = identity_block(set_channels[2], int(256 * self.beta))
        self.identity_block2 = identity_block(int(256 * self.beta), int(256 * self.beta))
        self.identity_block3 = identity_block(int(256 * self.beta), int(256 * self.beta))
        self.identity_block4 = identity_block(int(256 * self.beta), int(256 * self.beta))
        self.identity_block5 = identity_block(int(256 * self.beta), int(256 * self.beta))
        self.identity_block6 = identity_block(int(256 * self.beta), int(256 * self.beta))
        self.identity_block7 = identity_block(int(256 * self.beta), int(256 * self.beta))
        self.identity_block8 = identity_block(int(256 * self.beta), int(256 * self.beta))
        self.identity_block9 = identity_block(int(256 * self.beta), int(256 * self.beta))

        self.upsampling2D = nn.ReflectionPad2d((2, 2, 2, 2))

        self.deconv_block1 = nn.Sequential(nn.Upsample(scale_factor=2),
                                           nn.ZeroPad2d((1, 1, 1, 1)),
                                           nn.Conv2d(int(256 * self.beta), int(128 * beta), (3, 3)),
                                           nn.InstanceNorm2d(input_channel),
                                           nn.ReLU())

        self.deconv_block2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                           nn.ZeroPad2d((1, 1, 1, 1)),
                                           nn.Conv2d(int(128 * self.beta), int(64 * beta), (3, 3)),
                                           nn.InstanceNorm2d(input_channel),
                                           nn.ReLU())
        self.deconv_block3 = nn.Sequential(nn.ZeroPad2d((2, 2, 2, 2)),
                                           nn.Conv2d(int(64 * self.beta), output_channel, (5, 5)),
                                           nn.Tanh())

    def forward(self, inputs):
        # 3,128,128 -> 64,128,128
        x = self.conv_block1(inputs)
        # 64,128,128 -> 128,64,64
        x = self.conv_block2(x)
        # 128,64,64 -> 256,32,32
        x = self.conv_block3(x)
        x = self.identity_block1(x)
        x = self.identity_block2(x)
        x = self.identity_block3(x)
        x = self.identity_block4(x)
        x = self.identity_block5(x)
        x = self.identity_block6(x)
        x = self.identity_block7(x)
        x = self.identity_block8(x)
        x = self.identity_block9(x)

        # 256,32,32 -> 128,64,64
        x = self.deconv_block1(x)
        # 128,64,64 -> 64,128,128
        x = self.deconv_block2(x)
        # 64,128,128 -> 3,128,128
        x = self.deconv_block3(x)
        return x


def main():
    from torchsummary import summary
    b, c, h, w = 1, 3, 128, 128
    x = torch.ones([b, c, h, w], dtype=torch.float32).cuda()
    mask = torch.ones([b, 1, h, w], dtype=torch.float32).cuda()
    model = get_resnet(h, w, c, beta=1).cuda()
    y_pred = model(x)
    print(y_pred.shape)
    summary(model, input_size=(c, h, w))  # 2,309,600


if __name__ == '__main__':
    main()
