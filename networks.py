import os

import numpy as np
import torch.nn as nn
import torch
from numba.core.debuginfo import DIBuilder
from timm.models.vision_transformer import Block
from timm.models.xception import Xception
from torchsummary.torchsummary import summary


class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out


class DeeplabV3plusTransformer(nn.Module):
    def __init__(self, dim, depth):
        super(DeeplabV3plusTransformer, self).__init__()
        self.dim = dim
        self.backbone = Xception(num_classes=1, in_chans=1)
        self.backbone.conv3 = self.backbone.bn3 = self.backbone.act3 = self.backbone.conv4 = self.backbone.bn4 = self.backbone.act4 = self.backbone.global_pool = self.backbone.fc = nn.Identity()

        self.low_level_block = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.act1,
            self.backbone.conv2,
            self.backbone.bn2,
            self.backbone.act2,
            self.backbone.block1,
            self.backbone.block2,
        )
        self.deep_level_block = nn.Sequential(
            self.backbone.block3,
            self.backbone.block4,
            self.backbone.block5,
            self.backbone.block6,
            self.backbone.block7,
            self.backbone.block8,
            self.backbone.block9,
            self.backbone.block10,
            self.backbone.block11,
            self.backbone.block12
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(128)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.upsample1 = nn.ConvTranspose2d(256, 256, 3, 4, padding=0, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(64, 1, 3,  4, 0, 1)
        self.upsample3 = nn.ConvTranspose2d(1, 1, 1, 2, padding=0, output_padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.attn1 = nn.Sequential(
            *[
                Block(dim, 16)
                for i in range(depth)
            ]
        )
        self.attn2 = nn.Sequential(
            *[
                Block(dim, 8)
                for i in range(depth)
            ]
        )
        self.attn3 = nn.Sequential(
            *[
                Block(dim, 4)
                for i in range(depth)
            ]
        )
        self.attn4 = nn.Sequential(*[
            Block(dim, 2)
            for i in range(depth)
        ])
        self.attn5 = nn.Sequential(*[
            Block(dim, 1)
            for i in range(depth)
        ])

    def forward(self, x):
        low_level_feature = self.low_level_block(x)
        deep_level_feature = self.deep_level_block(low_level_feature)
        b, c, h, w = deep_level_feature.size()
        feature_embedding = deep_level_feature.reshape(b, c, -1).transpose(-2, -1)
        x = self.attn1(feature_embedding)
        attentionx = x
        deep_level_feature = attentionx.transpose(-2, -1).reshape(b, self.dim, h, w)
        deep_level_feature = self.conv2(deep_level_feature)
        out = self.relu(self.upsample1(deep_level_feature))
        low_level_feature = self.conv3(low_level_feature)
        out = torch.cat((out, low_level_feature), dim=1)
        out = self.conv4(out)
        out = self.relu(self.upsample2(out))
        out = self.relu(self.upsample3(out))
        return out

class Class1(nn.Module):
    def __init__(self, out_ch=96):
        super(Class1, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out

class L1Block(nn.Module):
    def __init__(self, in_channel, out_channel=64):
        super(L1Block, self).__init__()
        self.conv1 = ResidualBlock(in_channel, out_channel, 3)
        self.conv2 = ResidualBlock(out_channel, out_channel*2, 4)
        self.conv3 = ResidualBlock(out_channel*2, out_channel*4, 4)
        self.conv4 = ResidualBlock(out_channel*4, out_channel*8, 3)
        self.pool1 = nn.Sequential(nn.MaxPool2d(2, 2),
                                   nn.Conv2d(in_channel, out_channel, 1, 1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))
        self.pool2 = nn.Sequential(nn.MaxPool2d(2, 2),
                                   nn.Conv2d(out_channel, out_channel*2, 1, 1),
                                   nn.BatchNorm2d(out_channel*2),
                                   nn.ReLU(inplace=True))
        self.pool3 = nn.Sequential(nn.MaxPool2d(2, 2),
                                   nn.Conv2d(out_channel*2, out_channel*4, 1, 1),
                                   nn.BatchNorm2d(out_channel*4),
                                   nn.ReLU(inplace=True))

        self.up_sample1 = nn.ConvTranspose2d(out_channel*8, out_channel*4, 3, 2, padding=1, output_padding=1)
        self.up_sample2 = nn.ConvTranspose2d(out_channel*4, out_channel*2, 3, 2, padding=1, output_padding=1)
        self.up_sample3 = nn.ConvTranspose2d(out_channel*2, out_channel, 3, 2, padding=1, output_padding=1)
        self.up_sample4 = nn.ConvTranspose2d(out_channel, in_channel, 3, 2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        f1 = x
        f2 = self.conv1(f1)
        f3 = self.conv2(f2)
        f4 = self.conv3(f3)
        f5 = self.conv4(f4)
        pool1 = self.pool1(f1)
        pool2 = self.pool2(f2)
        pool3 = self.pool3(f3)
        u1 = self.relu(self.up_sample1(f5) + f4)
        u2 = self.relu(self.up_sample2(u1) + f3)
        u3 = self.relu(self.up_sample3(u2) + f2)
        output = self.relu(self.up_sample4(u3))
        # output = output + x
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, depth):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 2, padding=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(inplace=True))

        self.conv_list = nn.Sequential(*[self.conv2 for _ in range(depth-1)])
        self.relu = nn.ReLU()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv_list(x1)
        return x1 + x2

class L1Net(nn.Module):
    def __init__(self, in_channel, out_channel, depth):
        super(L1Net, self).__init__()
        self.blocks = nn.Sequential(*[L1Block(in_channel, out_channel) for _ in range(depth)])


    def forward(self, x):
        x = self.blocks(x)
        return x


class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilationBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, dilation=8, padding=14),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, dilation=4, padding=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels*3, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(x)
        o3 = self.conv3(x)
        o = torch.cat((o1, o2, o3), dim=1)
        o = self.conv4(o)
        return o
class DilationNet(nn.Module):
    def __init__(self, in_channels, dim):
        super(DilationNet, self).__init__()
        self.block1 = DilationBlock(in_channels, dim)
        self.block2 = DilationBlock(dim, dim)
        self.block3 = DilationBlock(dim, dim)

        self.u1 = nn.ConvTranspose2d(dim, dim, 5, 1)
        self.u2 = nn.ConvTranspose2d(dim, dim, 5, 1)
        self.u3 = nn.ConvTranspose2d(dim, 1, 5, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        f1 = x
        x = self.block1(x)
        f2 = x
        x = self.block2(x)
        f3 = x
        x = self.block3(x)
        x = self.relu(self.u1(x))
        x = x + f3
        x = self.relu(self.u2(x))
        x = x + f2
        x = self.relu(self.u3(x))
        x = x + f1
        return self.relu(x)





if __name__ == '__main__':
    model = L1Net(1, 256, 1)
    # model = DilationNet(1, 64)
    # model = DeeplabV3plusTransformer(1024, 12)
    # model = RED_CNN(96)
    # model = ResidualBlock(1, 64, 4)
    print(summary(model, (1, 512, 512), device='cpu'))

