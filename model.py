import torch
import torch.nn as nn
import torch.nn.functional as F

class ResizeDeconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)  
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.LeakyReLU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x + res  

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU()

        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.deconv1 = ResizeDeconv(128, 64)
        self.deconv2 = ResizeDeconv(64, 32)

        self.final_conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        inputs = x
        conv1_out = self.conv1(x)
        conv1_out = self.bn1(conv1_out)
        conv1_out = self.act1(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = self.bn2(conv2_out)
        conv2_out = self.act2(conv2_out)
        conv3_out = self.conv3(conv2_out)
        conv3_out = self.bn3(conv3_out)
        conv3_out = self.act3(conv3_out)
        
        res_out = self.res_blocks(conv3_out)

        deconv1_out = self.deconv1(res_out, (256, 256))
        deconv2_out = self.deconv2(deconv1_out, (256, 256))
        deconv2_out += conv1_out  

        output = self.final_conv(deconv2_out)
        output += inputs
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=4, stride=2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=4, stride=2),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=1, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.f(x)



