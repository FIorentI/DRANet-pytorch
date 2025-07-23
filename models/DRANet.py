import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import functools
from torch.nn.utils import spectral_norm
from .batchinstancenorm import BatchInstanceNorm2d as Normlayer


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                bin(filters)
            )

    def forward(self, x):
        output = self.main(x)
        output += self.shortcut(x)
        return output


class Encoder(nn.Module):
    def __init__(self, channels=1):
        super(Encoder, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.model(x)


class Separator(nn.Module):
    def __init__(self, converts, ch=64):
        super(Separator, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
        )
        self.w = nn.ParameterDict()
        self.ch = ch
        self.converts = converts  # Store for possible use later

    def forward(self, features, converts=None):
        contents, styles = dict(), dict()

        # Separate style/content
        for key in features.keys():
            styles[key] = self.conv(features[key])         # Style
            contents[key] = features[key] - styles[key]    # Content
            if '2' in key:
                source, target = key.split('2')
                contents[target] = contents[key]

        # Lazy init and apply conversion
        if converts is not None:
            for cv in converts:
                source, target = cv.split('2')

                # Check if parameter exists
                if cv not in self.w:
                    feat_shape = contents[source].shape  # [B, C, H, W]
                    self.w[cv] = nn.Parameter(
                        torch.ones(1, self.ch, feat_shape[2], feat_shape[3], device=contents[source].device),
                        requires_grad=True
                    )

                contents[cv] = self.w[cv] * contents[source]

        return contents, styles



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            spectral_norm(nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.Tanh()
        )

    def forward(self, content, style):
        return self.model(content + style)


class Classifier(nn.Module):
    def __init__(self, channels=1, num_classes=10):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(48, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        old_conv = features[0]
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.bias[:] = old_conv.bias
        features[0] = new_conv

        self.to_relu_1_1 = nn.Sequential(*features[:2])
        self.to_relu_2_1 = nn.Sequential(*features[2:7])
        self.to_relu_3_1 = nn.Sequential(*features[7:12])
        self.to_relu_4_1 = nn.Sequential(*features[12:21])
        self.to_relu_4_2 = nn.Sequential(*features[21:25])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_4_2(h)
        h_relu_4_2 = h
        return (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_4_2)


class Discriminator_USPS(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator_USPS, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Discriminator_MNIST(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator_MNIST, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class PatchGAN_Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(PatchGAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)
