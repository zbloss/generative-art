import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, discriminator_feature_size: int, n_channels: int = 3):
        super(Discriminator, self).__init__()

        self.discriminator_feature_size = discriminator_feature_size
        self.n_channels = n_channels

        self.conv = nn.Conv2d(
            self.n_channels, 
            self.discriminator_feature_size, 4, 2, 1, bias=False
        )
        self.conv1 = nn.Conv2d(
            self.discriminator_feature_size,
            self.discriminator_feature_size * 2,
            4,
            2,
            1,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(self.discriminator_feature_size * 2)
        
        self.conv2 = nn.Conv2d(
            self.discriminator_feature_size * 2,
            self.discriminator_feature_size * 4,
            4,
            2,
            1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(self.discriminator_feature_size * 4)
        
        self.conv3 = nn.Conv2d(
            self.discriminator_feature_size * 4,
            self.discriminator_feature_size * 8,
            4,
            2,
            1,
            bias=False,
        )
        self.batch_norm3 = nn.BatchNorm2d(self.discriminator_feature_size * 8)

        self.conv4 = nn.Conv2d(self.discriminator_feature_size * 8, 1, 4, 1, 0, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.leaky_relu(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)

        return x
