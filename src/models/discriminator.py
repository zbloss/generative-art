import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, discriminator_feature_size: int, n_channels: int = 3):
        super(Discriminator, self).__init__()

        self.discriminator_feature_size = discriminator_feature_size
        self.n_channels = n_channels
        
        out_1 = discriminator_feature_size
        out_2 = int(out_1 * 2)
        out_3 = int(out_2 * 2)
        out_4 = int(out_3 * 2)
        out_5 = int(out_4 * 2)
        out_6 = int(out_5 / 2)
        out_7 = int(out_6 / 2)
        out_8 = 1

        self.conv1 = nn.Conv2d(
            in_channels=n_channels, 
            out_channels=out_1,
            kernel_size=(4,4),
            stride=(2,2),
            padding=(1,1),
            bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(out_1)

        self.conv2 = nn.Conv2d(
            in_channels=out_1, 
            out_channels=out_2,
            kernel_size=(4,4),
            stride=(2,2),
            padding=(1,1),
            bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(out_2)

        self.conv3 = nn.Conv2d(
            in_channels=out_2, 
            out_channels=out_3,
            kernel_size=(4,4),
            stride=(2,2),
            padding=(1,1),
            bias=False
        )
        self.batch_norm3 = nn.BatchNorm2d(out_3)

        self.conv4 = nn.Conv2d(
            in_channels=out_3, 
            out_channels=out_4,
            kernel_size=(4,4),
            stride=(2,2),
            padding=(1,1),
            bias=False
        )
        self.batch_norm4 = nn.BatchNorm2d(out_4)

        self.conv5 = nn.Conv2d(
            in_channels=out_4, 
            out_channels=out_5,
            kernel_size=(4,4),
            stride=(2,2),
            padding=(1,1),
            bias=False
        )
        self.batch_norm5 = nn.BatchNorm2d(out_5)

        self.conv6 = nn.Conv2d(
            in_channels=out_5, 
            out_channels=out_6,
            kernel_size=(4,4),
            stride=(2,2),
            padding=(1,1),
            bias=False
        )
        self.batch_norm6 = nn.BatchNorm2d(out_6)

        self.conv7 = nn.Conv2d(
            in_channels=out_6, 
            out_channels=out_7,
            kernel_size=(4,4),
            stride=(2,2),
            padding=(1,1),
            bias=False
        )
        self.batch_norm7 = nn.BatchNorm2d(out_7)

        self.conv8 = nn.Conv2d(
            in_channels=out_7, 
            out_channels=out_8,
            kernel_size=(4,4),
            stride=(2,2),
            padding=(1,1),
            bias=False
        )
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
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
        x = self.batch_norm4(x)
        x = self.leaky_relu(x)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.leaky_relu(x)

        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = self.leaky_relu(x)

        x = self.conv7(x)
        x = self.batch_norm7(x)
        x = self.leaky_relu(x)

        x = self.conv8(x)
        return x
