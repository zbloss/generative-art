import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        ngpu,
        latent_vector_size: int,
        generator_feature_size: int,
        n_channels: int = 3,
    ):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.latent_vector_size = latent_vector_size
        self.generator_feature_size = generator_feature_size
        self.n_channels = n_channels

        out_1 = self.generator_feature_size * 8
        out_2 = int(out_1 / 2)
        out_3 = int(out_2 / 2)
        out_4 = int(out_3 / 2)

        self.conv1 = nn.ConvTranspose2d(
            in_channels=latent_vector_size, 
            out_channels=out_1, 
            kernel_size=(4,4), 
            stride=(1,1), 
            padding=0,
            bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(out_1)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=out_1, 
            out_channels=out_2, 
            kernel_size=(4,4), 
            stride=(2,2), 
            padding=(1,1),
            bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(out_2)

        self.conv3 = nn.ConvTranspose2d(
            in_channels=out_2, 
            out_channels=out_3, 
            kernel_size=(4,4), 
            stride=(2,2), 
            padding=(1,1),
            bias=False
        )
        self.batch_norm3 = nn.BatchNorm2d(out_3)

        self.conv4 = nn.ConvTranspose2d(
            in_channels=out_3, 
            out_channels=out_4, 
            kernel_size=(4,4), 
            stride=(2,2), 
            padding=(1,1),
            bias=False
        )
        self.batch_norm4 = nn.BatchNorm2d(out_4)

        self.conv5 = nn.ConvTranspose2d(
            in_channels=out_4, 
            out_channels=self.n_channels, 
            kernel_size=(4,4), 
            stride=(2,2), 
            padding=(1,1), 
            bias=False
        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.tanh(x)

        return x
