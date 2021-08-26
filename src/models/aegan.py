import os
import json
import sys
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from pytorch_lightning import LightningModule
from collections import OrderedDict
import wandb


EPS = 1e-6
ALPHA_RECONSTRUCT_IMAGE = 1
ALPHA_RECONSTRUCT_LATENT = 0.5
ALPHA_DISCRIMINATE_IMAGE = 0.005
ALPHA_DISCRIMINATE_LATENT = 0.1


class Generator(nn.Module):
    """A generator for mapping a latent space to a sample space.
    Input shape: (?, latent_vector_size)
    Output shape: (?, 3, 96, 96)
    """

    def __init__(
        self,
        latent_vector_size: int = 8,
        n_channels: int = 3,
        image_size: int = 128,
        projection_widths: list = [8, 8, 8, 8, 8, 8, 8],
        out_channels: int = 16,
    ):
        """Initialize generator.
        Args:
            latent_vector_size (int): latent dimension ("noise vector")
        """
        super().__init__()
        self.latent_vector_size = latent_vector_size
        self._init_modules()
        self.n_channels = n_channels
        self.image_size = image_size
        if type(self.image_size) == tuple:
            self.image_size = self.image_size[0]

        self.projection_widths = projection_widths
        self.out_channels = out_channels
        self._init_modules()

    def build_colorspace(self, input_dim: int, output_dim: int):
        """Build a small module for selecting colors."""
        colorspace = nn.Sequential(
            nn.Linear(input_dim, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim, bias=True),
            nn.Tanh(),
        )
        return colorspace

    def _init_modules(self):
        """Initialize the modules."""

        self.projection_dim = sum(self.projection_widths) + self.latent_vector_size
        self.projection = nn.ModuleList()
        for index, i in enumerate(self.projection_widths):
            self.projection.append(
                nn.Sequential(
                    nn.Linear(
                        self.latent_vector_size + sum(self.projection_widths[:index]),
                        i,
                        bias=True,
                    ),
                    nn.BatchNorm1d(8),
                    nn.LeakyReLU(),
                )
            )
        self.projection_upscaler = nn.Upsample(scale_factor=self.n_channels)

        self.colorspace_r = self.build_colorspace(
            self.projection_dim, self.out_channels
        )
        self.colorspace_g = self.build_colorspace(
            self.projection_dim, self.out_channels
        )
        self.colorspace_b = self.build_colorspace(
            self.projection_dim, self.out_channels
        )
        self.colorspace_upscaler = nn.Upsample(scale_factor=self.image_size)

        self.seed = nn.Sequential(
            nn.Linear(self.projection_dim, 512 * 3 * 3, bias=True),
            nn.BatchNorm1d(512 * 3 * 3),
            nn.LeakyReLU(),
        )

        self.upscaling = nn.ModuleList()
        self.conv = nn.ModuleList()

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(
            nn.Sequential(
                nn.ZeroPad2d((1, 1, 1, 1)),
                nn.Conv2d(
                    in_channels=(512) // 4,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
            )
        )

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(
            nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(
                    in_channels=(512 + self.projection_dim) // 4,
                    out_channels=256,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
            )
        )

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(
            nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(
                    in_channels=(256 + self.projection_dim) // 4,
                    out_channels=256,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
            )
        )

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(
            nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(
                    in_channels=(256 + self.projection_dim) // 4,
                    out_channels=256,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
            )
        ),

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(
            nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(
                    in_channels=(256 + self.projection_dim) // 4,
                    out_channels=64,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
        )

        self.upscaling.append(nn.Upsample(scale_factor=1))
        self.conv.append(
            nn.Sequential(
                nn.ZeroPad2d((2, 2, 2, 2)),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=self.out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Softmax(dim=1),
            )
        )

    def _apply_colorscaling(
        self,
        projection_tensor: torch.Tensor,
        intermediate_tensor: torch.Tensor,
        resize_dim: tuple = (-1, self.out_channels, 1, 1),
    ):

        # colorspaces
        r_space = self.colorspace_r(projection_tensor).view(resize_dim)
        g_space = self.colorspace_g(projection_tensor).view(resize_dim)
        b_space = self.colorspace_b(projection_tensor).view(resize_dim)

        # upscaling
        r_space = self.colorspace_upscaler(r_space)
        g_space = self.colorspace_upscaler(g_space)
        b_space = self.colorspace_upscaler(b_space)

        # applying intermediate tensor
        r_space = r_space * intermediate_tensor
        g_space = g_space * intermediate_tensor
        b_space = b_space * intermediate_tensor

        # Summing along the channel dimension
        r_space = torch.sum(r_space, dim=1, keepdim=True)
        g_space = torch.sum(g_space, dim=1, keepdim=True)
        b_space = torch.sum(b_space, dim=1, keepdim=True)

        output = torch.cat((r_space, g_space, b_space), dim=1)
        return output

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""

        last = input_tensor
        for module in self.projection:
            projection = module(last)
            last = torch.cat((last, projection), -1)

        projection = last

        intermediate = self.seed(projection)
        intermediate = intermediate.view((-1, 512, 3, 3))

        projection_2d = projection.view((-1, self.projection_dim, 1, 1))
        projection_2d = self.projection_upscaler(projection_2d)

        for i, (conv, upscaling) in enumerate(zip(self.conv, self.upscaling)):
            if i + 1 != len(self.upscaling):
                if i > 0:
                    intermediate = torch.cat((intermediate, projection_2d), 1)
                intermediate = torch.nn.functional.pixel_shuffle(
                    input=intermediate, upscale_factor=2
                )
            intermediate = conv(intermediate)
            projection_2d = upscaling(projection_2d)

        output = self._apply_colorscaling(
            projection_tensor=projection, intermediate_tensor=intermediate
        )

        return output


class Encoder(nn.Module):
    """An Encoder for encoding images as latent vectors.
    Input shape: (?, 3, 96, 96)
    Output shape: (?, latent_vector_size)
    """

    def __init__(
        self,
        device: str = "cpu",
        latent_vector_size: int = 8,
        down_channels: list = [3, 64, 128, 256, 512],
        up_channels: list = [256, 128, 64, 64, 64],
        final_down_channels: list = [64 + 3, 64, 64, 64, 64],
        scale_factors: list = [2, 2, 2, 1]
    ):
        """Initialize encoder.
        Args:
            device: chich GPU or CPU to use.
            latent_vector_size: output dimension
        """
        super().__init__()
        self.device = device
        self.latent_vector_size = latent_vector_size
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.final_down_channels = final_down_channels
        self.scale_factors = scale_factors

        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""

        self.down = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.down_channels[i],
                        out_channels=self.down_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.BatchNorm2d(self.down_channels[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.reducer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.down_channels[-1],
                out_channels=self.down_channels[-2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.down_channels[-2]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.up = nn.ModuleList()
        for i in range(len(self.up_channels) - 1):
            self.up.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.up_channels[i] + self.down_channels[-2 - i],
                        out_channels=self.up_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.BatchNorm2d(self.up_channels[i + 1]),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=self.scale_factors[i]),
                )
            )

        self.final_down_channels = [64 + 3, 64, 64, 64, 64]
        self.down_again = nn.ModuleList()
        for i in range(len(self.final_down_channels) - 1):
            self.down_again.append(
                nn.Conv2d(
                    in_channels=self.final_down_channels[i],
                    out_channels=self.final_down_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                )
            )
            self.down_again.append(nn.BatchNorm2d(self.final_down_channels[i + 1]))
            self.down_again.append(nn.LeakyReLU())

        self.projection = nn.Sequential(
            nn.Linear(
                512 * 6 * 6 + 64 * 6 * 6,
                256,
                bias=True,
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(
                256,
                128,
                bias=True,
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(
                128,
                self.latent_vector_size,
                bias=True,
            ),
        )

    def forward(self, input_tensor: torch.Tensor):
        """Forward pass; map latent vectors to samples."""

        #rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        
        rv = torch.randn(input_tensor.size()) * 0.02
        intermediate = input_tensor + rv
        intermediates = [intermediate]
        for module in self.down:
            intermediate = module(intermediate)
            intermediates.append(intermediate)

        intermediates = intermediates[:-1][::-1]

        down = intermediate.view(-1, 6 * 6 * 512)

        intermediate = self.reducer(intermediate)

        for index, module in enumerate(self.up):
            intermediate = torch.cat((intermediate, intermediates[index]), 1)
            intermediate = module(intermediate)

        intermediate = torch.cat((intermediate, input_tensor), 1)

        for module in self.down_again:
            intermediate = module(intermediate)

        intermediate = intermediate.view(-1, 6 * 6 * 64)
        intermediate = torch.cat((down, intermediate), -1)

        projected = self.projection(intermediate)

        return projected



class DiscriminatorImage(nn.Module):
    """A discriminator for discerning real from generated images.
    Input shape: (?, 3, 96, 96)
    Output shape: (?, 1)
    """

    def __init__(self, device="cpu", down_channels: list = [3, 64, 128, 256, 512]):
        """Initialize the discriminator."""
        super().__init__()
        self.device = device
        self.down_channels = down_channels
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""

        self.down = nn.ModuleList()
        leaky_relu = nn.LeakyReLU()
        for i in range(4):
            self.down.append(
                nn.Conv2d(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                )
            )
            self.down.append(nn.BatchNorm2d(self.down_channels[i + 1]))
            self.down.append(leaky_relu)

        self.classifier = nn.ModuleList()
        self.width = self.down_channels[-1] * 6 ** 2
        self.classifier.append(nn.Linear(self.width, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor: torch.Tensor):
        """Forward pass; map latent vectors to samples."""
        rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        intermediate = input_tensor + rv
        for module in self.down:
            intermediate = module(intermediate)
            rv = torch.randn(intermediate.size(), device=self.device) * 0.02 + 1
            intermediate *= rv

        intermediate = intermediate.view(-1, self.width)

        for module in self.classifier:
            intermediate = module(intermediate)

        return intermediate


class DiscriminatorLatent(nn.Module):
    """A discriminator for discerning real from generated vectors.
    Input shape: (?, latent_vector_size)
    Output shape: (?, 1)
    """

    def __init__(self, latent_vector_size=8, device="cpu", depth: int = 7, width: int = 8):
        """Initialize the Discriminator."""
        super().__init__()
        self.latent_vector_size = latent_vector_size
        self.device = device
        self.depth = depth
        self.width = width
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.pyramid = nn.ModuleList()
        for i in range(self.depth):
            self.pyramid.append(
                nn.Sequential(
                    nn.Linear(
                        self.latent_vector_size + self.width * i,
                        self.width,
                        bias=True,
                    ),
                    nn.BatchNorm1d(self.width),
                    nn.LeakyReLU(),
                )
            )

        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(self.depth * self.width + self.latent_vector_size, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor: torch.Tensor):
        """Forward pass; map latent vectors to samples."""

        last = input_tensor
        for module in self.pyramid:
            projection = module(last)
            rv = torch.randn(projection.size(), device=self.device) * 0.02 + 1
            projection *= rv
            last = torch.cat((last, projection), -1)
        
        for module in self.classifier:
            last = module(last)
        
        return last


class AEGAN(LightningModule):
    """An Autoencoder Generative Adversarial Network for making pokemon."""

    def __init__(
        self, 
        latent_vector_size, 
        noise_fn, 
        dataloader, 
        batch_size: int = 32, 
        device: str = "cpu",
        eps: float = 1e-6,
        image_reconstruction_alpha: float = 1.0,
        latent_reconstruction_alpha: float = 0.5,
        image_discriminator_alpha: float = 0.005,
        latent_discriminator_alpha: float = 0.1
    ):
        super().__init__()
        """Initialize the AEGAN.
        Args:
            latent_vector_size: latent-space dimension. Must be divisible by 4.
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
        """

        assert latent_vector_size % 4 == 0
        self.latent_vector_size = latent_vector_size
        self.device = device
        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size

        self.criterion_gen = nn.BCELoss()
        self.criterion_recon_image = nn.L1Loss()
        self.criterion_recon_latent = nn.MSELoss()

        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)

        self.generator = Generator(latent_vector_size=self.latent_vector_size)
        #self.generator = self.generator.to(self.device)
        self.optim_g = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-8
        )

        self.encoder = Encoder(
            latent_vector_size=self.latent_vector_size, device=self.device
        )
        #self.encoder = self.encoder.to(self.device)
        self.optim_e = optim.Adam(
            self.encoder.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-8
        )

        self.discriminator_image = DiscriminatorImage(device=self.device)
        #self.discriminator_image = self.discriminator_image.to(self.device)
        self.optim_di = optim.Adam(
            self.discriminator_image.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-8,
        )
        self.discriminator_latent = DiscriminatorLatent(
            latent_vector_size=self.latent_vector_size,
            device=self.device,
        )
        #self.discriminator_latent = self.discriminator_latent.to(self.device)
        self.optim_dl = optim.Adam(
            self.discriminator_latent.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-8,
        )

        self.validation_z = torch.randn(batch_size, latent_vector_size, 1, 1).to(
            next(self.generator.parameters()).device
        )
        self.image_reconstruction_alpha = image_reconstruction_alpha
        self.latent_reconstruction_alpha = latent_reconstruction_alpha
        self.image_discriminator_alpha = image_discriminator_alpha
        self.latent_discriminator_alpha = latent_discriminator_alpha

        self.example_input_array = self.validation_z
        self.save_hyperparameters()


    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.
        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.
        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None
        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        with torch.no_grad():
            samples = self.generator(self.validation_z)
        return samples

    def random_noise(self):
        return torch.randn((self.batch_size, self.latent_vector_size), device=device)

    def forward(self, noise: torch.Tensor):
        return self.generator(noise)

    def _generator_predictions(self, batch: torch.Tensor, noise: torch.Tensor):
        """
        Performs the initial generator & encoder prediction steps
        """

        noise_prediction = self(noise)
        batch_encoded = self.encoder(batch)
        batch_encoded_prediction = self(batch_encoded)
        noise_prediction_encoded = self.encoder(noise_prediction)

        return noise_prediction, batch_encoded, batch_encoded_prediction, noise_prediction_encoded


    def _generator_step(self, batch: torch.Tensor, noise: torch.Tensor):
        """
        Computes the generator and encoder losses
        """

        # Generator & Encoder predictions
        noise_prediction, \
        batch_encoded, \
        batch_encoded_prediction, \
        noise_prediction_encoded = self._generator_predictions(batch, noise)

        # Discriminator predictions
        noise_prediction_confidence = self.discriminator_image(noise_prediction)
        batch_encoded_confidence = self.discriminator_latent(batch_encoded)
        batch_encoded_prediction_confidence = self.discriminator_image(batch_encoded_prediction)
        noise_prediction_encoded_confidence = self.discriminator_latent(noise_prediction_encoded)

        # Computing losses
        noise_prediction_loss = self.criterion_gen(noise_prediction_confidence, self.target_ones)
        batch_encoded_loss = self.criterion_gen(batch_encoded_confidence, self.target_ones)
        batch_encoded_prediction_loss = self.criterion_gen(batch_encoded_prediction_confidence, self.target_ones)
        noise_prediction_encoded_loss = self.criterion_gen(noise_prediction_encoded_confidence, self.target_ones)

        # Image reconstruction loss
        batch_reconstruction_loss = self.criterion_recon_image(batch_encoded_prediction, batch) * self.image_reconstruction_alpha
        noise_reconstruction_loss = self.criterion_recon_latent(noise_prediction_encoded, noise) * self.latent_reconstruction_alpha

        # Combine loss metrics
        batch_loss = ((noise_prediction_loss + batch_encoded_prediction_loss) / 2) * self.image_discriminator_alpha
        noise_loss = ((batch_encoded_loss + noise_prediction_encoded_loss) / 2) * self.latent_discriminator_alpha

        return batch_reconstruction_loss, noise_reconstruction_loss, batch_loss, noise_loss

    def _discriminator_step(self, batch: torch.Tensor, noise: torch.Tensor):
        """
        Computes the discriminator steps

        """
        with torch.no_grad():

            # Generator & Encoder predictions
            noise_prediction, \
            batch_encoded, \
            batch_encoded_prediction, \
            noise_prediction_encoded = self._generator_predictions(batch, noise)

        batch_confidence = self.discriminator_image(batch)
        noise_prediction_confidence = self.discriminator_image(noise_prediction)
        batch_encoded_prediction_confidence = self.discriminator_image(batch_encoded_prediction)
        
        noise_confidence = self.discriminator_latent(noise)
        batch_encoded_condfidence = self.discriminator_latent(batch_encoded)
        noise_prediction_encoded_confidence = self.discriminator_latent(noise_prediction_encoded)

        # Overall losses
        batch_loss = 2 * self.criterion_gen(batch_confidence, self.target_ones)
        noise_prediction_loss = self.criterion_gen(noise_prediction_confidence, self.target_zeros)
        batch_encoded_prediction_loss = self.criterion_gen(batch_encoded_condfidence, self.target_zeros)

        noise_loss = 2 * self.criterion_gen(noise_confidence, self.target_ones)
        batch_encoded_loss = self.criterion_gen(batch_encoded_condfidence, self.target_zeros)
        noise_prediction_encoded_loss = self.criterion_gen(noise_prediction_encoded_confidence, self.target_zeros)
                    
        loss_images = (batch_loss + noise_prediction_loss + batch_encoded_prediction_loss) / 4
        loss_latent = (noise_loss + batch_encoded_loss + noise_prediction_encoded_confidence) / 4
        return loss_images, loss_latent
        

    def generator_step(self, batch: torch.Tensor, mode: str):
        
        noise = self.random_noise()

        # Generator Step
        batch_reconstruction_loss, noise_reconstruction_loss, batch_loss, noise_loss = self._generator_step(batch, noise)
        losses = torch.stack([batch_reconstruction_loss, batch_reconstruction_loss, batch_loss, noise_loss])
        loss = torch.sum(losses)

        self.log(f'{mode}_batch_reconstruction_loss', batch_reconstruction_loss)
        self.log(f'{mode}_noise_reconstruction_loss', noise_reconstruction_loss)
        self.log(f'{mode}_batch_loss', batch_loss)
        self.log(f'{mode}_noise_loss', noise_loss)
        self.log(f'{mode}_generator_loss', loss)

        return loss

    def discriminator_step(self, batch: torch.Tensor, mode: str):
        
        noise = self.random_noise()
        loss_images, loss_latent = self._discriminator_step(batch, noise)
        loss = loss_images + loss_latent

        self.log(f'{mode}_loss_images', loss_images)
        self.log(f'{mode}_loss_latent', loss_latent)
        self.log(f'{mode}_discriminator_loss', loss)

        return loss


    def _step(self, batch, batch_idx, optimizer_idx, mode: str):
        """
        Templates the training, validation, test steps. `mode` is used
        to prefix the `self.log()` call

        """

        if optimizer_idx == None:
            optimizer_idx = 0


        # generator
        if optimizer_idx == 0:

            loss = self.generator_step(batch, mode)

        # discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(batch, mode)
            
        return loss
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._step(batch, batch_idx, optimizer_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._step(batch, batch_idx, optimizer_idx, mode="val")
        return loss

    def test_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._step(batch, batch_idx, optimizer_idx, mode="test")
        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = self.hparams.beta1
        b2 = self.hparams.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        # log sampled images
        sample_imgs = self(self.validation_z)[:8]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.log(
            {
                "generated_images": [
                    wandb.Image(
                        grid, caption=f"Epoch: {self.current_epoch}\nValidation Images"
                    )
                ]
            }
        )