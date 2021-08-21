import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from models.generator import Generator
from models.discriminator import Discriminator
from collections import OrderedDict


class GAN(LightningModule):
    def __init__(
        self,
        n_channels: int,
        latent_vector_size: int,
        generator_feature_size: int,
        discriminator_feature_size: int,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.5,
        ngpus: int = 0
    ):
        super().__init__()
        
        self.generator = Generator(ngpu=ngpus, latent_vector_size=latent_vector_size, generator_feature_size=generator_feature_size, n_channels=n_channels)
        self.discriminator = Discriminator(ngpu=ngpus, discriminator_feature_size=discriminator_feature_size, n_channels=n_channels)
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        self.save_hyperparameters()


    def forward(self, noise):
        return self.generator(noise)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def adversarial_loss(self, y_hat, y):
        y_hat = y_hat.reshape(y.shape)
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch

        # sample noise
        noise = torch.randn(imgs.shape[0], self.hparams.latent_vector_size, 1, 1)
        noise = noise.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(noise)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_prediction = self(noise)
            g_loss = self.adversarial_loss(self.discriminator(self(noise)), valid)
            tqdm_dict = {"generator_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(noise).detach()), fake
            )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"discriminator_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = self.hparams.beta1
        b2 = self.hparams.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1,b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1,b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
