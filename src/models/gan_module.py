import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from models.generator import Generator
from models.discriminator import Discriminator
from collections import OrderedDict
import wandb

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
        batch_size: int = 4,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.generator = Generator(latent_vector_size=latent_vector_size, generator_feature_size=generator_feature_size, n_channels=n_channels).to(device)
        self.discriminator = Discriminator(discriminator_feature_size=discriminator_feature_size, n_channels=n_channels).to(device)
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        self.validation_z = torch.randn(batch_size, latent_vector_size, 1, 1).to(next(self.generator.parameters()).device)
        self.example_input_array = self.validation_z
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
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def generator_step(self, batch: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(batch.size(0), 1)
        valid = valid.type_as(batch)

        # adversarial loss is binary cross-entropy
        g_prediction = self(noise)
        g_loss = self.adversarial_loss(self.discriminator(self(noise)), valid)
        tqdm_dict = {"generator_loss": g_loss}
        output = OrderedDict(
            {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        
        return output, g_loss

    def discriminator_step(self, batch: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        valid = torch.ones(batch.size(0), 1)
        valid = valid.type_as(batch)

        real_loss = self.adversarial_loss(self.discriminator(batch), valid)

        # how well can it label as fake?
        fake = torch.zeros(batch.size(0), 1)
        fake = fake.type_as(batch)

        fake_loss = self.adversarial_loss(
            self.discriminator(self(noise).detach()), fake
        )

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        tqdm_dict = {"discriminator_loss": d_loss}
        output = OrderedDict(
            {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output, d_loss


    def _step(self, batch, batch_idx, optimizer_idx, mode: str):
        """
        Templates the training, validation, test steps. `mode` is used
        to prefix the `self.log()` call

        """

        if optimizer_idx == None:
            optimizer_idx = 0
        
        noise = torch.randn(batch.shape[0], self.hparams.latent_vector_size, 1, 1)
        noise = noise.type_as(batch)

        # generator
        if optimizer_idx == 0:

            _, loss = self.generator_step(batch, noise)
            self.log(f'{mode}_gen_loss', loss)
            
        # discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            _, loss = self.discriminator_step(batch, noise)        
            self.log(f'{mode}_disc_loss', loss)

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._step(batch, batch_idx, optimizer_idx, mode='train')
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._step(batch, batch_idx, optimizer_idx, mode='val')
        return loss

    def test_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._step(batch, batch_idx, optimizer_idx, mode='test')
        return loss
            
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = self.hparams.beta1
        b2 = self.hparams.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1,b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1,b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        # log sampled images
        sample_imgs = self(self.validation_z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.log({"generated_images": [wandb.Image(grid, caption=f'Epoch: {self.current_epoch}\nValidation Images')]})

