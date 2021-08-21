import torch
from models.gan_module import GAN
from data.data_module import CelebDataModule
from pytorch_lightning import Trainer

data_dir = "data/raw/img_align_celeba"
batch_size = 4
n_channels = 3
latent_vector_size = 100
generator_feature_size = discriminator_feature_size = 64
ngpus = 0
num_workers = 0

datamodule = CelebDataModule(data_dir, batch_size=batch_size, num_workers=num_workers)
datamodule.setup()

gan_model = GAN(
    n_channels=n_channels,
    latent_vector_size=latent_vector_size,
    generator_feature_size=generator_feature_size,
    discriminator_feature_size=discriminator_feature_size,
    ngpus=ngpus,
    beta1=0.5,
    beta2=0.999
)
gan_model = gan_model.float()

trainer = Trainer(max_epochs=2)
trainer.fit(gan_model, datamodule)
print('Done, trained!')
