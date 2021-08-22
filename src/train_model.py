import time
import torch
from models.gan_module import GAN
from data.data_module import CelebDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf
from models.wandb_image_callback import WandbImageCallback


@hydra.main(config_path="../configs", config_name="base_config")
def train_model(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    datamodule = CelebDataModule(data_dir=cfg.data.path_to_image_files, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    datamodule.setup()

    gan_model = GAN(
        n_channels=cfg.data.n_channels,
        latent_vector_size=cfg.model.latent_vector_size,
        generator_feature_size=cfg.model.generator_feature_size,
        discriminator_feature_size=cfg.model.discriminator_feature_size,
        beta1=cfg.model.beta1,
        beta2=cfg.model.beta2,
        batch_size=cfg.data.batch_size,
        device='cuda' if cfg.trainer.gpus>0 else 'cpu'
    )
    gan_model = gan_model.float()
    tb_logger = TensorBoardLogger(
        'tb_logs', 
        name=cfg.model.model_name, 
        version=time.strftime('%Y%m%dT%H%M%S'),
        log_graph=cfg.data.log_graph
    )

    wandb_logger = WandbLogger(
        project=cfg.model.model_name, 
        log_model="all", 
    )

    callbacks = []
    callbacks.append(
        EarlyStopping(
            monitor=cfg.early_stop.metric, 
            min_delta=cfg.early_stop.minimum_delta, 
            patience=cfg.early_stop.patience, 
            verbose=False, 
            mode=cfg.early_stop.mode
        )
    )
    # callbacks.append(
    #     WandbImageCallback(gan_model.validation_z)
    # )

    trainer_params = dict(cfg.trainer)
    trainer_params['logger'] = wandb_logger
    trainer_params['callbacks'] = callbacks
    trainer = Trainer(**trainer_params)
    wandb_logger.watch(gan_model)

    trainer.fit(gan_model, datamodule)
    print('Trained!')

    trainer.save_checkpoint("example.ckpt")
    print('Saved!')

if __name__ == "__main__":
    train_model()