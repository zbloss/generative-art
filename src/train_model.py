import time
import torch
from models.aegan import AEGAN
from models.gan_module import GAN
from data.monsters_data_module import MonstersDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf


def logger(name, ten):
    print(f"{name}: {ten}")


@hydra.main(config_path="../configs", config_name="monsters_config")
def train_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    datamodule = MonstersDataModule(
        data_dir=cfg.data.path_to_image_files,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=(cfg.data.image_height, cfg.data.image_width),
    )

    datamodule.setup()
    gan_model = AEGAN(
        latent_vector_size=cfg.model.latent_vector_size,
        batch_size=cfg.data.batch_size,
        image_size=96
    )
    gan_model = gan_model.float()
            
    wandb_logger = WandbLogger(
        project=cfg.model.model_name,
        log_model=True,
    )

    callbacks = []
    # callbacks.append(
    #     EarlyStopping(
    #         monitor=cfg.early_stop.metric,
    #         min_delta=cfg.early_stop.minimum_delta,
    #         patience=cfg.early_stop.patience,
    #         verbose=False,
    #         mode=cfg.early_stop.mode,
    #     )
    # )

    trainer_params = dict(cfg.trainer)
    trainer_params["logger"] = wandb_logger
    trainer_params["callbacks"] = callbacks
    # trainer_params[
    #     "resume_from_checkpoint"
    # ] = "D:/models/generative-art/monsters/monsters-gan/2e5zkykn/checkpoints/epoch=29-step=14489.ckpt"
    print(trainer_params)
    trainer = Trainer(**trainer_params)
    wandb_logger.watch(gan_model)

    trainer.fit(gan_model, datamodule)
    print("Trained!")

    trainer.save_checkpoint("example.ckpt")
    print("Saved!")


if __name__ == "__main__":
    train_model()
