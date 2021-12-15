from PIL import Image
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from glob import glob


def logger(name, ten):
    print(f"{name}: {ten}")


@hydra.main(config_path="../../configs", config_name="sol_ar_system_config")
def convert_rgba_to_rgb(cfg: DictConfig) -> None:

    png_image_files = glob(
        os.path.join(cfg.data.path_to_image_files, "**/*.png"), recursive=True
    )
    for png_image in tqdm(
        png_image_files,
        desc="Converting RGBA to RGB Images",
        total=len(png_image_files),
    ):
        image = Image.open(png_image)
        try:
            del image.info["transparency"]
        except KeyError:
            pass
        image.save(png_image)


if __name__ == "__main__":
    convert_rgba_to_rgb()
