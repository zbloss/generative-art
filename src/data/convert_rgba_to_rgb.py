from PIL import Image
import os
import hydra
from omegaconf import DictConfig, OmegaConf

def logger(name, ten):
    print(f'{name}: {ten}')

@hydra.main(config_path="../configs", config_name="monsters_config")
def convert_rgba_to_rgb(cfg : DictConfig) -> None:



src = "./resizedData"
dst = "./resized_black/"

for each in os.listdir(src):
    png = Image.open(os.path.join(src,each))
    # print each
    if png.mode == 'RGBA':
        png.load() # required for png.split()
        background = Image.new("RGB", png.size, (0,0,0))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
    else:
        png.convert('RGB')
        png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')