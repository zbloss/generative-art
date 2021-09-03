import torch
from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader

loader = ModelLoader(
    base_dir = 'D:\models\generative-art\monsters\monsters-stylegan2',   # path to where you invoked the command line tool
    name = 'monsters-gan\default',                   # the project name, defaults to 'default'
    load_from = 150
)

for i in range(1, 1000):
    loader.model.generate_interpolation(num=i, num_image_tiles=1)