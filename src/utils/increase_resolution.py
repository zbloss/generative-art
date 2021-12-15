import os
import argparse
from glob import glob
import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN
from tqdm import tqdm

ISR_MODELS = [
    'psnr-large',
    'psnr-small',
    'noise-cancel',
    'gans'
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CLI to increase image resolution')
    parser.add_argument(
        '--path-to-files', 
        help="takes either a single file or a directory.",
        type=str,
        required=True
    )
    parser.add_argument(
        '--path-to-export', 
        help='where you want to save the files.',
        type=str,
        required=True
    )
    parser.add_argument(
        '--image-file-type',
        help='filetype to expect',
        default='jpg',
        required=False,
        type=str
    )
    parser.add_argument(
        '--isr-model',
        help='which ISR model you want to use.',
        default='psnr-large',
        required=False,
        type=str
    )

    args, uargs = parser.parse_known_args()
    if os.path.isdir(args.path_to_files):
        files = glob(os.path.join(args.path_to_files, f'*.{args.image_file_type}'))
    else:
        files = [args.path_to_files]

    if args.isr_model not in ISR_MODELS:
        raise Exception(f"Invalid ISR Model passed to --isr-model: {args.isr_model}")

    if not os.path.exists(args.path_to_export):
        os.makedirs(args.path_to_export)

    print(f'args.isr_model: {args.isr_model}')

    if args.isr_model == 'gans':
        model = RRDN(weights=args.isr_model)
    else:
        model = RDN(weights=args.isr_model)

    for file in tqdm(files):

        file_directory, filename = os.path.split(file)

        base_img = Image.open(file)
        img_array = np.array(base_img)
        img_array = model.predict(img_array)
        isr_img = Image.fromarray(img_array)
        isr_img.save(os.path.join(args.path_to_export, filename))
        print(f'Enhanced {filename}')


