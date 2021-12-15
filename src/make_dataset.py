import os
from utils.pixabay import query_pixabay
from utils.download_image import download_image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('PIXABAY_API_KEY')
SEARCH_TERM = 'galaxy'
CATEGORY = 'all'

data = query_pixabay(API_KEY, SEARCH_TERM, category=CATEGORY)
search_results = data['hits']

for idx, json in tqdm(enumerate(search_results), total=len(search_results)):
    try:
        image_url = json['largeImageURL']
        response = download_image(
            image_url, 
            'D:\\data\\generative-art\\data\\raw\\black-noise', 
            filename=f'{idx}_{idx}'
        )
    except:
        pass
