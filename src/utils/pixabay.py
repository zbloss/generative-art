import urllib
import requests


def query_pixabay(
    api_key: str,
    search_term: str,
    max_number_of_images: int = 200,
    category: str = "animals",
) -> dict:
    """
    Given a search term, queries pixabay for free-use images, returning less than
    or equal to the max_number_of_images.

    :param api_key: (str) Auth API key for pixabay service.
    :param search_term: (str) The search term you want to query pixabay for.
    :param max_number_of_images: (int) The maximum number of images you want returned.
    :param category: (str) Optional photo category to search in.
    :returns: (dict) Pixabay JSON response containing the image details.
    """

    url = "https://pixabay.com/api/"
    url_encoded_search_term = urllib.parse.quote_plus(search_term)
    params = {
        "key": api_key,
        "q": url_encoded_search_term,
        "category": category,
        "image_type": "photo",
        "safesearch": "true",
        "per_page": max_number_of_images,
    }

    response = requests.get(url=url, params=params)
    data = response.json()
    return data
