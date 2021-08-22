import urllib
import requests


def query_unsplash(
    api_key: str,
    search_term: str,
    max_number_of_images: int = 30,
    page_number: int = 1
) -> dict:
    """
    Given a search term, queries pixabay for free-use images, returning less than
    or equal to the max_number_of_images.

    :param api_key: (str) Auth API key for pixabay service.
    :param search_term: (str) The search term you want to query pixabay for.
    :param max_number_of_images: (int) The maximum number of images you want returned.
    :param page_number: (int) The page of search results you want returned.
    :returns: (dict) Pixabay JSON response containing the image details.
    """

    url = "https://api.unsplash.com/search/photos"
    url_encoded_search_term = urllib.parse.quote_plus(search_term)
    params = {
        "client_id": api_key,
        "query": url_encoded_search_term,
        "per_page": max_number_of_images,
        "page": page_number
    }

    response = requests.get(url=url, params=params)   
    data = response.json()
    return data
    