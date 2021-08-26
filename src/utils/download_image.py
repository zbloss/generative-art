import os
import shutil
import requests


def download_image(
    image_url: str, prefix: str, filename: str, use_native_file_extension: bool = True
):
    """
    Given a image url, downloads the image to the specific prefix and filename.

    :param image_url: (str) URL of the image to download.
    :param prefix: (str) Directory to store the file in.
    :param filename: (str) Filename to save the file as.
    :param use_native_file_extension: (bool) use the provided filetype if available, overrides extension in filename.
    :returns: 200 if the operation completed successfully, else 500.
    """

    response_code = 200

    try:
        SUPPORTED_FILE_EXTENSIONS = ["png", "jpg", "jpeg"]
        native_file_extension = image_url.split(".")[-1]
        native_file_extension = (
            native_file_extension
            if native_file_extension in SUPPORTED_FILE_EXTENSIONS
            else None
        )

        file_extension = "png"
        if use_native_file_extension == True:
            if native_file_extension is not None:
                file_extension = native_file_extension

        if not os.path.exists(prefix):
            os.makedirs(prefix)

        filename = f"{filename}.{file_extension}"

        img_response = requests.get(image_url, stream=True)
        assert (
            img_response.status_code == 200
        ), f"Unable to download the image: {img_response.status_code}"

        with open(os.path.join(prefix, filename), "wb") as img_file:
            shutil.copyfileobj(img_response.raw, img_file)
            img_file.close()

    except Exception as e:
        response_code = 500
        pass

    return response_code
