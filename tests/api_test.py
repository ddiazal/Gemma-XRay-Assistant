import json
import pathlib
import requests

PATH: str = "data"
image_name: str = ""
IMAGE_FILE_PATH: pathlib.PosixPath = pathlib.Path(PATH, image_name)

with open(IMAGE_FILE_PATH, "rb") as buffer:
    image = buffer.read()

data: dict[str, str] = {"image": image, "ground_truth": ""}

api_endpoint: str = ""
