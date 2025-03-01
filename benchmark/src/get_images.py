import os
import requests
from tqdm import tqdm

from bson_data import pins

image_dir = "../pinterest_data/images"
os.makedirs(image_dir, exist_ok=True)

def download_image(url, filename):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

# Download images
for pin in tqdm(pins, desc="Downloading Images"):
    if os.path.exists(os.path.join(image_dir, f"{pin['pin_id']}.jpg")):
        continue
    if "im_url" in pin:
        image_url = pin["im_url"]
        image_id = pin["pin_id"] 
        filename = os.path.join(image_dir, f"{image_id}.jpg")
        download_image(image_url, filename)
