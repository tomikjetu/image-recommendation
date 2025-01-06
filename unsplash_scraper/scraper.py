# https://medium.com/@yahyamrafe202/scraping-unsplash-images-api-vs-web-scraping-a-comparative-guide-0aa8955cb605

import os
import httpx
from PIL import Image
from io import BytesIO

def API_request(keyword, per_page, num_pages):
    all_results = []
    for page in range(1, num_pages + 1):
        link = f"https://unsplash.com/napi/search/photos?page={page}&per_page={per_page}&query={keyword}"
        resp = httpx.get(link)
        if resp.status_code == 200:
            page_results = resp.json().get('results', [])
            all_results.extend(page_results)
        else:
            print(f"Error fetching page {page}: Status code {resp.status_code}")
            break  # Exit the loop if there's an error
    return all_results

def extract_raw_urls(response_json, skip_premium=True):
    urls = []
    for image_data in response_json:
        if skip_premium and image_data.get('premium', False):
            continue
        raw_url = image_data['urls']['raw']
        trimmed_url = raw_url.split('?')[0]
        urls.append((image_data['id'], trimmed_url))
    return urls

directory = f"../application/storage/downloaded"
max_width = 800
def download_images(keyword, urls):
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f"Starting download of {len(urls)} images for {keyword}...")
    for idx, (image_id, url) in enumerate(urls, start=1):
        try:
            image_response = httpx.get(url)
            if image_response.status_code == 200:
                image = Image.open(BytesIO(image_response.content))

                # Resize image if width is greater than max_width
                if image.width > max_width:
                    ratio = max_width / float(image.width)
                    new_height = int(float(image.height) * ratio)
                    image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

                image_path = os.path.join(directory, f"{keyword}_{image_id}.jpg")
                image.save(image_path, "JPEG")
                print(f"{idx}. Downloaded and resized {image_id} to {image_path}")
            else:
                print(f"{idx}. Failed to download image {image_id}")
        except Exception as e:
            print(f"{idx}. Error downloading or processing image {image_id}: {e}")


keywords = [
    # Animals
    "cat", "dog", "bird", "lion", "tiger", 
    "horse", "shark", "whale", "elephant", "penguin", 
    "panda", "bear", "dolphin", "fish", "parrot", 
    "wolf", "fox", "zebra", "koala", "kangaroo", 
    "turtle", "crocodile", "flamingo", "owl", "chameleon",

    # Cars and Vehicles
    "sportscar", "luxury car", "SUV", "motorcycle", "truck", 
    "racing car", "electric car", "convertible", "yacht", 
    "airplane", "train", "helicopter",

    # Home and Interior
    "interior", "house", "garden", "front yard", "kitchen", 
    "living room", "bedroom", "bathroom", "office space", 
    "balcony", "patio", "pool", "apartment", "skyscraper",

    # Nature and Landscapes
    "beach", "mountain", "forest", "desert", "waterfall", 
    "river", "lake", "island", "volcano", "glacier", 
    "canyon", "meadow", "savannah", "cliff", "cave", 
    "coral reef", "snowfield", "fjord",

    # Sports and Activities
    "soccer", "basketball", "tennis", "baseball", "swimming", 
    "cycling", "running", "skiing", "surfing", "yoga", 
    "hiking", "camping", "skateboarding", "fishing", "golf",

    # Food and Cuisine
    "pizza", "burger", "sushi", "pasta", "ice cream", 
    "salad", "coffee", "tea", "cake", "bread", 
    "fruit basket", "vegetables", "barbecue", "street food", "wine",

    # Technology and Gadgets
    "smartphone", "laptop", "smartwatch", "drone", "robot", 
    "gaming console", "keyboard", "circuit board", 
    "satellite dish", "solar panel", "3D printer", "electric scooter",

    # Art and Culture
    "painting", "sculpture", "museum", "theater", 
    "photography", "street art", "calligraphy", "literature", 
    "architecture", "pottery", "music instrument",
    "festival", "carnival"
]

if __name__ == "__main__":
    for keyword in keywords:
        num_pages = 2  # Number of pages you want to fetch
        per_page = 10  # Number of images per page
        all_results = API_request(keyword, per_page, num_pages)
        if all_results:  # Ensure we have results
            urls = extract_raw_urls(all_results)  # Extract URLs, skipping premium images
            if urls:
                print(f"Total images to be downloaded (excluding premium ones): {len(urls)}")
                download_images(keyword, urls)
            else:
                print("No non-premium images found for download.")
        else:
            print("No results found or there was an error.")