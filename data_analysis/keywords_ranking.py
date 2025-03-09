import numpy as np
from tqdm import tqdm
import bson
import torch
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

def load_bson_file(file_path):
    with open(file_path, "rb") as f:
        data = bson.decode_all(f.read())  
    return data

pins = load_bson_file("../benchmark/pinterest_iccv/subset_iccv_pin_im.bson")  # Pins and images
boards = load_bson_file("../benchmark/pinterest_iccv/subset_iccv_board_pins.bson")  # Boards & pins

embedding_file = "../benchmark/pinterest_data/image_embeddings.npz"
image_embeddings = np.load(embedding_file, allow_pickle=True)

# Define dicitonary and settings

keywords = {
    "mask": ["halloween", "skincare", "cosplay", "medical", "costume"],
    "paragraph of text": ["quote", "joke", "poem", "story", "essay", "prose"],
    "makeup": ["face", "eyeshadow", "lipstick", "nail", "foundation", "blush", "mascara"],
    "fashion": ["clothing", "outfit", "dress", "accessories", "shoes", "hat", "jacket"],
    "nature": ["landscape", "outdoor", "sky", "mountain", "forest", "river", "beach"],
    "exterior": ["house", "garden", "patio", "driveway", "fence", "roof"],
    "furniture": ["home", "decor", "design", "room", "furniture", "lighting", "wall art"],
    "animal": ["pet", "wildlife", "cat", "dog", "bird", "fish", "reptile", "amphibian"],
    "food": ["recipe", "cuisine", "dessert", "beverage", "snack", "healthy eating"],
    "sport": ["football", "basketball", "tennis", "gym", "yoga", "running"],
    "travel": ["destination", "vacation", "hotel", "adventure", "backpacking", "sightseeing"],
    "art": ["painting", "sculpture", "photography", "drawing", "digital art"],
    "music": ["instrument", "concert", "album", "song", "genre", "playlist"],
    "education": ["school", "university", "online course", "tutorial", "workshop"],
    "health": ["fitness", "nutrition", "mental health", "exercise", "wellness"],
    "business": ["startup", "entrepreneurship", "marketing", "finance", "investment"],
    "history": ["ancient", "medieval", "modern", "war", "culture", "archeology"],
    "vehicle": ["car", "motorcycle", "bicycle", "airplane", "boat", "public transport"],
}
top_n = 3

# Load CLIP model and processor

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    embedding = text_features.cpu().numpy()
    normalized_embedding = normalize(embedding, norm='l2')
    return normalized_embedding

# Embed keywords
topics_embeddings = {}
subtopics_embeddings = {}
for keyword, subword in tqdm(keywords.items(), "Embedding keyword categories"):
    topics_embeddings[keyword] = get_text_embedding(keyword)
    for subword in subword:
        subtopics_embeddings[subword] = get_text_embedding(subword)
# Create lists
topics_embeddings_matrix = np.vstack([topics_embeddings[key] for key in topics_embeddings.keys()])
topics_keys = list(topics_embeddings.keys())
subtopics_embeddings_matrix = np.vstack([subtopics_embeddings[key] for key in subtopics_embeddings.keys()])
subtopics_keys = list(subtopics_embeddings.keys())

def get_topics(embedding, threshold): 
    similarities = cosine_similarity(np.array(embedding).reshape(1, -1), topics_embeddings_matrix)
    top_indices = np.where(similarities[0] >= threshold)[0]
    top_topics = [topics_keys[idx] for idx in top_indices]
    return top_topics

def create_collage(image_paths, output_path, output_width, output_height, max_images_per_row=None):
    from PIL import Image
    import math
        # Load all images
    images = [Image.open(img_path) for img_path in image_paths]
    
    # Calculate the number of rows and columns for the grid
    num_images = len(images)
    if max_images_per_row is None:
        max_images_per_row = math.ceil(math.sqrt(num_images))
    
    num_rows = math.ceil(num_images / max_images_per_row)
    
    # Calculate the height and width of each image in the grid
    image_height = output_height // num_rows
    image_width = output_width // max_images_per_row
    
    # Resize and crop images to cover the grid cell
    resized_images = []
    for img in images:
        # Calculate the aspect ratio of the image and the grid cell
        img_aspect_ratio = img.width / img.height
        cell_aspect_ratio = image_width / image_height
        
        # Resize the image to cover the grid cell
        if img_aspect_ratio > cell_aspect_ratio:
            # Image is wider than the cell, so crop the sides
            new_height = image_height
            new_width = int(img.width * (new_height / img.height))
            resized_img = img.resize((new_width, new_height))
            # Crop the sides to fit the cell width
            left = (new_width - image_width) // 2
            top = 0
            right = left + image_width
            bottom = new_height
            resized_img = resized_img.crop((left, top, right, bottom))
        else:
            # Image is taller than the cell, so crop the top and bottom
            new_width = image_width
            new_height = int(img.height * (new_width / img.width))
            resized_img = img.resize((new_width, new_height))
            # Crop the top and bottom to fit the cell height
            left = 0
            top = (new_height - image_height) // 2
            right = new_width
            bottom = top + image_height
            resized_img = resized_img.crop((left, top, right, bottom))
        
        resized_images.append(resized_img)
    
    # Create a blank canvas for the collage
    collage = Image.new('RGB', (output_width, output_height))
    
    # Paste images into the grid
    x_offset, y_offset = 0, 0
    for i, img in enumerate(resized_images):
        collage.paste(img, (x_offset, y_offset))
        x_offset += image_width
        if (i + 1) % max_images_per_row == 0:
            x_offset = 0
            y_offset += image_height
    
    # Save the collage
    collage.save(output_path)
    print(f"Collage saved to {output_path}")


display_board = boards[21]
print(f"Board has {len(display_board["pins"])} pins")
ranking = {}
for pin in display_board["pins"]:
    pin_embedding = image_embeddings[f"{pin}.jpg"]
    topics = get_topics(pin_embedding, threshold=0.18)    
    
    for topic in topics:
        current_subtopics = keywords[topic]
        current_embeddings_matrix = np.vstack([subtopics_embeddings[key] for key in current_subtopics])
        current_keys = list(current_subtopics)

        similarities = cosine_similarity(np.array(pin_embedding).reshape(1, -1), current_embeddings_matrix)

        top_n_indices = np.argsort(similarities[0])[-top_n:][::-1]
        top_n_keywords = [current_keys[idx] for idx in top_n_indices]

        for top_keyword in top_n_keywords:
            if top_keyword not in ranking:
                ranking[top_keyword] = 0
            ranking[top_keyword] += 1

sorted_ranking = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
for keyword, count in sorted_ranking:
    print(f"{keyword}: {count}")

image_paths = [f"../benchmark/pinterest_data/images/{pin}.jpg" for pin in display_board["pins"]]
output_path = "board_collage.jpg"
output_width = 600
output_height = 1200
create_collage(image_paths, output_path, output_width, output_height, max_images_per_row=5)