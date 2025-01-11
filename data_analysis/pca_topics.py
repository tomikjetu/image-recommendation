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

import torch
from sklearn.preprocessing import normalize
from transformers import CLIPProcessor, CLIPModel
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

embeddings = {keyword: get_text_embedding(keyword) for keyword in keywords}
keys = list(embeddings.keys())

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

embeddings_array = np.vstack([embeddings[key] for key in keys])

pca = PCA(n_components=2)
projected = pca.fit_transform(embeddings_array)

explained_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(1, 1))
scatter = ax.scatter(projected[:, 0], projected[:, 1])
for i, key in enumerate(keys):
    ax.annotate(key, (projected[i, 0], projected[i, 1]))
ax.set_title('PCA of All Topics')
ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)')
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)')

plt.show()
