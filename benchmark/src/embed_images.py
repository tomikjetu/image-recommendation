import torch
from sklearn.preprocessing import normalize
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
from PIL import Image

image_dir = "../pinterest_data/images"
embedding_file = "../pinterest_data/image_embeddings.npz"

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    embedding = image_features.cpu().numpy()
    return normalize(embedding, norm='l2')

if os.path.exists(embedding_file):
    data = np.load(embedding_file, allow_pickle=True)
    image_embeddings = dict(data)  
else:
    image_embeddings = {}

# Embed images
amount = 0
for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)

    if filename in image_embeddings:
        continue 
    
    try:
        image = Image.open(image_path).convert("RGB")
        embedding = get_image_embedding(image)
        image_embeddings[filename] = embedding  
        amount += 1
        
        if amount % 100 == 0:
            np.savez(embedding_file, **image_embeddings)
            print(f"Saved {amount} embeddings")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save final embeddings
np.savez(embedding_file, **image_embeddings)
print(f"Total {len(image_embeddings)} embeddings saved.")
