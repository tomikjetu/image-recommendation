import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel
import subprocess  


storage_dir = "../application/storage"
post_embeddings_path = os.path.join(storage_dir, "p_embeddings.json")
images_folder = os.path.join(storage_dir, "images")

def load_embeddings(file):
    try:
        with open(file, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Embedding file not found: {file}")
        return {}

embeddings = load_embeddings(post_embeddings_path)

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    embedding = text_features.cpu().numpy()
    normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return normalized_embedding

def search_files(query, embeddings, top_k=5):
    query_embedding = get_text_embedding(query)
    keys = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[key] for key in keys])
    
    similarities = cosine_similarity(query_embedding, embedding_matrix).flatten()
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [(keys[i], similarities[i]) for i in sorted_indices]
    return results

def open_image(image_path):
    if os.path.exists(image_path):
        subprocess.run(["start", image_path], shell=True)
        # subprocess.run(["xdg-open", image_path])
#
if __name__ == "__main__":
    while True:
        query = input("Enter your search query: ").strip()
        top_k = 5 
        
        if embeddings:
            results = search_files(query, embeddings, top_k=top_k)
            print("\nSearch Results:")
            for filename, score in results:
                image_path = os.path.join(images_folder, f"{filename}.jpg")
                print(f"File: {filename} (Similarity: {score:.4f})")
                print(f"Opening: {image_path}")
                open_image(image_path)
        else:
            print("No embeddings found.")
