import os
import time
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored
import json

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

base_dir = "../application/storage/object_images"  
keyword_embeddings_path = "./topic_embeddings.json"  

# Load keyword embeddings
with open(keyword_embeddings_path, "r") as f:
    keyword_embeddings = json.load(f)

# Tracking results
total_images = 0
correct_predictions = 0


# Iterate through each category (directory)
for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if not os.path.isdir(category_path):
        continue
    
    if category not in keyword_embeddings:
        print(colored(f"Skipping {category}, no keyword embedding found.", "yellow"))
        continue

    keyword_embedding = np.array(keyword_embeddings[category]).reshape(1, -1)

    for image_name in os.listdir(category_path):
        start_time = time.time()
        image_path = os.path.join(category_path, image_name)
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cpu")
        
        # Get image embedding
        with torch.no_grad():
            image_embedding = model.get_image_features(**inputs).cpu().numpy()
        
        # Compare with keyword embeddings

        def normalize(vec):
            return vec / np.linalg.norm(vec, axis=1, keepdims=True)

        image_embedding = normalize(image_embedding)
        keyword_matrix = normalize(np.array(list(keyword_embeddings.values())))

        # Compute cosine similarity
        similarities = cosine_similarity(image_embedding, keyword_matrix)
        predicted_category = list(keyword_embeddings.keys())[np.argmax(similarities)]
        
        # Track results
        total_images += 1
        is_correct = predicted_category == category
        correct_predictions += int(is_correct)
        
        # Print result
        end_time = time.time()
        total_time = end_time - start_time
        result_text = f"[{category}] -> Predicted: {predicted_category} ({'✔' if is_correct else '✘'}) ({total_time:.2f}s)"
        print(colored(result_text, "green" if is_correct else "red"))

accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0

# Print summary
print(colored(f"\nTotal images processed: {total_images}", "cyan"))
print(colored(f"Correct predictions: {correct_predictions}", "cyan"))
print(colored(f"Accuracy: {accuracy:.2f}%", "cyan"))
