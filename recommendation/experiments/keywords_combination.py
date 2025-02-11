import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import subprocess
import numpy as np
import torch

post_embeddings = "../../application/storage/p_embeddings.json"
keywords_file = "../../data_analysis/all_topics.json"
keyword_embeddings = "./topic_embeddings.json"

# Load JSON data
def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

embeddings = load_json(post_embeddings)
keywords = load_json(keywords_file)
keyword_embeddings = load_json(keyword_embeddings)

from transformers import CLIPModel, CLIPProcessor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)


subtopic_weights = {
   "animal": .2,
   "savana": .5,
   "zebra": 1,
   "donkey": 1,
   "gazelle": 1,
   "kangaroo": 1
}

def get_text_embedding(text, weight=None):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    embedding = text_features.cpu().numpy()
    normalized_embedding = normalize(embedding, norm='l2')
    weight = weight if weight is not None else subtopic_weights.get(text, 1)
    return normalized_embedding * weight


# append subtopics to keywords
for subtopic in subtopic_weights:
    if(subtopic not in keyword_embeddings):
        keyword_embeddings[subtopic] = get_text_embedding(subtopic, 1).tolist()


# favorite_keywords = ["fish", "kangaroo", "koala", "bear"]
# favorite_embeddings = [get_text_embedding(keyword) for keyword in favorite_keywords]

# # Calculate cosine similarity
# for i, key in enumerate(favorite_keywords):
#     for j in range(i + 1, len(favorite_keywords)):
#         sim = cosine_similarity(favorite_embeddings[i].reshape(1, -1), favorite_embeddings[j].reshape(1, -1))[0][0]
#         print(f"Similarity between {key} and {favorite_keywords[j]}: {sim:.4f}")

positive_keywords = ["animal", "savana"]
negative_keywords = ["zebra"]

positive_embeddings = np.array([get_text_embedding(k) for k in positive_keywords])
negative_embeddings = np.array([get_text_embedding(k) for k in negative_keywords])

# Define weights
w_p = 1.0  # Positive weight
w_n = 0.1  # Negative weight

# Compute weighted sum
if negative_embeddings.size == 0:
    combined_embedding = w_p * positive_embeddings.mean(axis=0)
else:
    combined_embedding = (w_p * positive_embeddings.mean(axis=0)) - (w_n * negative_embeddings.mean(axis=0))
combined_embedding = normalize(combined_embedding.reshape(1, -1), norm='l2')  # Normalize

#Find similar keywords
similar_keywords = []
for keyword, emb in keyword_embeddings.items():
    sim = cosine_similarity(combined_embedding, np.array(emb).reshape(1, -1))[0][0]
    similar_keywords.append((keyword, sim))

similar_keywords.sort(key=lambda x: x[1], reverse=True)

# Find similar posts
similar_posts = []
for key, emb in embeddings.items():
    sim = cosine_similarity(combined_embedding, np.array(emb).reshape(1, -1))[0][0]
    similar_posts.append((key, sim))

similar_posts.sort(key=lambda x: x[1], reverse=True)

def open_image(image_path):
    if os.path.exists(image_path):
        subprocess.run(["start", image_path], shell=True)
        # subprocess.run(["xdg-open", image_path])

#Print top 5 similar keywords
for keyword, sim in similar_keywords[:5]:
    print(f"Keyword: {keyword}, Similarity: {sim:.4f}")

# Print top 5 similar posts
for post, post_sim in similar_posts[:5]:
    post_embedding = np.array(embeddings[post])

    # Get keywords of the post
    post_keywords = []
    for keyword, emb in keyword_embeddings.items():
        keyword_sim = cosine_similarity(post_embedding.reshape(1, -1), np.array(emb).reshape(1, -1))[0][0]  
        post_keywords.append((keyword, keyword_sim))
    
    post_keywords.sort(key=lambda x: x[1], reverse=True)

    print(f"Post: {post}, Similarity: {post_sim:.4f}, ({post_keywords[0][0]}, {post_keywords[1][0]}, {post_keywords[2][0]}, {post_keywords[3][0]}, {post_keywords[4][0]})")
    open_image(f"../../application/storage/images/{post}.jpg")