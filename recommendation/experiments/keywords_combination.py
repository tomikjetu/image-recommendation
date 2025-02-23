import json
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from transformers import CLIPModel, CLIPProcessor

# Load CLIP model
def get_embedding_model():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

model, processor = get_embedding_model()

# Load JSON data
def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

post_embeddings = "../../application/storage/clustered_embeddings.json"
embeddings = load_json(post_embeddings)["animal"]

# Define subtopics
high_subtopics = ['lion', 'dog', 'cat', 'hamster']
neutral_subtopics = ["baby", "bones"]
low_subtopics = ['rat', 'mouse', 'squirrel']

def get_text_embedding(text):
    """ Get normalized text embedding from CLIP """
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    embedding = text_features.cpu().numpy()
    return normalize(embedding, norm='l2')

# Initialize session embedding
session_embedding = np.zeros((1, 512))  # Assuming 512D CLIP embeddings

# Function to update session embedding using NAGI's formula
def update_session_embedding(s_u_prev, new_embedding):
    """ Update session embedding using adaptive weighting (NAGI method) """
    alpha_1 = (cosine_similarity(new_embedding, s_u_prev)[0][0] + 1) / 2
    alpha_2 = 1 - alpha_1
    s_u_new = alpha_1 * s_u_prev + alpha_2 * new_embedding
    return normalize(s_u_new, norm='l2')

# Process keywords and update session embedding dynamically
keyword_embeddings = {}

for subtopic in high_subtopics + neutral_subtopics + low_subtopics:
    keyword_embedding = get_text_embedding(subtopic)
    keyword_embeddings[subtopic] = keyword_embedding
    session_embedding = update_session_embedding(session_embedding, keyword_embedding)

# Find similar posts
similar_posts = []
for key, emb in embeddings.items():
    sim = cosine_similarity(session_embedding, np.array(emb).reshape(1, -1))[0][0]
    similar_posts.append((key, sim))

similar_posts.sort(key=lambda x: x[1], reverse=True)

# Print top 5 similar posts
for post, post_sim in similar_posts[:5]:
    print(f"Post: {post}, Similarity: {post_sim:.4f}")
