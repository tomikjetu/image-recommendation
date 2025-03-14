import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load embeddings
storage_dir = "../application/storage"
post_embeddings = os.path.join(storage_dir, "p_embeddings.json")
suffix = ".jpg"

# storage_dir = "../benchmark/pinterest_data"
# post_embeddings = os.path.join(storage_dir, "image_embeddings.npz")
# embeddings = np.load(post_embeddings, allow_pickle=True)
# suffix = ""

images_folder = os.path.join(storage_dir, "images")

def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

embeddings = load_json(post_embeddings)
keys = list(embeddings.keys())
embeddings_matrix = np.array([embeddings[key] for key in keys])

keywords = [
    "animal",
    "vehicle",
    "building",
    "landscape",
    "sport",
    "food",
    "technology",
    "culture and art",
    "room interior"
]

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

topic_embeddings = {keyword: get_text_embedding(keyword) for keyword in keywords}
topic_keys = list(topic_embeddings.keys())
topic_embeddings_matrix = np.vstack([topic_embeddings[key] for key in topic_keys])

# Compute new embeddings for topics using top-n similar images
n = 5  # Number of top similar images to consider
new_topic_embeddings = {}

for i, topic in enumerate(topic_keys):
    topic_emb = topic_embeddings[topic]
    similarities = cosine_similarity(topic_emb, embeddings_matrix).flatten()
    top_n_indices = np.argsort(similarities)[-n:]  # Indices of top-n most similar embeddings
    top_n_embeddings = embeddings_matrix[top_n_indices]
    new_topic_emb = np.mean(top_n_embeddings, axis=0)  # Mean of top-n embeddings
    new_topic_embeddings[topic] = new_topic_emb

# Prepare data for PCA and visualization
new_topic_embeddings_matrix = np.array([new_topic_embeddings[topic] for topic in topic_keys])

pca = PCA(n_components=2)
projected_embeddings = pca.fit_transform(embeddings_matrix)
projected_topics = pca.transform(np.array([new_topic_embeddings[topic] for topic in topic_keys]))

explained_variance = pca.explained_variance_ratio_

# Plotting
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_title('PCA of Image Embedding with Topic Centers')
ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)')
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)')

# Plot images
n_images = 1000
random_indices = np.random.choice(len(projected_embeddings), n_images, replace=False)

def load_image(file_path, size=(30, 30)):
    try:
        img = Image.open(file_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

for i in random_indices:
    x, y = projected_embeddings[i]
    image_path = os.path.join(images_folder, f"{keys[i]}{suffix}")
    img = load_image(image_path)
    if img is not None:
        imagebox = OffsetImage(img, zoom=0.75)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

# Plot new topic centers
ax.scatter(
    projected_topics[:, 0],
    projected_topics[:, 1],
    c='red',
    marker='x',
    s=100,
    label='Topic Centers',
    zorder=10
)

for i, topic in enumerate(topic_keys):
    ax.text(
        projected_topics[i, 0] - .005,
        projected_topics[i, 1],
        topic,
        fontsize=9,
        zorder=11,
        ha='right',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

scatter = ax.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], cmap='tab10', s=10, alpha=0)
ax.legend()
plt.show()
