import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import subprocess
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

storage_dir = "../application/storage"
post_embeddings = os.path.join(storage_dir, "p_embeddings.json")
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

top_n = 5  
new_topic_embeddings = {}

for i, topic in enumerate(topic_keys):
    topic_emb = topic_embeddings[topic]
    similarities = cosine_similarity(topic_emb, embeddings_matrix).flatten()
    top_n_indices = np.argsort(similarities)[-top_n:]  # Indices of top-n most similar embeddings
    top_n_embeddings = embeddings_matrix[top_n_indices]
    new_topic_emb = np.mean(top_n_embeddings, axis=0)  # Mean of top-n embeddings
    new_topic_embeddings[topic] = new_topic_emb


new_topic_embeddings_matrix = np.array([new_topic_embeddings[topic] for topic in topic_keys])
combined_matrix = np.vstack((embeddings_matrix, new_topic_embeddings_matrix))

pca = PCA(n_components=2)
projected_embeddings = pca.fit_transform(embeddings_matrix)
projected_topics = pca.transform(np.array([new_topic_embeddings[topic] for topic in topic_keys]))

explained_variance = pca.explained_variance_ratio_

labels = []

for emb in projected_embeddings:
    distances = np.linalg.norm(projected_topics - emb, axis=1)
    closest_topic_index = np.argmin(distances)
    labels.append(closest_topic_index)

labels = np.array(labels)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(
    projected_embeddings[:, 0],
    projected_embeddings[:, 1],
    c=labels,
    s=50,
    alpha=0.6,
    label='Image Embeddings'
)

# Plot new topic centers
ax.scatter(
    projected_topics[:, 0],
    projected_topics[:, 1],
    c='red',
    marker='x',
    s=100,
    label='Topic Centers'
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

ax.set_title('PCA with Image Embedding clusters with Topic Centers (2 Components)')
ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)')
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)')
ax.legend()

# Interactivity: Image Preview
def on_click(event):
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont:
            index = ind["ind"][0]
            key = keys[index]
            image_path = os.path.join(images_folder, f"{key}.jpg")
            if os.path.exists(image_path):
                subprocess.run(["start", image_path], shell=True)  # Windows
                # subprocess.run(["xdg-open", image_path])  # Linux
            else:
                print(f"Image not found: {image_path}")

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
