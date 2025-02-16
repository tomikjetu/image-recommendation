import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

storage_dir = "../application/storage"
post_embeddings = os.path.join(storage_dir, "clustered_embeddings.json")
images_folder = os.path.join(storage_dir, "images")
n_images = 1000

def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

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
keyword = keywords[8]
embeddings = load_json(post_embeddings).get(keyword)
n_images = min(len(embeddings), n_images)
keys = list(embeddings.keys())
embeddings_matrix = np.array([embeddings[key] for key in keys])

pca = PCA(n_components=2)
projected = pca.fit_transform(embeddings_matrix)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(projected)

def load_image(file_path, size=(30, 30)):
    try:
        img = Image.open(file_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_title(f'PCA {keyword} showing {n_images} images', fontsize=16)
ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')

random_indices = np.random.choice(len(projected), n_images, replace=False)
for i in random_indices:
    x, y = projected[i]
    image_path = os.path.join(images_folder, f"{keys[i]}.jpg")
    img = load_image(image_path)
    if img is not None:
        imagebox = OffsetImage(img, zoom=0.75)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap='tab10', s=10, alpha=0)

plt.show()
