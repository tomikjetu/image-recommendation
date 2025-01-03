import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


storage_dir = "../application/storage"
post_embeddings = os.path.join(storage_dir, "p_embeddings.json")
images_folder = os.path.join(storage_dir, "images")
n_images = 1500

def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

embeddings = load_json(post_embeddings)
keys = list(embeddings.keys())[:n_images]  
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
ax.set_title('2D Projection of Image Embeddings', fontsize=16)
ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')

for i, (x, y) in enumerate(projected):
    image_path = os.path.join(images_folder, f"{keys[i]}.jpg")
    img = load_image(image_path)
    if img is not None:
        imagebox = OffsetImage(img, zoom=0.5)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)

plt.show()
