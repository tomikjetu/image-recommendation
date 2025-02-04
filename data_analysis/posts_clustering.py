import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mplcursors
import subprocess 

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

pca = PCA(n_components=2)
projected = pca.fit_transform(embeddings_matrix)

n_clusters = 9
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(projected)

explained_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(8, 8)) 
scatter = ax.scatter(
    projected[:, 0], 
    projected[:, 1], 
    c=labels, 
    cmap='tab10',  
    s=50,          
    alpha=0.5      
)

ax.set_title('K-Means clustering')
ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)')
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)')

cursor = mplcursors.cursor(scatter, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'{keys[sel.index]}'))

def on_click(event):
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont:
            index = ind["ind"][0]
            key = keys[index]
            image_path = os.path.join(images_folder, f"{key}.jpg")
            if os.path.exists(image_path):
                subprocess.run(["start", image_path], shell=True)  #  Windows
                # subprocess.run(["xdg-open", image_path])  # Linux
            else:
                print(f"Image not found: {image_path}")

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
