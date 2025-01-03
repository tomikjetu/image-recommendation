import os
import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mplcursors

storage_dir = "../application/storage"

post_embeddings = os.path.join(storage_dir, "p_embeddings.json")

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

explained_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(1, 1))
scatter = ax.scatter(projected[:, 0], projected[:, 1])
ax.set_title('2D Projection of All Embeddings')
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
            print(key)

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
