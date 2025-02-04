import torch
from sklearn.preprocessing import normalize
from transformers import CLIPProcessor, CLIPModel
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

# Load keywords from json file
import json
keywords_file = "all_topics.json"
with open(keywords_file, "r") as f:
    keywords = json.load(f)


embeddings = {keyword: get_text_embedding(keyword) for keyword in keywords}
keys = list(embeddings.keys())

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

embeddings_array = np.vstack([embeddings[key] for key in keys])

pca = PCA(n_components=2)
projected = pca.fit_transform(embeddings_array)

explained_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(1, 1))
scatter = ax.scatter(projected[:, 0], projected[:, 1])
for i, key in enumerate(keys):
    ax.annotate(key, (projected[i, 0], projected[i, 1]))
ax.set_title('PCA of All Topics')
ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)')
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)')

plt.show()
