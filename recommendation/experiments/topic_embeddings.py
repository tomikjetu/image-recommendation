import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import torch


all_topics = "../../data_analysis/all_topics.json"

# Load JSON data
def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

keywords = load_json(all_topics)

from transformers import CLIPModel, CLIPProcessor
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

# Get embeddings for all keywords
keyword_embeddings = {}
for keyword in keywords:
    keyword_embeddings[keyword] = get_text_embedding(keyword).tolist()

# Save embeddings to file
json.dump(keyword_embeddings, open("topic_embeddings.json", "w"), indent=2, ensure_ascii=False)