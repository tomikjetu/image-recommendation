import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import torch

storage_dir = "../application/storage"
post_embeddings = os.path.join(storage_dir, "p_embeddings.json")
keywords_file = "all_topics.json"
keywords_embeddings_file = "topic_embeddings.json"

# Load JSON data
def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

# Load embeddings
embeddings = load_json(post_embeddings)
keys = list(embeddings.keys())
embeddings_matrix = np.array([embeddings[key] for key in keys])

# Load keywords
with open(keywords_file, "r") as f:
    keywords = json.load(f)

keywords_embeddings = load_json(keywords_embeddings_file, None)

if keywords_embeddings is None:
    # Initialize CLIP model
    from transformers import CLIPModel, CLIPProcessor
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    def get_text_embedding(text):
        print(f"Generating embedding for: {text}")
        inputs = processor(text=text, return_tensors="pt")
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        embedding = text_features.cpu().numpy()
        normalized_embedding = normalize(embedding, norm='l2')
        return normalized_embedding

    # Generate topic embeddings
    topic_embeddings = {keyword: get_text_embedding(keyword) for keyword in keywords}
    topic_keys = list(topic_embeddings.keys())
    topic_embeddings_matrix = np.vstack([topic_embeddings[key] for key in topic_keys])

    # Generate new topic embeddings by averaging top-N similar embeddings
    new_topic_embeddings = {}

    top_n = 200  # Number of similar embeddings to consider

    for i, topic in enumerate(topic_keys):
        print(f"Generating new embedding for: {topic}")
        topic_emb = topic_embeddings[topic]
        topic_emb_reshaped = topic_emb.reshape(1, -1)
        similarities = cosine_similarity(topic_emb_reshaped, embeddings_matrix).flatten()
        top_n_indices = np.argsort(similarities)[-top_n:]  # Indices of top-n most similar embeddings
        top_n_embeddings = embeddings_matrix[top_n_indices]
        new_topic_emb = np.mean(top_n_embeddings, axis=0)  # Mean of top-n embeddings
        new_topic_embeddings[topic] = new_topic_emb

    new_topic_embeddings_matrix = np.vstack([new_topic_embeddings[key] for key in topic_keys])

    # Save topic embeddings

    with open(keywords_embeddings_file, "w") as f:
        json.dump({k: v.tolist() for k, v in new_topic_embeddings.items()}, f)
else:
    new_topic_embeddings_matrix = np.array([keywords_embeddings[key] for key in keywords])
    print("Loaded embeddings from file")


# Original matching
original_matching_indexes = []

for i, topic in enumerate(keywords):
    topic_emb = new_topic_embeddings_matrix[i]
    topic_emb_reshaped = topic_emb.reshape(1, -1)
    topic_similarities = cosine_similarity(topic_emb_reshaped, embeddings_matrix).flatten()
    top_index = np.argmax(topic_similarities)
    original_matching_indexes.append(top_index)

# Function to iteratively remove least variant component and compare similarity
def reduce_dimensions_and_compare(step=1):
    similarities = []
    dimensions = []
    reduced_image_matrix = embeddings_matrix.copy()
    reduced_text_matrix = new_topic_embeddings_matrix.copy()

    while reduced_image_matrix.shape[1] > 2:

        # Ensure at least two dimensions remain
        if reduced_image_matrix.shape[1] - step < 2:
            break

        pca_image = PCA(n_components=reduced_image_matrix.shape[1])
        pca_image.fit(reduced_image_matrix)
        pca_text = PCA(n_components=reduced_text_matrix.shape[1])
        pca_text.fit(reduced_text_matrix)

        # Remove least variant components based on step size
        least_variant_indices_image = np.argsort(pca_image.explained_variance_)[:step]
        reduced_image_matrix = np.delete(reduced_image_matrix, least_variant_indices_image, axis=1)

        least_variant_indices_text = np.argsort(pca_text.explained_variance_)[:step]
        reduced_text_matrix = np.delete(reduced_text_matrix, least_variant_indices_text, axis=1)

        dimensions.append(reduced_image_matrix.shape[1])
        print(f"Reducing to {reduced_image_matrix.shape[1]} dimensions...")

        # Create reduced matching for each keyword

        reduced_matching_indexes = []

        for i, topic in enumerate(keywords):
            topic_emb = reduced_text_matrix[i]
            topic_emb_reshaped = topic_emb.reshape(1, -1)
            topic_similarities = cosine_similarity(topic_emb_reshaped, reduced_image_matrix).flatten()
            top_index = np.argmax(topic_similarities)
            reduced_matching_indexes.append(top_index)

        # Compute cosine similarity to original embeddings
        similarity = 0
        for original_idx, reduced_idx in zip(original_matching_indexes, reduced_matching_indexes):
            original_emb = embeddings_matrix[original_idx].reshape(1, -1)  # Get original embedding
            reduced_emb = embeddings_matrix[reduced_idx].reshape(1, -1)  # Get reduced embedding
            similarity += cosine_similarity(original_emb, reduced_emb)[0, 0]  # Compute similarity

        # Average the similarity across all topics
        similarity /= len(original_matching_indexes)
        similarities.append(similarity)
        print(f"Similarity: {similarity}")


    return dimensions, similarities

# Reduce dimensions and compute similarity with step size 10
dimensions, similarities = reduce_dimensions_and_compare(step=10)

# Plot similarity to original embeddings
plt.figure(figsize=(10, 6))
plt.plot(dimensions, similarities, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Dimensions")
plt.ylabel("Similarity between Original and Reduced Embeddings")
plt.title("Effect of Dimensionality Reduction on Text-Image Embedding Similarity")
plt.gca().invert_xaxis()
plt.grid()
plt.show()
