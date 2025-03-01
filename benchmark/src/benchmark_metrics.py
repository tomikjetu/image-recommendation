import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def recommend_images(board_embedding, image_embeddings, top_k=10):
    if board_embedding is None:
        return []
    image_ids = list(image_embeddings.keys())
    image_matrix = np.array([image_embeddings[img_id].flatten() for img_id in image_ids]) 
    similarities = cosine_similarity(board_embedding.reshape(1, -1), image_matrix)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(image_ids[i].replace(".jpg", ""), similarities[i]) for i in top_indices]

def split_board_pins(board_pins, test_size=0.2):
    if(test_size == 0):
        return board_pins, board_pins
    if len(board_pins) < 2:
        return board_pins, []
    train_pins, test_pins = train_test_split(board_pins, test_size=test_size, random_state=42)
    return train_pins, test_pins

def precision_recall_at_k(recommended, ground_truth, k=10):
    """Precision and Recall @ K: Precision and Recall at K for recommended items."""
    recommended_set = set([img[0] for img in recommended[:k]])
    ground_truth_set = set(ground_truth)
    
    intersection = recommended_set.intersection(ground_truth_set)
    hit = 1 if intersection else 0
    precision = len(intersection) / k
    recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0
    
    return precision, recall, hit

def ndcg_at_k(recommended, ground_truth, k=10):
    """Normalized Discounted Cumulative Gain @ K: Discounted Cumulative Gain (DCG) at K normalized by Ideal DCG at K."""
    # Extract relevance scores for recommended items
    relevance = [1 if img_id in ground_truth else 0 for img_id, _ in recommended[:k]]
    
    # Calculate DCG
    dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance))
    
    # Calculate IDCG (Ideal DCG)
    ideal_relevance = sorted([1 if img_id in ground_truth else 0 for img_id, _ in recommended], reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    
    # Normalize DCG by IDCG
    return dcg / idcg if idcg > 0 else 0