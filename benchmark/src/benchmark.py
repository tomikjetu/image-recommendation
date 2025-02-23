from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
import time

from bson_data import boards

def get_board_mean_embedding(board_pins, image_embeddings):
    valid_embeddings = [image_embeddings[f"{pin}.jpg"].flatten() for pin in board_pins if f"{pin}.jpg" in image_embeddings]
    if not valid_embeddings:
        return None  
    return np.mean(valid_embeddings, axis=0) 

def split_board_pins(board_pins, test_size=0.2):
    if len(board_pins) < 2:
        return board_pins, []
    train_pins, test_pins = train_test_split(board_pins, test_size=test_size, random_state=42)
    return train_pins, test_pins

def recommend_images(board_embedding, image_embeddings, top_k=10):
    if board_embedding is None:
        return []
    image_ids = list(image_embeddings.keys())
    image_matrix = np.array([image_embeddings[img_id].flatten() for img_id in image_ids]) 
    similarities = cosine_similarity(board_embedding.reshape(1, -1), image_matrix)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(image_ids[i].replace(".jpg", ""), similarities[i]) for i in top_indices]

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

embedding_file = "../pinterest_data/image_embeddings.npz"
image_embeddings = np.load(embedding_file, allow_pickle=True)

k = 20
sum_precision = 0
sum_recall = 0
sum_ndcg = 0
sum_hr = 0

time_start = time.time()

if __name__ == "__main__":
    # Filter out boards with missing images
    filled_boards = []
    for board in boards:
        pins = board["pins"]
        found = True
        for pin in pins:
            if f"{pin}.jpg" not in image_embeddings.keys():
                found = False
                break
        if found:
            filled_boards.append(board)

    print(f"Total {len(filled_boards)}/{len(boards)} boards found.")

    for board in filled_boards:
        board_pins = board["pins"]
        train_pins, test_pins = split_board_pins(board_pins, test_size=0.2)

        board_embedding = get_board_mean_embedding(train_pins, image_embeddings) # Train pins for embeddings
        recommended_images = recommend_images(board_embedding, image_embeddings, k)

        # Test pins for evaluation
        precision, recall, hit = precision_recall_at_k(recommended_images, test_pins, k)
        ndcg = ndcg_at_k(recommended_images, test_pins, k)

        print(f"Precision@{k}: {precision:.3f}, Recall@{k}: {recall:.3f}, NDCG@{k}: {ndcg:.3f}, HitR@{k}: {hit} ({len(board_pins)} pins)")
        sum_precision += precision
        sum_recall += recall
        sum_ndcg += ndcg
        sum_hr += hit

    print("-" * 50)
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start:.2f}s")

    avg_precision = sum_precision / len(filled_boards)
    avg_recall = sum_recall / len(filled_boards)
    avg_ndcg = sum_ndcg / len(filled_boards)
    avg_hr = sum_hr / len(filled_boards)
    print(f"Average Precision@{k}: {avg_precision:.3f}, Average Recall@{k}: {avg_recall:.3f}, Average NDCG@{k}: {avg_ndcg:.3f}, Average HitR@{k}: {avg_hr:.3f}")
    average_board_pins = sum(len(board["pins"]) for board in boards) / len(boards)
    print(f"Average board pins: {average_board_pins:.2f}")
