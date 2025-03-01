import numpy as np
import time
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from benchmark_metrics import recommend_images, split_board_pins, precision_recall_at_k, ndcg_at_k

from bson_data import boards

def get_board_embedding(board_pins, image_embeddings):
    valid_embeddings = [image_embeddings[f"{pin}.jpg"].flatten() for pin in board_pins if f"{pin}.jpg" in image_embeddings]
    if not valid_embeddings:
        return None  
    
    embedding = np.array(valid_embeddings[0])

    for i, emb in enumerate(valid_embeddings[1:]):
        similarity = cosine_similarity(embedding.reshape(1, -1), emb.reshape(1,-1))[0][0]
        a1 = (similarity + 1) / 2
        a2 = 1 - a1
        embedding = (a1 * embedding + a2 * emb)
    return embedding
        
embedding_file = "../pinterest_data/image_embeddings.npz"
image_embeddings = np.load(embedding_file, allow_pickle=True)

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

    train_boards, test_boards = split_board_pins(filled_boards, test_size=0.2)
    
    past_sessions = []
    for board in tqdm(train_boards, desc="Creating training sessions"):
        board_pins = board["pins"]
        board_embedding = get_board_embedding(board_pins, image_embeddings)
        past_sessions.append(board_embedding)

    dataset_embedding = past_sessions[-1]
    weight = 0
    for i, session in enumerate(past_sessions[:-1]):
        beta = (cosine_similarity(np.array(dataset_embedding).reshape(1, -1), np.array(session).reshape(1, -1))[0][0] + 1) / 2
        dataset_embedding = beta * np.array(session) + (1 - beta) * dataset_embedding
        weight += beta
    dataset_embedding = dataset_embedding / weight

    results = {}
    for split in [0, 0.2, .5]:
        results[split] = {}
        print(f"Split: {split}")
        for k in [100, 20, 50]:
            sum_precision = 0
            sum_recall = 0
            sum_ndcg = 0
            sum_hr = 0
            print(f"Top K: {k}")
            for board in test_boards:
                board_pins = board["pins"]
                train_pins, test_pins = split_board_pins(board_pins, test_size=split)

                session_embedding = get_board_embedding(train_pins, image_embeddings)

                gama = (cosine_similarity(np.array(dataset_embedding).reshape(1, -1), np.array(session_embedding).reshape(1, -1))[0][0] + 1) / 2 
                realtime_embedding = gama * np.array(session_embedding) + (1 - gama) * np.array(dataset_embedding)
                
                recommended_images = recommend_images(realtime_embedding, image_embeddings, k)

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

            avg_precision = sum_precision / len(test_boards)
            avg_recall = sum_recall / len(test_boards)
            avg_ndcg = sum_ndcg / len(test_boards)
            avg_hr = sum_hr / len(test_boards)
            print(f"Average Precision@{k}: {avg_precision:.3f}, Average Recall@{k}: {avg_recall:.3f}, Average NDCG@{k}: {avg_ndcg:.3f}, Average HitR@{k}: {avg_hr:.3f}")
            average_board_pins = sum(len(board["pins"]) for board in boards) / len(boards)
            print(f"Average board pins: {average_board_pins:.2f}")
            results[split][k] = (avg_precision, avg_recall, avg_ndcg, avg_hr)
    print(results)
