from sklearn.model_selection import train_test_split
import numpy as np
import time
from tqdm import tqdm

from benchmark_metrics import recommend_images, split_board_pins, precision_recall_at_k, ndcg_at_k

from bson_data import boards

def get_board_mean_embedding(board_pins, image_embeddings):
    valid_embeddings = [image_embeddings[f"{pin}.jpg"].flatten() for pin in board_pins if f"{pin}.jpg" in image_embeddings]
    if not valid_embeddings:
        return None  
    return np.mean(valid_embeddings, axis=0) 

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
            for board in tqdm(filled_boards, desc="Evaluating boards"):
                board_pins = board["pins"]
                train_pins, test_pins = split_board_pins(board_pins, test_size=split)

                board_embedding = get_board_mean_embedding(train_pins, image_embeddings) # Train pins for embeddings
                recommended_images = recommend_images(board_embedding, image_embeddings, k)

                # Test pins for evaluation
                precision, recall, hit = precision_recall_at_k(recommended_images, test_pins, k)
                ndcg = ndcg_at_k(recommended_images, test_pins, k)

                # print(f"Precision@{k}: {precision:.3f}, Recall@{k}: {recall:.3f}, NDCG@{k}: {ndcg:.3f}, HitR@{k}: {hit} ({len(board_pins)} pins)")
                sum_precision += precision
                sum_recall += recall
                sum_ndcg += ndcg
                sum_hr += hit

            time_end = time.time()
            print(f"Time elapsed: {time_end - time_start:.2f}s")

            avg_precision = sum_precision / len(filled_boards)
            avg_recall = sum_recall / len(filled_boards)
            avg_ndcg = sum_ndcg / len(filled_boards)
            avg_hr = sum_hr / len(filled_boards)
            results[split][k] = {"precision": avg_precision, "recall": avg_recall, "ndcg": avg_ndcg, "hit": avg_hr}
            print(f"Average Precision@{k}: {avg_precision:.3f}, Average Recall@{k}: {avg_recall:.3f}, Average NDCG@{k}: {avg_ndcg:.3f}, Average HitR@{k}: {avg_hr:.3f}")
            print("-" * 25)
        print("-" * 50)
    print(results)
    average_board_pins = sum(len(board["pins"]) for board in boards) / len(boards)
    print(f"Average board pins: {average_board_pins:.2f}")
