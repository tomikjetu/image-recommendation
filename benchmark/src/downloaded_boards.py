from bson_data import boards
import os
import numpy as np

embedding_file = "../pinterest_data/image_embeddings.npz"

if os.path.exists(embedding_file):
    data = np.load(embedding_file, allow_pickle=True)
    image_embeddings = dict(data)  
else:
    print("Image embeddings file not found.")
    exit()

board_amount = 0
full_boards = 0
board_total = 0
for board in boards:
    found = True
    least_pins = False
    pins = board["pins"]
    for pin in pins:
        if f"{pin}.jpg" not in image_embeddings.keys():
            found = False
        else:
            least_pins = True
    if least_pins:
        board_amount += 1
    board_total += 1
    if not found:
        continue
    full_boards += 1

print(f"Total {full_boards}/{board_amount}/{board_total} boards found.")