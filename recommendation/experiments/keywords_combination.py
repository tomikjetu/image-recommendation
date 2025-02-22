import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import subprocess
import numpy as np
import torch

post_embeddings = "../../application/storage/clustered_embeddings.json"

# Load JSON data
def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

embeddings = load_json(post_embeddings)["animal"]

from transformers import CLIPModel, CLIPProcessor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

high_subtopics = ['aardvark', 'alligator', 'alpaca', 'ant', 'anteater', 'antelope', 'badger', 'barnacle', 'bear', 'beaver', 'bee', 'beetle', 'bird', 'bison','blowfish', 'boar', 'bongo', 'bug', 'bull', 'butterfly', 'camel', 'cardinal', 'cat', 'caterpillar', 'catfish', 'cheetah', 'chick', 'chihuahua', 'chinchilla', 'chipmunk', 'clam', 'cobra', 'cockatoo', 'cockroach', 'cougar', 'cow', 'coyote', 'crab', 'crane', 'crayfish', 'crow', 'dalmatian', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'duckling', 'eagle', 'earwig', 'eel', 'elephant', 'ferret', 'fish', 'flamingo', 'fly', 'fox', 'frog', 'gazelle', 'giraffe', 'goat', 'goldfish', 'goose', 'gopher', 'gorilla', 'grasshopper', 'groundhog', 'guinea_pig', 'hamster', 'hawk', 'hedgehog', 'hippopotamus', 'horse', 'hummingbird', 'hyena', 'iguana', 'jack', 'jellyfish', 'kangaroo', 'kitten', 'kiwi', 'koala', 'ladybug', 'lamb', 'leech', 'leopard', 'lightning_bug', 'lion', 'lizard', 'llama', 'lobster', 'maggot', 'manatee', 'meerkat', 'mole', 'mongoose', 'monkey', 'moose', 'mosquito', 'moth', 'mullet', 'mussel', 'octopus', 'orangutan', 'ostrich', 'otter', 'owl', 'oyster', 'panda', 'panther', 'parrot', 'peacock', 'pelican', 'penguin', 'pheasant', 'pig', 'pigeon', 'piglet', 'platypus', 'polar_bear', 'pony', 'poodle', 'porcupine', 'possum', 'poster', 'praying_mantis', 'puffin', 'pug', 'puppy', 'rabbit', 'raccoon', 'racehorse', 'ram', 'rat', 'rattlesnake', 'reindeer', 'rhinoceros', 'roller', 'rooster', 'sardine', 'scallop', 'scorpion', 'sea_urchin', 'seagull', 'seahorse', 'seal', 'shark', 'sheep', 'shrimp', 'siren', 'skunk', 'sloth', 'slug', 'snail', 'snake', 'spider', 'sponge', 'squid', 'squirrel', 'starfish', 'stilt', 'stingray', 'swan', 'sweeper', 'swordfish', 'tadpole', 'tarantula', 'tick', 'tiger', 'toad', 'torpedo', 'toucan', 'turkey', 'turtle', 'vulture', 'walrus', 'warthog', 'wasp', 'weasel', 'whale', 'wolf', 'worm', 'yak', 'zebra']
neutral_subtopics = ["baby", "bones"]
low_subtopics = ['desert', 'polar', 'wetlands', 'forest', 'grassland', 'mountain', 'ocean', 'rainforest', 'savanna', 'tundra', 'urban']

def get_text_embedding(text, weight=None):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    embedding = text_features.cpu().numpy()
    normalized_embedding = normalize(embedding, norm='l2')
    weight = 1 if weight is None else weight
    return normalized_embedding * weight

keyword_embeddings = {}

# append subtopics to keywords
for subtopic, weight in [[high_subtopics, 1], [neutral_subtopics, .5], [low_subtopics, .2]]:
    for keyword in subtopic:
        if(keyword not in keyword_embeddings):
            keyword_embeddings[keyword] = get_text_embedding(keyword, weight).tolist()

negative_threshold = 0.2

# Define weights
w_p = 1.0  # Positive weight
w_n = 0.05 # Negative weight

N_positive = 20
N_negative = 5

ranking = {keyword: np.random.rand() for subtopics in [high_subtopics, neutral_subtopics, low_subtopics] for keyword in subtopics}

positive_keywords = [(k, v) for k, v in ranking.items() if v > negative_threshold]
negative_keywords = [(k, v) for k, v in ranking.items() if v <= negative_threshold]

positive_embeddings = [(k, get_text_embedding(k)) for k, _ in positive_keywords]
negative_embeddings = [(k, get_text_embedding(k)) for k, _ in negative_keywords]

positive_values = np.array([v for _, v in positive_keywords]) 
negative_values = np.array([v for _, v in negative_keywords]) 

# Sort by values
sorted_positive_indices = np.argsort(positive_values)[::-1]
sorted_negative_indices = np.argsort(negative_values)

picked_positive_embeddings = [positive_embeddings[i] for i in sorted_positive_indices[:N_positive]]
picked_negative_embeddings = [negative_embeddings[i] for i in sorted_negative_indices[:N_negative]]

for keyword, emb in picked_positive_embeddings:
    print(f"Positive Keyword: {keyword}, ranking = {ranking[keyword]}")

for keyword, emb in picked_negative_embeddings:
    print(f"Negative Keyword: {keyword}, ranking = {ranking[keyword]}")

positive_emb_vectors = np.array([emb for _, emb in picked_positive_embeddings])
negative_emb_vectors = np.array([emb for _, emb in picked_negative_embeddings]) if picked_negative_embeddings else None

# Compute weighted sum
if negative_emb_vectors is None:
    combined_embedding = w_p * positive_emb_vectors.mean(axis=0)
else:
    combined_embedding = (w_p * positive_emb_vectors.mean(axis=0)) - (w_n * negative_emb_vectors.mean(axis=0))
combined_embedding = normalize(combined_embedding.reshape(1, -1), norm='l2')  # Normalize

#Find similar keywords
similar_keywords = []
for keyword, emb in keyword_embeddings.items():
    sim = cosine_similarity(combined_embedding, np.array(emb).reshape(1, -1))[0][0]
    similar_keywords.append((keyword, sim))

similar_keywords.sort(key=lambda x: x[1], reverse=True)

# Find similar posts
similar_posts = []
for key, emb in embeddings.items():
    sim = cosine_similarity(combined_embedding, np.array(emb).reshape(1, -1))[0][0]
    similar_posts.append((key, sim))

similar_posts.sort(key=lambda x: x[1], reverse=True)

def open_image(image_path):
    if os.path.exists(image_path):
        subprocess.run(["start", image_path], shell=True)
        # subprocess.run(["xdg-open", image_path])

#Print top 5 similar keywords
for keyword, sim in similar_keywords[:5]:
    print(f"Keyword: {keyword}, Similarity: {sim:.4f}")

# Print top 5 similar posts
for post, post_sim in similar_posts[:5]:
    post_embedding = np.array(embeddings[post])

    # Get keywords of the post
    post_keywords = []
    for keyword, emb in keyword_embeddings.items():
        keyword_sim = cosine_similarity(post_embedding.reshape(1, -1), np.array(emb).reshape(1, -1))[0][0]  
        post_keywords.append((keyword, keyword_sim))
    
    post_keywords.sort(key=lambda x: x[1], reverse=True)

    print(f"Post: {post}, Similarity: {post_sim:.4f}, ({post_keywords[0][0]}, {post_keywords[1][0]}, {post_keywords[2][0]}, {post_keywords[3][0]}, {post_keywords[4][0]})")
    open_image(f"../../application/storage/images/{post}.jpg")