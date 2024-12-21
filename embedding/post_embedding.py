from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_post(post):     
    description = f"{post['description']}"

    embedding = model.encode(description)
    return embedding.tolist() 