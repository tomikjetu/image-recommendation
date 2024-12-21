# implementation 2
from models_manager import get_sentence_transformer_model

def embed_user(user):     
    model = get_sentence_transformer_model()
    description = f"{post['description']}"

    embedding = model.encode(description)
    return embedding.tolist() 