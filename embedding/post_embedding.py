from models_manager import get_sentence_transformer_model


def embed_post(post):     
    model = get_sentence_transformer_model()
    description = f"{post['description']}"

    embedding = model.encode(description)
    return embedding.tolist() 