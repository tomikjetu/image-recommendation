import torch
from sklearn.preprocessing import normalize
from models_manager import get_embedding_model

def get_image_embedding(image):
    model, processor = get_embedding_model()
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    embedding = image_features.cpu().numpy()
    normalized_embedding = normalize(embedding, norm='l2')
    return normalized_embedding
