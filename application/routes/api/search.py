from flask import request, jsonify
import numpy as np
import torch
from application.app import app
from sklearn.metrics.pairwise import cosine_similarity
from application.routes.api.get_recommendation import close_user_session
from models_manager import get_embedding_model

from application.storage.storage_manager import p_embeddings, recommendation_storage

def get_text_embedding(text):
    model, processor = get_embedding_model()
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    embedding = text_features.cpu().numpy()
    normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return normalized_embedding

def search_files(query, embeddings, top_k=5):
    query_embedding = get_text_embedding(query)
    keys = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[key] for key in keys])
    
    similarities = cosine_similarity(query_embedding, embedding_matrix).flatten()
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [(keys[i], similarities[i]) for i in sorted_indices]
    return results

@app.route('/query', methods=['GET'])
def query():
    query_param = request.args.get('query')
    user_id = request.cookies.get('user_id')
    if not query_param:
        return jsonify({'error': 'Query parameter is missing'}), 400

    close_user_session(user_id)
    results = search_files(query_param, p_embeddings, top_k=10)

    recommendation_storage[user_id] = [result[0] for result in results]

    result = {'message': f'You searched for: {query_param}'}

    return jsonify(result), 200