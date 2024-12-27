from sklearn.metrics.pairwise import cosine_similarity

from application.storage.storage_manager import users, u_embeddings
from embedding.user_embedding import embed_user

def similar_users(user_id, top_n=5):
    u_embedding = embed_user(user_id)
    if(len(u_embedding)==0):
        return []
    
    candidate_ids = [u_id for u_id in users if u_id != user_id]
    candidate_embeddings = [embed_user(u_id) for u_id in users if u_id != user_id]
    
    if(len(candidate_ids)==0):
        return []
    
    similarities = cosine_similarity(u_embedding, candidate_embeddings[0])[0]
    user_similarity_pairs = list(zip(candidate_ids, similarities))
    sorted_pairs = sorted(user_similarity_pairs, key=lambda x: x[1], reverse=True)
    
    print(sorted_pairs)

    sorted_users = [u_id for u_id, _ in sorted_pairs]
    return sorted_users[:top_n]