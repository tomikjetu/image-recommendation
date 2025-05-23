from recommendation.cold_start import cold_start
from recommendation.similar_post import similar_posts
import uuid

import numpy as np
from application.storage.storage_manager import u_embeddings, sessions, users
from sklearn.metrics.pairwise import cosine_similarity

def session_recommendation(user_id):
    if u_embeddings.get(user_id) is None:
        return cold_start(user_id)
    
    from application.routes.api.get_recommendation import session_ids
    
    if session_ids.get(user_id) is None:
        session_ids[user_id] = uuid.uuid4().hex

    user_embedding = u_embeddings[user_id]
    session = sessions[user_id].get(session_ids[user_id])

    if session is None:
        return cold_start(user_id)
    
    gama = (cosine_similarity(np.array(session).reshape(1, -1), np.array(user_embedding).reshape(1, -1))[0][0] + 1 ) / 2
    recommendation = gama * np.array(session) + (1 - gama) * np.array(user_embedding)

    print("Recommendation done")

    similar_posts_list = similar_posts(user_id, recommendation, 5)
    return similar_posts_list