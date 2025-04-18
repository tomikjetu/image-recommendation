from application.app import app
from flask import request
from application.storage.storage_manager import users, save_json, users_file, sessions, sessions_file, p_embeddings

import uuid
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@app.route('/like', methods=['POST'])
def like():
    data = request.get_json()
    user_id = data.get('user_id')
    post_id = data.get('post_id')
    post_embedding  = p_embeddings.get(post_id)

    if(post_embedding is None):
        return {'error': 'Invalid post_id'}, 400
    
    if not user_id or not post_id:
        return {'error': 'Invalid request'}, 400

    if user_id not in users:
        users[user_id] = {
            'liked_posts': [],
            'has_seen': [],
            'updated': False
        }
    user = users[user_id]

    if post_id not in user["liked_posts"]:
        user["liked_posts"].append(post_id)
        user["updated"] = True
        save_json(users_file, users)

    if sessions.get(user_id) is None:
        sessions[user_id] = {}
        save_json(sessions_file, sessions)

    from application.routes.api.get_recommendation import session_ids

    if session_ids.get(user_id) is None:
        session_ids[user_id] = uuid.uuid4().hex

    if sessions[user_id].get(session_ids[user_id]) is None:
        sessions[user_id][session_ids[user_id]] = post_embedding
    else:
        session_embedding = np.array(sessions[user_id][session_ids[user_id]])
        post_embedding = np.array(post_embedding)
        similarity = cosine_similarity([session_embedding], [post_embedding])[0][0]
        a1 = (similarity + 1) / 2
        a2 = 1 - a1
        sessions[user_id][session_ids[user_id]] = (a1 * session_embedding + a2 * post_embedding).tolist()

    save_json(sessions_file, sessions)

    return {'message': 'Post liked'}, 200