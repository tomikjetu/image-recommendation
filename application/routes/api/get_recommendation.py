#implementation 1

from application.app import app
from flask import request, jsonify
import uuid

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from application.storage.storage_manager import users, recommendation_storage, save_json, users_file, sessions, sessions_file, u_embeddings, u_embedding_file
from recommendation.similar_post import similar_posts
from recommendation.similar_user import similar_users
from recommendation.session_recommendation import session_recommendation


session_ids = {}

def close_user_session(user_id):
    global session_ids
    if not user_id in session_ids:
        return

    last_session = sessions[user_id].get(session_ids[user_id])
    if last_session is None:
        return
    if u_embeddings.get(user_id) is None:
        u_embeddings[user_id] = last_session
    else:
        user_embedding = last_session
        weight = 0
        for session in sessions.get(user_id).values():
            if session == last_session:
                continue
            beta = (cosine_similarity(np.array(last_session).reshape(1, -1), np.array(session).reshape(1, -1))[0][0] + 1) / 2
            user_embedding += beta * np.array(session)
            weight += beta
        user_embedding = user_embedding / weight
        u_embeddings[user_id] = user_embedding.tolist()

    session_ids[user_id] = uuid.uuid4().hex
    save_json(u_embedding_file, u_embeddings)

@app.route("/close_session", methods=['POST'])
def close_session():
    global session_ids
    user_id = request.cookies.get('user_id')
    close_user_session(user_id)
    return {'message': 'Session closed'}, 200

@app.route('/get_recommendation', methods=['GET'])
def get_recommendation():
    user_id = request.cookies.get('user_id')

    resp = jsonify({}, 200)
    if not user_id:
        user_id = uuid.uuid4().hex
        resp.set_cookie('user_id', user_id)
        users[user_id] = {
            "liked_posts": [],
            "has_seen": [],
            "updated": False
        }
        save_json(users_file, users)

    if user_id not in users:
        users[user_id] = {
            "liked_posts": [],
            "has_seen": [],
            "updated": False
        }
        save_json(users_file, users)

    if sessions.get(user_id) is None:
        sessions[user_id] = {}
        save_json(sessions_file, sessions)
    
    if user_id not in recommendation_storage:
        recommendation_storage[user_id] = []

    if len(recommendation_storage[user_id]) == 0:
        recommendation_storage[user_id] = session_recommendation(user_id)

    if(len(recommendation_storage[user_id])==0):
        resp.set_data(jsonify({"id": None}).get_data())
        return resp

    recommendation = recommendation_storage[user_id].pop(0)
    users[user_id]["has_seen"].append(recommendation)
    save_json(users_file, users)
    resp.set_data(jsonify({"id": recommendation}).get_data())
    return resp;