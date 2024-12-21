#implementation 1

from application.app import app
from flask import request, jsonify
import uuid

from application.storage.storage_manager import recommendation_storage, users, save_json, users_file

@app.route('/get_recommendation', methods=['GET'])
def get_recommendation():
    user_id = request.cookies.get('user_id')

    resp = jsonify({}, 200)
    if not user_id:
        user_id = uuid.uuid4()
        resp.set_cookie('user_id', user_id)
        users[user_id] = {
            "liked_posts": [],
            "has_seen": []
        }
        save_json(users_file, users)

    if user_id not in users:
        users[user_id] = {
            "liked_posts": [],
            "has_seen": []
        }
        save_json(users_file, users)

    if user_id not in recommendation_storage:
        recommendation_storage[user_id] = []

    # implement recommendation storage 
    # collaborative and content based filtering

    if len(recommendation_storage[user_id]) == 0:
        recommendation_storage[user_id] = ["378ac5db84d44e9ab71349dfb518eb20", "346920a29e734578beddde34758d62b8", "a93c3a4e99954d998db332705a43993f"]

    recommendation = recommendation_storage[user_id].pop(0)
    users[user_id]["has_seen"].append(recommendation)
    save_json(users_file, users)
    resp.set_data(jsonify({"id": recommendation}).get_data())
    return resp;