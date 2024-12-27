from application.app import app
from flask import request
from application.storage.storage_manager import users, save_json, users_file

@app.route('/like', methods=['POST'])
def like():
    data = request.get_json()
    user_id = data.get('user_id')
    post_id = data.get('post_id')
    
    if not user_id or not post_id:
        return {'error': 'Invalid request'}, 400

    if user_id not in users:
        users[user_id] = {
            'liked_posts': [],
            'has_seen': []
        }
    user = users[user_id]

    if post_id not in user.liked_posts:
        user["liked_posts"].append(post_id)
        user["updated"] = True
        save_json(users_file, users)

    return {'message': 'Post liked'}, 200