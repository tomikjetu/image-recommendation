class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.liked_posts = []  
        self.has_seen = []  

    def like_post(self, post_id, post_class):
        if post_id not in self.liked_posts:
            self.liked_posts.append(post_id)