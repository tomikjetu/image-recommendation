# 1 random from each: 
# top 1% most liked
# top 1% most seen
# top 1% least seen

from application.storage.storage_manager import posts
import random


def cold_start(): 
    # top_liked = posts.find().sort("likes", -1).limit(int(posts.count() * 0.01))
    # top_seen = posts.find().sort("views", -1).limit(int(posts.count() * 0.01))
    # least_seen = posts.find().sort("views", 1).limit(int(posts.count() * 0.01))

    # selected_posts = random.sample(list(top_liked), 1) + \
    #                  random.sample(list(top_seen), 1) + \
    #                  random.sample(list(least_seen), 1)

    # additional_posts = random.sample(list(posts.find()), 2)
    # selected_posts += additional_posts


    selected_posts = random.sample(list(posts), 5)

    selected_posts_ids = [post['id'] for post in selected_posts]

    return selected_posts_ids