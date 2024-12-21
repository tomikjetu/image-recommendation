import os
import requests

downloaded =  f"../application/storage/downloaded"
api = "http://localhost:80/upload_image"

if not os.path.exists(downloaded):
    print(f"Directory {downloaded} does not exist")
    exit()

for filename in os.listdir(downloaded):
    if not filename.endswith(".jpg"):
        continue

    with open(os.path.join(downloaded, filename), "rb") as f:
        files = {"image": (filename, f, "image/jpeg")}
        response = requests.post(api, files=files)
    if response.status_code == 200:
        print(f"Uploaded {filename}")
        os.remove(os.path.join(downloaded, filename))
    else:
        print(f"Failed to upload {filename}: {response.text}")