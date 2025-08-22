import requests
import os
import time

subjects = [
    "Modern Indian History",
    "Sociology",
    "Political Science",
    "Economics",
    "English",
    "Legal Language",
    "History of Courts India",
    "Logic"
]

base_dir = "reference_texts"
search_url_template = "https://archive.org/advancedsearch.php?q={query}&fl[]=identifier&sort[]=downloads+desc&rows=10&page=1&output=json"

def download_text(archive_id, folder):
    for ext in [".txt", ".pdf"]:
        url = f"https://archive.org/download/{archive_id}/{archive_id}{ext}"
        r = requests.get(url)
        if r.status_code == 200:
            with open(os.path.join(folder, f"{archive_id}{ext}"), "wb") as f:
                f.write(r.content)
            print(f"Downloaded {archive_id}{ext} to {folder}/")
            return True
    print(f"Could not download TXT or PDF for {archive_id}")
    return False

for subject in subjects:
    folder = os.path.join(base_dir, subject.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    query = subject.replace(" ", "+")
    search_url = search_url_template.format(query=query)
    print(f"Searching for: {subject}")
    resp = requests.get(search_url)
    data = resp.json()
    ids = [doc["identifier"] for doc in data["response"]["docs"]]
    for archive_id in ids:
        download_text(archive_id, folder)
        time.sleep(1)  # Be polite to the server
