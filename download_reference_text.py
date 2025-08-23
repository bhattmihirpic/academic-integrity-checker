import requests
import os

# User input: subject or keywords
subject_input = input("Enter the subject or keywords for the reference text: ").strip()
subject = subject_input.replace(' ', '+')

# Search Internet Archive for the subject
search_url = f"https://archive.org/advancedsearch.php?q={subject}&fl[]=identifier&rows=1&page=1&output=json"
resp = requests.get(search_url)
data = resp.json()

if not data['response']['docs']:
    print("No results found for this subject.")
    exit(1)

archive_id = data['response']['docs'][0]['identifier']
print(f"Found: {archive_id}")

# Try to download TXT file, fallback to PDF
subject_folder = os.path.join('reference_texts', subject_input.replace(' ', '_'))
os.makedirs(subject_folder, exist_ok=True)
for ext in ['.txt', '.pdf']:
    url = f"https://archive.org/download/{archive_id}/{archive_id}{ext}"
    r = requests.get(url)
    if r.status_code == 200:
        with open(os.path.join(subject_folder, f"{archive_id}{ext}"), "wb") as f:
            f.write(r.content)
        print(f"Downloaded {archive_id}{ext} to {subject_folder}/")
        break
else:
    print("Could not download TXT or PDF for this item.")
