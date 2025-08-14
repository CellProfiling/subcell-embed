import os
import requests
from datetime import datetime

### setup env variables
GITHUB_PAT = os.getenv("GHRS_GITHUB_API_TOKEN")
GITHUB_OWNER = os.getenv("TRAFFIC_ACTION_OWNER")
GITHUB_REPO = os.getenv("TRAFFIC_ACTION_REPO")


output_dir="traffic/"
os.makedirs(output_dir, exist_ok=True)
current_date = datetime.today().strftime('%Y-%m-%d')

# Define API endpoint URLs
endpoints = [
    "/repos/{}/{}/traffic/clones".format(GITHUB_OWNER, GITHUB_REPO),
    "/repos/{}/{}/traffic/popular/paths".format(GITHUB_OWNER, GITHUB_REPO),
    "/repos/{}/{}/traffic/popular/referrers".format(GITHUB_OWNER, GITHUB_REPO),
    "/repos/{}/{}/traffic/views".format(GITHUB_OWNER, GITHUB_REPO)
]

# Function to fetch data from GitHub API
def fetch_data(endpoint):
    url = "https://api.github.com" + endpoint
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_PAT}"
    }
    response = requests.get(url, headers=headers)
    return response.json()


# Loop through the endpoints and fetch traffic data
for endpoint in endpoints:
    data = fetch_data(endpoint)
    print(data)
    with open(os.path.join(output_dir, current_date, endpoint.split('/')[-1] + ".json"), "w") as f:
        json.dump(data, f)


# Write summary to a CSV file
with open(os.path.join(output_dir, "summary.csv"), "a") as summary_file:
    summary_file.write(
        "{},{},{},{},{}\n".format(
            current_date,
            json.loads(open(os.path.join(output_dir, current_date, "clones.json")).read())["count"],
            json.loads(open(os.path.join(output_dir, current_date, "clones.json")).read())["uniques"],
            json.loads(open(os.path.join(output_dir, current_date, "views.json")).read())["count"],
            json.loads(open(os.path.join(output_dir, current_date, "views.json")).read())["uniques"]
        )
    )