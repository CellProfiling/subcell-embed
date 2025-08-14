import os
import requests
import json
from datetime import datetime

# --- Configuration ---
# Get environment variables set by the GitHub Actions workflow
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_FULL_NAME = os.getenv("GITHUB_REPOSITORY")

# Exit with an error if the required environment variables are not set
if not GITHUB_TOKEN or not REPO_FULL_NAME:
    print("Error: GITHUB_TOKEN and GITHUB_REPOSITORY environment variables are required.")
    print("Please run this script within a GitHub Actions workflow.")
    exit(1)

# Split "owner/repo" string into two parts
try:
    GITHUB_OWNER, GITHUB_REPO = REPO_FULL_NAME.split("/")
except ValueError:
    print(f"Error: GITHUB_REPOSITORY format is incorrect: '{REPO_FULL_NAME}'")
    exit(1)

# --- Setup Directories and Date ---
output_dir = "traffic"
current_date = datetime.today().strftime('%Y-%m-%d')
date_specific_dir = os.path.join(output_dir, current_date)

# Create both the base and the date-specific directories
os.makedirs(date_specific_dir, exist_ok=True)

# --- API Definitions ---
# Use a dictionary for clearer mapping and file naming
endpoints = {
    "clones": f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/traffic/clones",
    "paths": f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/traffic/popular/paths",
    "referrers": f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/traffic/popular/referrers",
    "views": f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/traffic/views"
}

headers = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {GITHUB_TOKEN}"
}

# --- Data Fetching and Saving ---
# Store data in memory to avoid reading files multiple times for the summary
clones_data = {}
views_data = {}

for name, endpoint in endpoints.items():
    url = "https://api.github.com" + endpoint
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)
        data = response.json()

        # Store data needed for the summary
        if name == "clones":
            clones_data = data
        elif name == "views":
            views_data = data

        # Save the raw data to a JSON file
        file_path = os.path.join(date_specific_dir, f"{name}.json")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved {name} data to {file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {name}: {e}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON for {name}. Response: {response.text}")

# --- Write Summary to CSV ---
summary_csv_path = os.path.join(output_dir, "summary.csv")
file_exists = os.path.isfile(summary_csv_path)

try:
    with open(summary_csv_path, "a", newline='') as summary_file:
        # Write a header row only if the file is being created for the first time
        if not file_exists:
            summary_file.write("date,clones_count,clones_uniques,views_count,views_uniques\n")
        
        # Append the new data row using the stored data
        # Use .get() to prevent errors if the keys are missing (e.g., no traffic)
        summary_file.write(
            f"{current_date},"
            f"{clones_data.get('count', 0)},"
            f"{clones_data.get('uniques', 0)},"
            f"{views_data.get('count', 0)},"
            f"{views_data.get('uniques', 0)}\n"
        )
    print(f"Successfully updated summary at {summary_csv_path}")
except Exception as e:
    print(f"Could not write to summary file: {e}")