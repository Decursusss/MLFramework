import requests
import json
import csv
import os

# Helping functions
def load_config(data_source_name:str):
    config = os.path.join(
        os.path.dirname(__file__),
        f"sources/{data_source_name}/config.json"
    )
    with open(config, "r") as f:
        config = json.load(f)
    return config

def fetch(url):
    request = requests.get(url)
    try:
        raw_data = request.json()
        return raw_data
    except json.decoder.JSONDecodeError:
        return None

#Helping functions to save and load data
def save_csv(filename, rows, columns, data_directory):
    path = os.path.join(os.path.dirname(__file__), data_directory, filename)
    directory_path = os.path.join(os.path.dirname(__file__), data_directory)
    os.makedirs(directory_path, exist_ok=True)
    #file_exists = os.path.isfile(path) use if need for append to not rewrite headers

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})

    print(f"Saved {len(rows)} rows to {data_directory}/{filename}")
    return path

def load_csv(data_directory, columns=None):
    path = os.path.join(os.path.dirname(__file__), data_directory)
    if not os.path.isfile(path):
        print("File not found in: ", path)
        return []

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            if columns:
                row = {k: row[k] for k in columns if k in row}
            rows.append(row)

        print(f"Loaded {len(rows)} rows from {data_directory}")
        return rows