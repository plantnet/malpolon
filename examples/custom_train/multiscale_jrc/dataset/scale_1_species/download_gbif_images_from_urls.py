import os
import requests
import pandas as pd

# Configuration
csv_file = 'PN_gbif_France_2005-2025_illustrated_CBN-med.csv'           # Replace with your actual CSV file
output_dir = 'Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500/'  # symbolic link to /data/zenith/share/GeoLifeCLEF2024
url_column = 'identifier'
id_column = 'gbifID'            # Column used for renaming the downloaded file

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the DataFrame
df = pd.read_csv(csv_file)

# Iterate through the DataFrame rows
for index, row in df.iterrows():
    url = str(row.get(url_column, '')).strip()
    gbif_id = str(row.get(id_column, '')).strip()

    if not url or not gbif_id:
        print(f"Skipping row {index}: Missing URL or gbifID")
        continue

    # Extract the file extension from the original URL
    original_filename = os.path.basename(url)
    extension = os.path.splitext(original_filename)[1]
    if not extension:
        extension = '.jpeg'  # Default to .jpeg if no extension found

    new_filename = f"{gbif_id}{extension}"
    filepath = os.path.join(output_dir, new_filename)

    if os.path.exists(filepath):
        print(f"Skipping {new_filename} (already exists)")
        continue

    print(f"Downloading {url} as {new_filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)
        print(f"Saved to {filepath}")
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")

