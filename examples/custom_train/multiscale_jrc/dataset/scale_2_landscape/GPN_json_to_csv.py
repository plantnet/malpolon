import json
import pandas as pd
from pathlib import Path


def load_habitat_dataframe(folder_path):
    folder = Path(folder_path)

    records = []

    for json_file in folder.glob("response_*.json"):
        # --- Extract surveyId from filename ---
        # format: response_<surveyId>.json
        survey_id = json_file.stem.replace("response_", "")

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            habitats = data.get("habitats", [])

            # Safe extraction (avoid index errors)
            lvl1 = habitats[0]["code"] if len(habitats) > 0 else None
            lvl2 = habitats[1]["code"] if len(habitats) > 1 else None
            lvl3 = habitats[2]["code"] if len(habitats) > 2 else None
            lvl1_name = habitats[0]["name"] if len(habitats) > 0 else None
            lvl2_name = habitats[1]["name"] if len(habitats) > 1 else None
            lvl3_name = habitats[2]["name"] if len(habitats) > 2 else None

        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            lvl1 = lvl2 = lvl3 = None

        records.append({
            "surveyId": survey_id,
            "habitats_lvl1": lvl1,
            "habitats_lvl2": lvl2,
            "habitats_lvl3": lvl3,
            "habitats_lvl1_name": lvl1_name,
            "habitats_lvl2_name": lvl2_name,
            "habitats_lvl3_name": lvl3_name,
        })

    df = pd.DataFrame(records)

    return df


if __name__ == "__main__":
    folder = "./GPN_API_habitats"  # change to your folder
    df = load_habitat_dataframe(folder)

    print(df.head())

    # Optional: save to CSV
    df.to_csv("GPN_API_habitats/response.csv", index=False)
