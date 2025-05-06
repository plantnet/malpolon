from pathlib import Path
import pandas as pd
import numpy as np

from malpolon.data.utils import split_obs_spatially

FP_INPUT = 'lucas_harmo_cover_exif.csv'


def to_chunks(fp_input: str, n: int):
    """Split a CSV file into n chunks and saves them as separate CSV files.

    Parameters
    ----------
    fp_input : str
        Path to the input CSV file.
    n : int
        Number of chunks to split the DataFrame into.
    """
    # Read the CSV file
    df = pd.read_csv(fp_input)

    # Sort the DataFrame by coordinates so that the chunks are spatially small
    df_sorted = df.sort_values(by=['th_long', 'th_lat'], ascending=[True, True])  # Ascending order for both columns

    # Split into n chunks
    chunks = np.array_split(df_sorted, n)

    # Print each chunk
    fp_chunks = []
    for i, chunk in enumerate(chunks):
        fp = f'{str(Path(fp_input).stem)}_chunk_{i}.csv'
        print(f"Exporting chunk {i} to {fp}")
        chunk.to_csv(fp, index=False)
        fp_chunks.append(fp)
    with open('chunks_fps.txt', 'w') as f:
        for fp in fp_chunks:
            f.write(f"{fp}\n")
    return fp_chunks

if __name__ == "__main__":
    n_chunks = 10  # Number of chunks to split the DataFrame into
    fp_chunks = to_chunks(FP_INPUT, n_chunks)