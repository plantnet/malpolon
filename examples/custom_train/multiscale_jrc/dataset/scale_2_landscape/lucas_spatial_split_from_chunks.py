import pandas as pd
import numpy as np

from malpolon.data.utils import split_obs_spatially

FP_INPUT = 'chunks_fps.txt'


def main(fp_input: str,):
    """Run spatial split over multiple files.

    Parameters
    ----------
    fp_input : str
        Path to the input CSV file.
    """
    # Read the file paths from the text file
    with open(fp_input, 'r') as f:
        fp_chunks = [line.strip() for line in f.readlines()]

    for fp_chunk in fp_chunks:
        split_obs_spatially(fp_chunk, val_size=0.1,
                            col_lon='th_long', col_lat='th_lat')

if __name__ == "__main__":
    main(FP_INPUT)
    