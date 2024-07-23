#!/bin/bash

# This script re-organizes files in one folder into folders and sub-folders in the same way as for the GeoLifeCLEF challenge.
# That is to say in the following manner.
#
# Each file is re-arranged in folders and sub-folders in the following way:
# A file named 'ABCDWXYZ.pt' located at 'root_path/'  will be moved to
# 'root_path/YZ/WX/ABCDWXYZ.pt'.
#
# Each file name must be at least 3 characters long. For instance:
# A file named 'XYZ.pt' located at 'root_path/'  will be moved to
# 'root_path/YZ/X/XYZ.pt'.
#
# Author: Theo Larcher <theo.larcher@inria.fr>


## Define constants & functions
# Directory containing the .pt files
SOURCE_DIR="./"
FILE_EXT=".jpeg"
STRING_OFFSET=0
PARSING_CHAR="_"
PARSING_POS=-0


# Function to display usage information
usage() {
  echo "Usage: $0 [-s|--src /path/to/source] [-e|--ext file_extension] [-o|--offset string offset value] [--parsing_char character to parse filenames with] [--parsing_pos pick the string between parsing_char at the position provided by this argument (-0 = last position)] [-h|--help]"
  exit 1
}

## Parse named arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      exit 0
      ;;
    -s|-src)
      SOURCE_DIR="$2"
      shift
      ;;
    -e|--ext)
      FILE_EXT="$2"
      shift
      ;;
    -o|--offset)
      STRING_OFFSET="$2"
      shift
      ;;
    --parsing_char)
      PARSING_CHAR="$2"
      shift
      ;;
    --parsing_pos)
      PARSING_POS="$2"
      shift
      ;;
    *)
      echo "Invalid option: $1" >&2
      usage
      ;;
  esac
  shift
done


## Run the file re-organization process
# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source directory does not exist."/
  exit 1
fi

# Process each .pt file in the source directory
for file in "$SOURCE_DIR"/*"$FILE_EXT"; do
  # Extract the filename without the path and extension
  filename=$(basename "$file" "$FILE_EXT")
  parsed_filename=$(echo "$filename" | awk -F $PARSING_CHAR '{print $(NF'"$PARSING_POS"')}')

  # Determine the length of the filename
  len=${#parsed_filename}

  if [ "$len" -ge 3 ]; then
    # Extract the last 2 digits for my_folder
    my_folder="${parsed_filename: -2-$STRING_OFFSET:2}"

    if [ "$len" -gt 3 ]; then
      # Extract the 2 digits preceding the last 2 digits for my_subfolder
      my_subfolder="${parsed_filename: -4-$STRING_OFFSET:2}"
    else
      # If filename is 3 digits long, my_subfolder is the digit preceding the last 2 digits
      my_subfolder="${parsed_filename:0:1}"
    fi

    # Create the target directory if it does not exist
    target_dir="$SOURCE_DIR/$my_folder/$my_subfolder"
    mkdir -p "$target_dir"

    # Move the file to the target directory
    mv "$file" "$target_dir/"
  else
    echo "Filename $filename is too short to process."
  fi
done

echo "Files have been re-arranged successfully."
