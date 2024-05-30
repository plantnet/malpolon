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


# Function to display usage information
usage() {
  echo "Usage: $0 [-s|--src /path/to/source] [-e|--ext file_extension] [-h|--help]"
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
  echo "Source directory does not exist."
  exit 1
fi

# Process each .pt file in the source directory
for file in "$SOURCE_DIR"/*"$FILE_EXT"; do
  # Extract the filename without the path and extension
  filename=$(basename "$file" "$FILE_EXT")

  # Determine the length of the filename
  len=${#filename}

  if [ "$len" -ge 3 ]; then
    # Extract the last 2 digits for my_folder
    my_folder="${filename: -2}"

    if [ "$len" -gt 3 ]; then
      # Extract the 2 digits preceding the last 2 digits for my_subfolder
      my_subfolder="${filename: -4:2}"
    else
      # If filename is 3 digits long, my_subfolder is the digit preceding the last 2 digits
      my_subfolder="${filename:0:1}"
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
