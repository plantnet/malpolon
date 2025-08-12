#!/bin/bash

SRC_DIR="./"          # Folder containing JPEGs
DEST_DIR="./output_png"    # Folder to store PNGs

mkdir -p "$DEST_DIR"

for img in "$SRC_DIR"/*.{jpg,jpeg,JPG,JPEG}; do
    [ -e "$img" ] || continue  # Skip if no matching files
    base=$(basename "$img")
    name="${base%.*}"
    convert "$img" "$DEST_DIR/$name.png"
done
