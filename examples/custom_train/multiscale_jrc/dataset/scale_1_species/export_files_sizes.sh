#!/bin/bash

# Check if folder path is given
if [[ -z "$1" ]]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

# Folder containing JPEG images
folder="$1"
output_file="resolution_stats.txt"
total_width=0
total_height=0
count=0
progress=0

min_width=
max_width=
min_height=
max_height=

# Count total number of files in the directory (recursively)
nb_files=$(find "$folder" -type f | wc -l)

# Loop through JPEG images
for img in "$folder"/*.jpg "$folder"/*.jpeg; do
    if [[ -f "$img" ]]; then
        progress=$((progress + 1))
        percent=$((progress * 100 / nb_files))
        echo -ne "Processing: $percent% [$progress/$nb_files]\r"

        # Get width and height using ImageMagick
        dimensions=$(identify -format "%w %h" "$img" 2>/dev/null)
        if [[ $? -eq 0 ]]; then
            width=$(echo "$dimensions" | awk '{print $1}')
            height=$(echo "$dimensions" | awk '{print $2}')
            total_width=$((total_width + width))
            total_height=$((total_height + height))
            count=$((count + 1))

            # Initialize min/max values if unset
            if [[ -z "$min_width" || $width -lt $min_width ]]; then
                min_width=$width
            fi
            if [[ -z "$max_width" || $width -gt $max_width ]]; then
                max_width=$width
            fi
            if [[ -z "$min_height" || $height -lt $min_height ]]; then
                min_height=$height
            fi
            if [[ -z "$max_height" || $height -gt $max_height ]]; then
                max_height=$height
            fi
        fi
    fi
done

echo ""

# Calculate and print statistics
if [[ $count -gt 0 ]]; then
    mean_width=$(echo "$total_width / $count" | bc)
    mean_height=$(echo "$total_height / $count" | bc)

    echo "Mean resolution: ${mean_width} x ${mean_height}"
    echo "Min resolution: ${min_width} x ${min_height}"
    echo "Max resolution: ${max_width} x ${max_height}"
    echo "Total number of files: $nb_files"

    echo "Mean resolution: ${mean_width} x ${mean_height}" > "$output_file"
    echo "Min resolution: ${min_width} x ${min_height}" >> "$output_file"
    echo "Max resolution: ${max_width} x ${max_height}" >> "$output_file"
    echo "Total number of files: $nb_files" >> "$output_file"
    echo "Results exported to $output_file"
else
    echo "No JPEG images found in $folder."
fi

