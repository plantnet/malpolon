#!/bin/bash
# Requires image files supporting transparency like PNG

# Step 1: Compute max dimensions
max_width=0
max_height=0

for img in *.png; do
    read w h <<< $(identify -format "%w %h" "$img")
    (( w > max_width )) && max_width=$w
    (( h > max_height )) && max_height=$h
done

echo "Max canvas: ${max_width}x${max_height}"

# Step 2: Pad and center all images
mkdir -p padded
for img in *.png; do
    convert "$img" -gravity center -background none -extent "${max_width}x${max_height}" "padded/$(basename "$img")"
done

# Step 3: Build frames with blur only on the immediately previous image
mkdir -p frames
images=(padded/*.png)
total=${#images[@]}

for ((i=0; i<total; i++)); do
    curr="${images[$i]}"
    frame="frames/frame_$(printf "%03d" "$i").png"

    if (( i == 0 )); then
        # First frame: only the first image, no blur
        cp "$curr" "$frame"
    else
        prev="${images[$((i-1))]}"
        # Blur only the previous image, overlay current
        convert "$prev" -blur 0x8 matted.png
        convert matted.png "$curr" -gravity center -composite "$frame"
        rm matted.png
    fi
done

# Step 4: Create GIF with one frame per image, no extra frames
convert -delay 80 -loop 0 frames/frame_*.png final_output.gif
