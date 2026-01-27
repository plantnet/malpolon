#!/bin/bash

# Step 1: Find max width and height
max_width=0
max_height=0

for img in *.png;
do
    read w h <<< $(identify -format "%w %h" "$img")
    if (( w > max_width )); then max_width=$w; fi
    if (( h > max_height )); then max_height=$h; fi
done

echo "Max size: ${max_width}x${max_height}"

# Step 2: Pad all images to that size, centered
mkdir -p padded
for img in *.png;
do
    convert "$img" -gravity center -background none -extent "${max_width}x${max_height}" "padded/$img"
done

# Step 3: Create the GIF
convert -delay 300 -loop 0 -fuzz 10 padded/*.png -layers OptimizeTransparency output.gif
