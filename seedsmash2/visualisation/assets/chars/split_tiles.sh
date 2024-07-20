#!/bin/bash

# Define the width of each tile
tile_width=136

# Loop through all PNG images in the current directory
for img in *_Palette_\(SSBM\).png; do
  # Get the width and height of the image
  img_width=$(identify -format "%w" "$img")

  # Calculate the number of tiles
  num_tiles=$((img_width / tile_width))
  
  echo "${img%_Palette_\(SSBM\).png}"

  # Split the image into tiles of the specified width
  convert "$img" -crop ${tile_width}x188 +repage "${img%_Palette_\(SSBM\).png}_%d.png"
done
