#!/bin/bash


for file in *_Palette_\(SSBM\).png; do
    # Extract the base name and the extension
    base="${file%_Palette_\(SSBM\).png}"
    ext="${file#${base}_}"
    
    # Convert the base name to uppercase
    uppercase_base=$(echo "$base" | tr '[:lower:]' '[:upper:]')
    
    # Construct the new file name
    newfile="${uppercase_base}.png"
    
    # Rename the file
    mv "$file" "$newfile"
    #echo "$newfile"
done
