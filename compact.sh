#!/bin/bash

# Define the target directory
TARGET_DIR="results"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Initialize a counter
counter=1

# Process each directory starting with "results"
for dir in results*/; do
  # Skip if it's not a directory
  [ -d "$dir" ] || continue

  # Extract the base name of the directory (e.g., results_2024)
  base_name=$(basename "$dir")

  # Rename and move each file in the directory
  for file in "$dir"*; do
    if [ -f "$file" ]; then
      # Create the new file name with the format image*~ (e.g., image1_results_2024)
      new_name=$(printf "image%d_%s" "$counter" "$base_name")

      # Move and rename the file to the target directory
      mv "$file" "$TARGET_DIR/$new_name"

      # Increment the counter
      counter=$((counter + 1))
    fi
  done
done

echo "Files have been renamed and moved to $TARGET_DIR"
