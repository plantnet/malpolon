#!/bin/bash

# Input CSV file
INPUT_FILE="input.csv"

# Output CSV file
OUTPUT_FILE="filtered_output.csv"

# Columns to check (1-based index, separated by spaces)
COLUMNS_TO_CHECK="2 3 4 5 6 7"  # Replace with the actual column indices

# Temporary file for processing
TEMP_FILE="temp.csv"

# Extract the header and write it to the output file
head -n 1 "$INPUT_FILE" > "$OUTPUT_FILE"

# Process each row (excluding the header)
tail -n +2 "$INPUT_FILE" | while IFS=, read -r line; do
  # Extract the file paths from the specified columns
  FILE_PATHS=()
  for col in $COLUMNS_TO_CHECK; do
    FILE_PATH=$(echo "$line" | cut -d',' -f"$col")
    
    # Process the file path: split by '/', take the last 5 elements, and prepend "LUCAS/"
    MODIFIED_PATH=$(echo "$FILE_PATH" | awk -F'/' '{n=NF; print "LUCAS/"$(n-4)"/"$(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    FILE_PATHS+=("$MODIFIED_PATH")
  done

  # Check if any of the modified files exist
  FILES_EXIST=false
  for path in "${FILE_PATHS[@]}"; do
    if [ -f "$path" ]; then
      FILES_EXIST=true
      break
    fi
  done

  # If at least one file exists, keep the row
  if $FILES_EXIST; then
    echo "$line" >> "$OUTPUT_FILE"
  fi
done

# Clean up temporary file
rm -f "$TEMP_FILE"

echo "Filtered CSV saved to: $OUTPUT_FILE"
