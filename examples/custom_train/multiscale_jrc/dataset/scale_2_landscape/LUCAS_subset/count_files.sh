#!/bin/bash

for dir in "$1"/*/; do
    echo "$(basename "$dir"): $(find "$dir" -type f | wc -l)"
done

