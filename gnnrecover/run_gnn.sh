#!/bin/bash

# Exit on any error
set -e

# Change to the script's directory
cd "$(dirname "$0")"

# Array of all dataset names supported by GraphDecoding/src/utils/datasets.py
DATASETS=(
    "moon"
    "circles"
    "spiral"
    "swissroll2d"
    "scurve2d"
    "clusters"
    "grid"
    "ring"
    "line"
    "wave"
    "adult"
)

echo "Starting evaluation on all datasets..."

for dataset in "${DATASETS[@]}"
do
    echo ""
    echo ">>> Running dataset: $dataset <<<"
    python main_all_dataset_gnn.py --dataset "$dataset"
done

echo ""
echo "All datasets processed successfully! Images are saved in the 'imgs' directory."
