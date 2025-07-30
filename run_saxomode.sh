#!/bin/bash
# This script runs the saxomode segmentation pipeline for all patients in the ED_segmentation_data directory.
# It assumes that the saxomode package is installed and available in the Python environment.
# Usage: ./run_saxomode.sh

# Check if saxomode is installed
if ! python -c "import saxomode" &> /dev/null; then
    echo "saxomode is not installed. Please install it using pip or pipenv."
    exit 1
fi

# Define the directory containing the NIfTI files, change to your directory
DATA_DIR="/Users/inad001/Documents/SSCP25/segmented_data"

# Check if the directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Directory $DATA_DIR does not exist. Please check the path."
    exit 1
fi

# Loop through each patient directory in the data directory
for patient_dir in "$DATA_DIR"/*; do
    if [ -d "$patient_dir" ]; then
        echo "Processing patient directory: $patient_dir"
        # Get patient id by splitting directory name on last slash
        patient_id=$(basename "$patient_dir")
        echo "Patient ID: $patient_id"

        saxomode-pc --n $patient_id
        if [ $? -ne 0 ]; then
            echo "Segmentation failed for patient directory: $patient_id"
        else
            echo "Segmentation completed for patient directory: $patient_id"
        fi
        saxomode-createmesh --n $patient_dir
        if [ $? -ne 0 ]; then
            echo "Mesh creation failed for patient directory: $patient_id"
        else
            echo "Mesh created for patient directory: $patient_id"
        fi
    else
        echo "Skipping non-directory: $patient_dir"
    fi
done
