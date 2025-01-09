#!/bin/bash

FOLDER="./data"
OUTPUT_FILE="dataset.zip"
SERVER_URL="https://dataverse.harvard.edu"
PERSISTENT_ID="doi:10.7910/DVN/N8ITTG"

mkdir -p "$FOLDER"
echo "Created folder: $FOLDER"

echo "Downloading data..."
wget ${SERVER_URL}/api/access/dataset/:persistentId/?persistentId=${PERSISTENT_ID} -O $OUTPUT_FILE

echo "Unzipping data..."
unzip $OUTPUT_FILE -d $FOLDER

echo "Download completed and data transferred to ${FOLDER}!"

read -p "Do you want to remove ${OUTPUT_FILE}? (y/n): " remove_data

if [ "$remove_data" == "y" ]; then
    rm $OUTPUT_FILE
    echo "Removed ${OUTPUT_FILE}"
fi

exit 0
