#!/bin/bash

BASE=output

OUT_DIR="$BASE/all3"
IMAGE_PATH="images/cow.jpg"
TRAIN_DIR="images/train"

IMAGE_URL='https://get.pxhere.com/photo/grass-field-farm-meadow-prairie-food-cow-cattle-herd-pasture-grazing-livestock-mammal-agriculture-milk-farmland-close-up-graze-outdoors-bull-grassland-habitat-dairy-rural-area-dairy-cow-natural-environment-cattle-like-mammal-texas-longhorn-royalty-free-images-981776.jpg'

# Check if JSON file is provided
if [ -z "$1" ]; then
    echo "Usagee: $0 <config_json>"
    exit 1
fi

CONFIG_JSON="$1"

# Create necessary directories
mkdir -p "$BASE" "$OUT_DIR" "$TRAIN_DIR"

# Check if the image already exists
# if [ ! -f "$IMAGE_PATH" ]; then
#     echo "Downloading image..."
#     echo "$IMAGE_URL"
#     wget -O "$IMAGE_PATH" "$IMAGE_URL"
# 
#     if [ $? -eq 0 ]; then  # Check if wget was successful
#         magick "$IMAGE_PATH" -crop 800x600+2000+1400 "$TRAIN_DIR/vaca.jpg"
#         magick "$IMAGE_PATH" -crop 800x600+0+0 "$TRAIN_DIR/cielo.jpg"
#         magick "$IMAGE_PATH" -crop 800x600+0+2000 "$TRAIN_DIR/pasto.jpg"
#     else
#         echo "Failed to download image."
#         exit 1  # Exit with an error code
#     fi
# else
#     echo "Image already exists, skipping download."
# fi

# Run the Python script
python3 -u python/ej_2.py "$TRAIN_DIR" "$IMAGE_PATH" "$OUT_DIR" "$CONFIG_JSON"
# python3 -u python/ej_2_results.py $OUT_DIR/metrics.csv $OUT_DIR/metrics_processed.csv
