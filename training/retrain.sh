#!/bin/bash
# Automated retraining script for scheduled execution

set -e  # Exit on error

echo "================================"
echo "Starting Model Retraining"
echo "================================"
echo "Timestamp: $(date)"

# Configuration
DATA_PATH="../data"
OUTPUT_DIR="./models"
VERSION=$(date +%Y%m%d_%H%M%S)
MIN_ACCURACY=0.90

# 1. Check if new data is available
echo ""
echo "Step 1: Checking for new data..."
# Add your data checking logic here
# For example, check if a new CSV file exists
# if [ ! -f "$DATA_PATH/new_data.csv" ]; then
#     echo "No new data found. Exiting."
#     exit 0
# fi

# 2. Train new model version
echo ""
echo "Step 2: Training new model version $VERSION..."
python train.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --version "$VERSION" \
    --epochs 5 \
    --batch-size 64

# 3. Check if training was successful
if [ ! -d "$OUTPUT_DIR/$VERSION" ]; then
    echo "ERROR: Training failed - model directory not created"
    exit 1
fi

# 4. Extract accuracy from metadata
echo ""
echo "Step 3: Evaluating new model..."
ACCURACY=$(python -c "
import json
with open('$OUTPUT_DIR/$VERSION/metadata.json') as f:
    metadata = json.load(f)
    print(metadata['metrics']['accuracy'])
")

echo "New model accuracy: $ACCURACY"

# 5. Compare with minimum threshold
if (( $(echo "$ACCURACY < $MIN_ACCURACY" | bc -l) )); then
    echo "WARNING: Model accuracy ($ACCURACY) below threshold ($MIN_ACCURACY)"
    echo "Model saved but NOT promoted to production"
    exit 0
fi

# 6. Promote to production if better
echo ""
echo "Step 4: Promoting model to production..."

# Copy to production directory
PRODUCTION_DIR="$OUTPUT_DIR/production"
rm -rf "$PRODUCTION_DIR"
cp -r "$OUTPUT_DIR/$VERSION" "$PRODUCTION_DIR"

echo "Model $VERSION promoted to production!"

# 7. Optional: Restart backend service
echo ""
echo "Step 5: Restarting backend service..."
# Uncomment if you want automatic restart
# cd ../backend
# pkill -f "uvicorn main:app" || true
# nohup uvicorn main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &

echo ""
echo "================================"
echo "Retraining Complete!"
echo "================================"
echo "Version: $VERSION"
echo "Accuracy: $ACCURACY"
echo "Status: Promoted to production"
echo "Timestamp: $(date)"
