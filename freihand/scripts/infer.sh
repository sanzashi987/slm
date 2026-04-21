#!/usr/bin/env bash
# Run a trained FastMETRO checkpoint on images and export hand meshes as .obj files.
#
# Usage:
#   # Single image
#   bash scripts/infer.sh --input ./my_hand.jpg
#
#   # Folder of images (all *.jpg / *.png)
#   bash scripts/infer.sh --input ./dataset/FreiHAND/evaluation/rgb --limit 20
#
# Edit the three path variables below to match your setup.

set -e

CHECKPOINT="./outputs/run1/checkpoint_epoch099.pth"
MANO_DIR="./models"
OUTPUT_DIR="./exports"
# INPUT="./sample_0001.png"
INPUT="./input"
# INPUT="./sample_0001.jpg"
# INPUT="./dataset/FreiHAND/evaluation/rgb/00000045.jpg"

python -m src.export_obj \
    --checkpoint    "$CHECKPOINT" \
    --mano_dir      "$MANO_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --input         "$INPUT"
    # "$@"
