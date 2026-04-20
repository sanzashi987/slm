#!/usr/bin/env bash
# Minimal FastMETRO FreiHAND training launcher (RTX 4090).
# All defaults in train_freihand.py are tuned for this setup. Edit the 3 paths below.

set -e

FREIHAND_DIR="./dataset/FreiHAND"
MANO_DIR="./models"
OUTPUT_DIR="./outputs/run1"

python -m src.train_freihand \
    --freihand_dir "$FREIHAND_DIR" \
    --mano_dir     "$MANO_DIR" \
    --resume       ./outputs/run1/checkpoint_latest.pth \
    --epochs       120 \
    --output_dir   "$OUTPUT_DIR"
