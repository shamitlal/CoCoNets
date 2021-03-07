#!/bin/bash

set -e # exit on error

echo "-----"
echo "KITTI_SIAMESE GO"
echo "-----"

MODE="KITTI_SIAMESE"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "KITTI_SIAMESE GO DONE"
echo "----------"

