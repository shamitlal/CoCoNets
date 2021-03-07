#!/bin/bash

set -e # exit on error

echo "-----"
echo "KITTI_ZOOM GO"
echo "-----"

MODE="KITTI_ZOOM"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "KITTI_ZOOM GO DONE"
echo "----------"

