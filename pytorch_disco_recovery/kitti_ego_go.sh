#!/bin/bash

set -e # exit on error

echo "-----"
echo "KITTI_EGO GO"
echo "-----"

MODE="KITTI_EGO"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "KITTI_EGO GO DONE"
echo "----------"

