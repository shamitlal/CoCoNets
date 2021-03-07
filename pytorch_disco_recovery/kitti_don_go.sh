#!/bin/bash

set -e # exit on error

echo "-----"
echo "KITTI_DON GO"
echo "-----"

MODE="KITTI_DON"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "KITTI_DON GO DONE"
echo "----------"

