#!/bin/bash

set -e # exit on error

echo "-----"
echo "KITTI_MOC GO"
echo "-----"

MODE="KITTI_MOC"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "KITTI_MOC GO DONE"
echo "----------"

