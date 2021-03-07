#!/bin/bash

set -e # exit on error

echo "-----"
echo "KITTI_FREE GO"
echo "-----"

MODE="KITTI_FREE"
export MODE
python -W ignore main.py

echo "----------"
echo "KITTI_FREE GO DONE"
echo "----------"

