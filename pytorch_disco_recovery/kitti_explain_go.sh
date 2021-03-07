#!/bin/bash

set -e # exit on error

echo "-----"
echo "KITTI_EXPLAIN GO"
echo "-----"

MODE="KITTI_EXPLAIN"
export MODE
python -W ignore main.py

echo "----------"
echo "KITTI_EXPLAIN GO DONE"
echo "----------"

