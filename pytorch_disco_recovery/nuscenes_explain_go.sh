#!/bin/bash

set -e # exit on error

echo "-----"
echo "NUSCENES_EXPLAIN GO"
echo "-----"

MODE="NUSCENES_EXPLAIN"
export MODE
python -W ignore main.py

echo "----------"
echo "NUSCENES_EXPLAIN GO DONE"
echo "----------"

