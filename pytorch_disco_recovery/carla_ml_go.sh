#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_ML GO"
echo "-----"

MODE="CARLA_ML"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CARLA_ML GO DONE"
echo "----------"

