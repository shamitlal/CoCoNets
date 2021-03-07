#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_RESOLVE GO"
echo "-----"

MODE="CARLA_RESOLVE"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CARLA_RESOLVE GO DONE"
echo "----------"

