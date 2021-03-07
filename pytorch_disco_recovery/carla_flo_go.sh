#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_FLO GO"
echo "-----"

MODE="CARLA_FLO"
export MODE
#CUDA_VISIBLE_DEVICES=2
python -W ignore main.py

echo "----------"
echo "CARLA_FLO GO DONE"
echo "----------"

