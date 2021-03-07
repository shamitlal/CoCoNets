#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_GOODVAR GO"
echo "-----"

MODE="CARLA_GOODVAR"
export MODE
# python -W ignore main.py
python main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CARLA_GOODVAR GO DONE"
echo "----------"

