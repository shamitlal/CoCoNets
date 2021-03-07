#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_GEN3DVQ GO"
echo "-----"

MODE="CARLA_GEN3DVQ"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CARLA_GEN3DVQ GO DONE"
echo "----------"

