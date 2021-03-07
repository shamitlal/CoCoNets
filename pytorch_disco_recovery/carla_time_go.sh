#!/bin/bash

set -e # exit on error

echo "SOURCING ~/.bashrc"
source ~/.bashrc

echo "-----"
echo "CARLA_TIME GO"
echo "-----"

MODE="CARLA_TIME"
export MODE
#CUDA_VISIBLE_DEVICES=2
python -W ignore main.py

echo "----------"
echo "CARLA_TIME GO DONE"
echo "----------"

