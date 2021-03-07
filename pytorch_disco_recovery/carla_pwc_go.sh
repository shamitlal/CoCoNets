#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_PWC GO"
echo "-----"

MODE="CARLA_PWC"
export MODE
#CUDA_VISIBLE_DEVICES=2
python -W ignore main.py

echo "----------"
echo "CARLA_PWC GO DONE"
echo "----------"

