#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_PRECOMPUTE GO"
echo "-----"

MODE="CARLA_PRECOMPUTE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_PRECOMPUTE GO DONE"
echo "----------"

