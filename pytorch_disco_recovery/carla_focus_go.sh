#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_FOCUS GO"
echo "-----"

MODE="CARLA_FOCUS"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_FOCUS GO DONE"
echo "----------"

