#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_PIPE GO"
echo "-----"

MODE="CARLA_PIPE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_PIPE GO DONE"
echo "----------"

