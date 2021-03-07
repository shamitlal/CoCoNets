#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_OBJ GO"
echo "-----"

MODE="CARLA_OBJ"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_OBJ GO DONE"
echo "----------"

