#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_OCC GO"
echo "-----"

MODE="CARLA_OCC"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_OCC GO DONE"
echo "----------"

