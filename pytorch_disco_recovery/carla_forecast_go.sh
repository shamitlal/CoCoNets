#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_FORECAST GO"
echo "-----"

MODE="CARLA_FORECAST"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_FORECAST GO DONE"
echo "----------"

