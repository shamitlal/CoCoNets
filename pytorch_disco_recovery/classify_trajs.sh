#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_CLASSIFY_TRAJS GO"
echo "-----"

MODE="CARLA_FORECAST"
export MODE
python -W ignore classify_trajs.py

echo "----------"
echo "CARLA_CLASSIFY_TRAJS GO DONE"
echo "----------"

