#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_TRACK GO"
echo "-----"

MODE="CARLA_TRACK"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_TRACK GO DONE"
echo "----------"

