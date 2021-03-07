#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_COMPOSE GO"
echo "-----"

MODE="CARLA_COMPOSE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_COMPOSE GO DONE"
echo "----------"

