#!/bin/bash

set -e # exit on error

echo "-----"
echo "INTPHYS_FORECAST GO"
echo "-----"

MODE="INTPHYS_FORECAST"
export MODE
python -W ignore main.py

echo "----------"
echo "INTPHYS_FORECAST GO DONE"
echo "----------"

