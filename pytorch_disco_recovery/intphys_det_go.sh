#!/bin/bash

set -e # exit on error

echo "-----"
echo "INTPHYS_DET GO"
echo "-----"

MODE="INTPHYS_DET"
export MODE
python -W ignore main.py

echo "----------"
echo "INTPHYS_DET GO DONE"
echo "----------"

