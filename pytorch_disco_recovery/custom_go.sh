#!/bin/bash

set -e # exit on error

echo "-----"
echo "CUSTOM GO"
echo "-----"

MODE="CUSTOM"
export MODE
python -W ignore main.py

echo "----------"
echo "CUSTOM GO DONE"
echo "----------"

