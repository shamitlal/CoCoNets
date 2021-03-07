#!/bin/bash

FPS=24
# THING="CameraRGB11_vis"
# THING="CameraRGB10_vis"
THING="CameraRGB14_vis"
OUTPUT="${THING}.mp4"

# ffmpeg -framerate $FPS -pattern_type glob -i "${THING}*.png" -s:v 256x256 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $OUTPUT
# ffmpeg -framerate $FPS -pattern_type glob -i "${THING}*.png" -s:v 512x256 -c:v libx264 -profile:v high -crf 1 -pix_fmt yuv420p $OUTPUT
ffmpeg -framerate $FPS -pattern_type glob -i "${THING}*.png" -s:v 768x256 -c:v libx264 -profile:v high -crf 1 -pix_fmt yuv420p $OUTPUT

open ${THING}.mp4
