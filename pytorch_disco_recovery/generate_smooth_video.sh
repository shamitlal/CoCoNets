#!/bin/bash

FPS=24
# THING="01_m128x32x128_p64x192_F32i_Oc_c1_s1_V_d32_c1_E2_m1_e.1_n2_d32_E3_m1_e.1_n2_d16_quicktest9_ns_load09"
# THING="01_m128x32x128_p64x192_F32i_Oc_c1_s1_V_d32_c1_E2_m1_e.1_n2_d32_E3_m1_e.1_n2_d16_quicktest9_ns_load10"
THING="emb3D_g_concat"
OUTPUT="${THING}.mp4"

# ffmpeg -framerate $FPS -pattern_type glob -i "${THING}*.png" -s:v 256x256 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $OUTPUT
# ffmpeg -framerate $FPS -pattern_type glob -i "${THING}*.png" -s:v 512x256 -c:v libx264 -profile:v high -crf 1 -pix_fmt yuv420p $OUTPUT
ffmpeg -framerate $FPS -pattern_type glob -i "${THING}*.png" -s:v 512x512 -c:v libx264 -profile:v high -crf 1 -pix_fmt yuv420p $OUTPUT

open ${THING}.mp4
