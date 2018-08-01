#!/bin/bash
mkdir valid_vis
python3 evaluate.py\
            --max-detections 3\
            --score-threshold 0.05\
            --save-path valid_vis\
            --backbone p3d\
            --index -1\
            --convert-model $1
# CUDA_VISIBLE_DEVICES=""
