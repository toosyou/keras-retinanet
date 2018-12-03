#!/bin/bash
rm -rf logs/*.mip1070
TF_CPP_MIN_LOG_LEVEL=2 python3 train.py\
        --multi-gpu 2\
        --batch-size 4\
        --multi-gpu-force\
        --steps 100\
        --epochs 200\
        --backbone p3d\
        --no-evaluation\
        --no-weights\
        lung

#--gpu 0\
#         --multi-gpu 2\
        #--multi-gpu-force\
