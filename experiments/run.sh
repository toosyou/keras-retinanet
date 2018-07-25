#!/bin/bash
python3 train.py\
        --multi-gpu 2\
        --multi-gpu-force\
        --batch-size 4\
        --steps 100\
        --epochs 200\
        --backbone p3d\
        --no-evaluation\
        --no-weights\
        lung
