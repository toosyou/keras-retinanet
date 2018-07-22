#!/bin/bash
python3 train.py\
        --gpu $1\
        --multi-gpu 0\
        --multi-gpu-force\
        --batch-size 1\
        --steps 200\
        --no-weights\
        --backbone resnet18\
        lung ./
