#!/bin/bash

    for i in 0 1 2
    do
       python3 train.py --device-ids 0 --batch-size 3 --fold $i --workers 0 --lr 0.0001 --n-epochs 10 --jaccard-weight 1
    done
