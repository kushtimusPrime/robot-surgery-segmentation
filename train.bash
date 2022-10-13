#!/bin/bash

    for i in 0 1 2 3
    do
       python3 train.py --device-ids 0 --batch-size 16 --fold $i --workers 0 --lr 0.0001 --n-epochs 10 --jaccard-weight 1
       python3 train.py --device-ids 0 --batch-size 16 --fold $i --workers 0 --lr 0.00001 --n-epochs 20 --jaccard-weight 1
    done
