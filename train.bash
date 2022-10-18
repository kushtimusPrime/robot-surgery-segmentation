#!/bin/bash

    for i in 0 1 2
    do
       python3 train.py --device-ids 0 --batch-size 1 --fold $i --workers 0 --lr 0.0001 --n-epochs 4 --jaccard-weight 1
       python3 train.py --device-ids 0 --batch-size 1 --fold $i --workers 0 --lr 0.00001 --n-epochs 6 --jaccard-weight 1
       python3 train.py --device-ids 0 --batch-size 1 --fold $i --workers 0 --lr 0.000001 --n-epochs 8 --jaccard-weight 1
       python3 train.py --device-ids 0 --batch-size 1 --fold $i --workers 0 --lr 0.0000001 --n-epochs 10 --jaccard-weight 1
       python3 train.py --device-ids 0 --batch-size 1 --fold $i --workers 0 --lr 0.00000001 --n-epochs 12 --jaccard-weight 1
    done
