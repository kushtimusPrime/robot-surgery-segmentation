===========================================
MICCAI 2017 Robotic Instrument Segmentation
===========================================

Here we present our wining solution and its improvement for `MICCAI 2017 Robotic Instrument Segmentation Sub-Challenge`_.

In this work, we describe our winning solution for MICCAI 2017 Endoscopic Vision Sub-Challenge: Robotic Instrument Segmentation and demonstrate further improvement over that result. Our approach is originally based on U-Net network architecture that we improved using state-of-the-art semantic segmentation neural networks known as LinkNet and TernausNet. Our results shows superior performance for a binary  as well as for multi-class robotic instrument segmentation. We believe that our methods can lay a good foundation for the tracking and pose estimation in the vicinity of surgical scenes.

.. contents::

Team members
------------
`Alexey Shvets`_, `Alexander Rakhlin`_, `Alexandr A. Kalinin`_, `Vladimir Iglovikov`_

Citation
----------

If you find this work useful for your publications, please consider citing::

    @inproceedings{shvets2018automatic,
    title={Automatic Instrument Segmentation in Robot-Assisted Surgery using Deep Learning},
    author={Shvets, Alexey A and Rakhlin, Alexander and Kalinin, Alexandr A and Iglovikov, Vladimir I}},
    booktitle={2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA)},
    pages={624--628},
    year={2018}
    }

Dependencies
------------

* Python 3.6
* PyTorch 0.4.0
* TorchVision 0.2.1
* numpy 1.14.0
* opencv-python 3.3.0.10
* tqdm 4.19.4

To install all these dependencies you can run
::
    pip install -r requirements.txt



How to run
----------

1. Preprocessing
~~~~~~~~~~~~~~~~~~~~~~
As a preprocessing step we cropped black unindormative border from all frames with a file ``prepare_data.py`` that creates folder ``data/cropped_train.py`` with masks and images of the smaller size that are used for training. Then, to split the dataset for 4-fold cross-validation one can use the file: ``prepare_train_val``.


2. Training
~~~~~~~~~~~~~~~~~~~~~~
The main file that is used to train all models -  ``train.py``.

Running ``python train.py --help`` will return set of all possible input parameters.

To train all models we used the folloing bash script :

::

    #!/bin/bash

    for i in 0 1 2 3
    do
       python train.py --device-ids 0,1,2,3 --batch-size 16 --fold $i --workers 12 --lr 0.0001 --n-epochs 10 --type binary --jaccard-weight 1
       python train.py --device-ids 0,1,2,3 --batch-size 16 --fold $i --workers 12 --lr 0.00001 --n-epochs 20 --type binary --jaccard-weight 1
    done

