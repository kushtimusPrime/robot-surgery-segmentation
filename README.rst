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

Overview
--------
Semantic segmentation of robotic instruments is an important problem for the robot-assisted surgery. One of the main challenges is to correctly detect an instrument's position for the tracking and pose estimation in the vicinity of surgical scenes. Accurate pixel-wise instrument segmentation is needed to address this challenge. Our approach demonstrates an improvement over the state-of-the-art results using several novel deep neural network architectures. It addressed the binary segmentation problem, where every pixel in an image is labeled as an instrument or background from the surgery video feed. In addition, we solve a multi-class segmentation problem, in which we distinguish between different instruments or different parts of an instrument from the background. In this setting, our approach outperforms other methods in every task subcategory for automatic instrument segmentation thereby providing state-of-the-art results for these problems.

Data
----
The training dataset consists of 8 |times| 225-frame sequences of high resolution stereo camera images acquired from a `da Vinci Xi surgical system`_ during several different porcine procedures. Training sequences are provided with 2 Hz frame rate to avoid redundancy. Every video sequence consists of two stereo channels taken from left and right cameras and has a 1920 |times| 1080 pixel resolution in RGB format. The articulated parts of the robotic surgical instruments, such as a rigid shaft, an articulated wrist and claspers have been hand labelled in each frame. Furthermore, there are instrument type labels that categorize instruments in following categories: left/right prograsp forceps, monopolar curved scissors, large needle driver, and a miscellaneous category for any other surgical instruments.

.. class:: center

    |gif1| |gif2|
    |br|
    |gif3| |gif4|
    |br|
    Original sequence (top left). Binary segmentation, 2-class (top right). Parts, 3-class (bottom left). Instruments, 7-class (bottom right)

Method
------
We evaluate 4 different deep architectures for segmentation: `U-Net`_, 2 modifications of `TernausNet`_, and a modification of `LinkNet`_. The output of the model is a pixel-by-pixel mask that shows the class of each pixel. Our winning submission to the MICCAI 2017 Endoscopic Vision Sub-Challenge uses slightly modified version of the original U-Net model.

As an improvement over U-Net, we use similar networks with pre-trained encoders. TernausNet is a U-Net-like architecture that uses relatively simple pre-trained VGG11 or VGG16 networks as an encoder:

.. figure:: images/TernausNet.png
    :scale: 65 %

|br|
|br|

LinkNet model uses an encoder based on a ResNet-type architecture. In this work, we use pre-trained ResNet34. The decoder of the network consists of several decoder blocks that are connected with the corresponding encoder block. Each decoder block includes 1 |times| 1 convolution operation that reduces the number of filters by 4, followed by batch normalization and transposed convolution to upsample the feature map:

.. figure:: images/LinkNet34.png
    :scale: 72 %

Training
--------

We use Jaccard index (Intersection Over Union) as the evaluation metric. It can be interpreted as a similarity measure between a finite number of sets. For two sets A and B, it can be defined as following:

.. raw:: html

    <figure>
        <img src="images/iou.gif" align="center"/>
    </figure>

Since an image consists of pixels, the expression can be adapted for discrete objects in the following way:

.. figure:: images/jaccard.gif
    :align: center

where |y| and |y_hat| are a binary value (label) and a predicted probability for the pixel |i|, respectively.

Since image segmentation task can also be considered as a pixel classification problem, we additionally use common classification loss functions, denoted as H. For a binary segmentation problem H is a binary cross entropy, while for a multi-class segmentation problem H is a categorical cross entropy.

.. figure:: images/loss.gif
    :align: center

As an output of a model, we obtain an image, where every pixel value corresponds to a probability of belonging to the area of interest or a class. The size of the output image matches the input image size. For binary segmentation, we use 0.3 as a threshold value (chosen using validation dataset) to binarize pixel probabilities. All pixel values below the specified threshold are set to 0, while all values above the threshold are set to 255 to produce final prediction mask. For multi-class segmentation we use similar procedure, but we assign different integer numbers for each class.

Results
-------

For binary segmentation the best results is achieved by TernausNet-16 with IoU=0.836 and Dice=0.901. These are the best values reported in the literature up to now (`Pakhomov`_, `Garcia`_). Next, we consider multi-class segmentation of different parts of instruments. As before, the best results reveals TernausNet-16 with IoU=0.655 and Dice=0.760. For the multi-class instrument segmentation task the results look less optimistic. In this case the best model is TernausNet-11 with IoU=0.346 and Dice=0.459 for 7 class segmentation. Lower performance can be explained by the relatively small dataset size. There are 7 instrument classes and some of them appear just few times in the training dataset. Nevertheless, in the competition we achieved the best performance in this sub-category too.

.. raw:: html

    <figure>
        <img src="images/grid-1-41.png" width="60%" height="auto" align="center"/>
        <figcaption>Comparison between several architectures for binary and multi-class segmentation.</figcaption>
    </figure>
|
|
|

.. table:: Segmentation results per task. Intersection over Union, Dice coefficient and inference time, ms.

    ============= ========= ========= ========= ========= ========= ====== ========= ========= =======
    Task:         Binary segmentation           Parts segmentation         Instrument segmentation
    ------------- ----------------------------- -------------------------- ---------------------------
    Model         IOU, %    Dice, %   Time      IOU, %    Dice, %   Time     IOU, %  Dice, %   Time
    ============= ========= ========= ========= ========= ========= ====== ========= ========= =======
    U-Net         75.44     84.37     93.00     48.41     60.75     106    15.80     23.59     **122**
    TernausNet-11 81.14     88.07     142.00    62.23     74.25     157    **34.61** **45.86** 173
    TernausNet-16 **83.60** **90.01** 184.00    **65.50** **75.97** 202    33.78     44.95     275
    LinkNet-34    82.36     88.87     **88.00** 34.55     41.26     **97** 22.47     24.71     177
    ============= ========= ========= ========= ========= ========= ====== ========= ========= =======

Pre-trained weights for all model of all segmentation tasks can be found at `google drive`_

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
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install cuda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh
    source .bashrc
    conda create --name unet python=3.6
    conda activate unet
    conda install pytorch=0.4.1 cuda92 -c pytorch
    conda install torchvision=0.2
    pip3 install opencv-python==3.3.0.10 tqdm==4.19.4 albumentations==0.0.4
    sudo apt-get install libsm6 libxrender1 libfontconfig1
    # Installing the Dataset Folders
    sudo apt-get install zip unzip
    pip3 install gdown
    gdown <ID OF ZIP FILE IN GOOGLE DRIVE 1UoTJgaTK11skThXYRSYVGwIua8JIT8Mw> -O <Dataset.zip>
    unzip <Path to zip file>
    cd ~/
    git clone https://github.com/kushtimusPrime/robot-surgery-segmentation/
    cd robot-surgery-segmentation
    git checkout feature/submission
    python3 prepare_data.py
    chmod +x train.bash
    screen -m bash -c "./train.bash"
