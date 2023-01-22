`Agent_detect.py` is the main file to be executed.

`resnet_policy.py` has the policy definition.

`Yolo_detect.py` has the detector.

`utils.py` has utilities for `Agent_detect.py` and `utils.py` has all utilities for `Yolo_detect.py`


# Explanation
## src_phase2/cfg/ :
- This directory contains values of all the parameters that are being used in the code. One can easily tune the values from here and run the executable without building it for every change.

## src_phase2/data :
- This directory contains `.names` files. These files contain the names of the classes that are used for classification. These classes are loaded into other files like `Agent_detect.py`.

## src_phase2/ssd_pytorch/data/scripts/VOC2007.sh :
- This script file creates a directory named `data` and downloads the VOC2007 training and test data inside it. It removes the tar files after extracting the data from them.

## src_phase2/ssd_pytorch/data/__init__.py :
- This file has a definition of a custom collate function for dealing with batches of images that have different numbers of associated object annotations(bounding boxes).
- This file also contains a class named BaseTransform whose one attribute function base_transform returns an image after resizing a given image with given dimension and data type.

## src_phase2/Agent_detect.py :
- This file is of the RL agent which changes the image brightness according to the image which is the state.
- This does so to get the maximum performance from a pre-trained network.
- YOLO has been used in this case.
- The agent network is ResNet18 with the REINFORCE algorithm. (ResNet18 stands for Residual Networks which is a 72-layer architecture with 18 deep layers.)
