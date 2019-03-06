# CS 512 - Computer Vision

This repository contains assignments and project for CS512 during the term of Fall 2018

# Project details

The last decade has seen an astronomical shift from imaging with DSLR and point-and-shoot cameras to imaging with smartphone cameras. Due to the small aperture and sensor size, smartphone images have notably more noise than their DSLR counterparts. Using neural networks, we can denoise the image and improve its quality.

While denoising for smartphone images is an active research area, the research community currently lacks a denoising image dataset representative of real noisy images from smartphone cameras with high-quality ground truth. The Smartphone Image Denoising Dataset (SIDD), of ~30,000 noisy images from 10 scenes under different lighting conditions using five representative smartphone cameras and generated their ground truth images.

The dataset would consist of static scenes to avoid misalignments caused by scene motion. The images in the dataset is captured using smartphone cameras. 5 different smartphones have been used to capture each scene multiple times in different settings, and/or different lighting conditions. Each combination of these is called a scene instance.

The paper followed for this project deals with sonar based images and enhances the image by denoising. We try follow the same steps to denoise the images captured by smartphones. This would be done by using implementing 3 convolutional layers and 3 deconvolutional layers. The image will be increased artificially by 16 times in order to ensure the final image is not blurred.

Dataset: https://www.eecs.yorku.ca/~kamel/sidd/dataset.php

Technologies used:
1. Python 3.5
2. OpenCV
3. Keras

For more details go to the [Project Report](project/doc/Project%20Report.pdf)

Instructions to execute the code is available [here](project/doc/Instructions.pdf)

