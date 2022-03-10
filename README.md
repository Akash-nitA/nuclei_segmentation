# Objective
Automated cell nuclei segmentation is the most crucial step toward the implementation of a computer-aided diagnosis system for cancer cells. Studies on the automated analysis of cytology pleural effusion images are few because of the lack of reliable cell nuclei segmentation methods. The segmentation step is an important step toward automatic image analysis. This step aims to discriminate between the foreground (the desired object) and the background of the image. The objective of this stage is to extract the cell nuclei from the entire image. 
## About the architecture used
![image](https://user-images.githubusercontent.com/76721146/157659513-6558a44d-8bdb-47cc-9338-dcaa22a49008.png)
Unets are convolution based Deep neural network models, specifically designed for biomedical image segmentation. It has two parts. One convolution based contraction part and one upsampling part. In this model, we train a model to map cell images to their corresponding mask.
