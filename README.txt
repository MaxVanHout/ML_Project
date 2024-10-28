# Project Title

Transfer Learning As An Approach To Facilitate Feature Extraction For Predicting Brain Responses To Visual Stimuli Using FMRI Data

## Description

The code for this project includes four files:

‘Model_NumPCs.py’: contains python code for the exploration of how many principal components will be included in further analysis. Running this code will also generate figure 3 of the scientific poster.

‘Model_Per_Layer.py’: contains python code to evaluate which layers of the VGG16 model would yield the most informative features. Due to computational cost, extracting features from multiple layers at a time was not possible on my machine. Running this code multiple times, changing the layer parameter in between runs yielded the data that was used to create figure 4.

‘Figure_4.py’: contains python code for the generation of figure 4 of the scientific poster.

‘Model_Per_Regression.py’: contains python code that evaluates the performance of different regression models. Running this code will also generate figure 5 of the scientific poster.

## Prerequisites and Usage

The code for this project was written in python 3.11  

Running the following command installs all necessary packages: pip install numpy matplotlib pandas scikit-learn tensorflow


