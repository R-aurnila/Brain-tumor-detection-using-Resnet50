# Brain Tumor Detection using ResNet50

This project is focused on detecting brain tumors using the **ResNet50** model for binary classification. The dataset used is from Kaggle and is designed to classify MRI images as either having a tumor or not. The model uses **ResNet50**, a deep convolutional neural network, for accurate image classification.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The goal of this project is to create a machine learning model that can accurately detect brain tumors from MRI images. Using the **ResNet50** model, this application performs binary classification to distinguish between images with and without a tumor.

## Dataset

The dataset used in this project is from Kaggle and contains MRI images of brain scans, labeled as either "tumor" or "no tumor". The dataset is used for training and evaluating the **ResNet50** model.

You can download the dataset from Kaggle:
- [Brain MRI Dataset on Kaggle](https://www.kaggle.com/datasets)

## Technologies Used

- **Deep Learning Framework**: TensorFlow, Keras
- **Model**: ResNet50
- **Dataset**: Kaggle Brain MRI Dataset
- **Programming Language**: Python
- **Libraries**: Numpy, Matplotlib, OpenCV, PIL

## Installation

### Prerequisites

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/brain-tumor-detection-using-resnet50.git
   cd brain-tumor-detection-using-resnet50
   ```
   
## Dataset: 
- Download the Brain MRI dataset from Kaggle

## Usage: 
1. Open the Jupyter notebook (brain_tumor_detection.ipynb) and run all the cells.
2. The notebook will preprocess the images, train the ResNet50 model, and evaluate its performance on the validation set.
3. After training, the model will classify new MRI images as either containing a tumor or not.
