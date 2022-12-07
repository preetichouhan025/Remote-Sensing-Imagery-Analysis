# Landscape Classification (COMP6721_GROUP_N) #
Remote Sensing Imagery Analysis using Convolutional Neural Networks for Landscape Classification and Insights

## Project Overview ##

<p align="justify">
Satellite image classification is a challenging problem in the computer vision and remote sensing domain due to the sheer volume of complex feature-rich data being generated which needs to be analyzed and classified. The traditional classification techniques have been rendered unreliable to solve this problem. It is necessary to consider spatial and spectral resolution along with general geometric characteristics of the imagery. Furthermore, natural phenomena such as clouds can obscure the subject matter. Therefore, project implements and studies the performance of three convolutional neural network (CNN) architectures - VGG16, ResNet50, and EfficientNet in classifying images from three satellite image datasets - Land-use Scene Classification, EuroSat and Satellite RSI-CB256. 

As a result, these models can be used in real-time to monitor the effectiveness of the sustainable development policy by allowing a network of remote sensing tools to automatically classify, analyse, and report on land use. 

**Note**: Please refer to Main branch as final submission for the project. 
</p>

## Steps to predict classes with trained model weights ##

* Access the drive having all trained model weights along with sample test images : [ COMP6721_GROUP_N Drive ](https://drive.google.com/drive/folders/1pVE89-GnIktZOd2Te4tHndG6wkYxyA3T?usp=share_link)
* Add Shortcut to the drive in your account so that it can accessed in google colab via mounting.
* Once a shortcut to the drive is added in your account, open the [ Predict_Test_Images.ipynb file](https://github.com/SabaSalehi/LandscapeClassification/blob/main/Predict_Test_Images.ipynb).
* Run the cells and follow the steps mentioned in the jupyter notebook to predict/test images. 

## The major components of this project are: ##

- Implementing multi-class classification using different CNN Model architectures
- Performing hyperparameter tuning
- Explanatory Data Analysis, Data augmentation and perfromance metrics
- Implementing transfer learning
- Model performance analysis and optimization
- Detailed results study and comparison 

## Dataset Samples ##

#### 1) Satellite RSI-CB256 Dataset samples
![satellite_samples](https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/satEx.png)

#### 2) EUROSAT Dataset samples
![eurosat_samples](https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/EuEx.png)

#### 3) Land-Use Dataset samples
![landuse_samples](https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/landEx.png)

## Model Training Results on three aforementioned datasets ##

<p float="left">
  <img src="https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/Val_ACC_overall.JPG" width="500" />
  <img src="https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/Val_loss_overall.JPG" width="500" /> 
</p>

## VGG16, ResNet50 and EfficientNet_B0 performance ##

<p float="left">
  <img src="https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/VGG16_performance.jpeg" width="300" />
  <img src="https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/ResNet50_performance.jpeg" width="300" /> 
  <img src="https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/Enet_performance.jpeg" width="300" />
</p>

## Training Score ##
![ACC_LOSS](https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/Model_results1.JPG)
![f1score](https://github.com/chouhanpreeti/COMP6721_GROUP_N/blob/main/readme_images/Model_results2.JPG)

