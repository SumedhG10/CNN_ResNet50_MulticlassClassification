# Multiclass Ball Classification using ResNet50 Architecure.


This model walks through the demonstration of ResNet50 model

The demonstration is based on knowledge gained from one task (source task) to a different but similar task (target task) in short a Transfer Learning Process.

When Transfer Learning is done we generally used a pretrained model and add the FCN (Fully Connected Layer)

ResNet50 model was used considering the trained on more than 1 million images from the ImageNet database. Just like VGG-19, it can classify up to 1000 objects and the network was trained on 224x224 pixels colored images.(https://arxiv.org/abs/1512.03385)

## Dataset

Data can be procured from [Kaggle](https://www.kaggle.com/gpiosenka/balls-image-classification)

The data set contains train, test , validation folders for all the balls specified.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tensorflow.

```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install cv2 
```

## Data Augmentation
### Why I have used Data Augmentation techniques?

Data Augmentation helps to expand the size of training dataset by adding variations to the same data. E.g If the training dataset have the horses head towards right direction but while comparing it with the real world imagery which has a horse image in it with heads in a left direction the model wont be able to predict well.

Different types data augmentation techniques can be used in order to maake model trained on unseen scenerios.

In the cell below I am creating batches and vclass mode is categorical because we have to classify it into different number of class. For this dataset we have 24 classes.

## Model Preparation
1) Used ResNet50 as a base model and kept the FCN empty which is created using few dense layers
2) Global Average pooling is added because it gives single value for feature map and keeps the dimensions as it is
e.g we have dimension of i/p layer as (H X W X D) , it takes the Global Average across Height and Width and gives you a tensor with dimensions of (1 x D)

3) One of the activation function used in ReLu (Rectified Linear Unit):
--> a) It interinterspersed nonlinearity between many of the convolution layers

b) If value is negative it will be given as zero (0) and varies till one

c)The main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time

4) Softmax Activation at last
Since this is multiclass classification and it has 24 classes , softmax function is used because softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class label

## Featured Notebooks/Analysis/Deliverables
* [SumedhG10/CNN_ResNet50_MulticlassClassification/Resne50_multiclass_ball](https://github.com/SumedhG10/CNN_ResNet50_MulticlassClassification/blob/master/Resne50_multiclass_ball.ipynb)


## Contributor

**Team Leads (Contacts) : [Sumedh Ghatage](https://github.com/SumedhG10)**
						  [LinkedIn](www.linkedin.com/in/sumedh-ghatage)
						  [Email](sumedhghatage10@gmail.com)
