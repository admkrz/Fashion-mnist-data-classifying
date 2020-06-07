# Fashion-mnist data classifying
## Introduction
The Fashion-MNIST Data Set is for use in benchmarking machine learning algorithms in image analysis. It contains 70,000 28x28 pixel grayscale images of fashion articles divided in 10 categories. The training set contains 60,000 images (6000 from each category) and the test set 10,000 images (1000 from each category). A sample from the data set is visualized in the image below:

![](https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png)

When working with this data set, the feature variables will be the grayscale values for each of the 784 pixels in a given image, loaded either as a 784x1 vector or as a 28x28 array. The target variable is a number from 0 to 9, representing the category of the each item.

The problem that is solved by this program is to classify images in the test set into their true categories as accurate as possible, based on what the model learned from the training set.
## Methods
In this project, there are two models applied to the fashion-mnist data.

### KNN algorithm

Firstly, the program contains K-Nearest-Neighbour algorithm, which I prepared for the previous class on my university.
It takes 54000 images from the training set for the train data and 6000 images for the validation data, which it uses to choose the best value of the k parameter of the model.

Source of this model implementation is on page: [KNeighbours algorithm](https://www.ii.pwr.edu.pl/~zieba/zad2_msid.pdf)

### CNN model

The second model, which is meant to output the best accuracy on the test data as possible is a convolutional neural network.

Before building a model, at the begining, the data set is preprocessed. The images are reshaped into 24x24 arrays, instead of 784 one-dimensional arrays. Then, the values of every pixel are converted to floats and scaled to the range 0 to 1. 
The categories, which originally are marked with an integer number 0-9, are converted to categorical arrays, which means that for example category number 3 is converted to array [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].

The model is build with the Sequential model from Keras library. There are multiple layers added to the model to achieve the best results. 

Then it is trained with the 54000 images from the train data set and validated with the remaining 6000 images from the train data set.

Finally, the trained model is evaluated on the 10000 test data set, which gives us the final accuracy value of the model

Sources of inspiration and information for building cnn model:

## Results
### KNN model
After selecting the best knn model, it turns out that the best k is XX:

[WYKRES]

When the model with that k is tested on the test data, we get the accuracy of the model: **XXX**

It is significantly less than the accuracy presented in the benchmark table:

| Name | Accuracy |
|------|----------|
| MyKNN | XXX |
| KNeighborsClassifier | 0,86 |

There could be a few explanations for this difference in the results, but I think that the main two are: the difference in measuring the distance between two sets of images - my algorithm uses hamming distance, while the KNeighboursClassifier uses manhattan distance, the second one is that the algorithm from scikit-learn library weights all neighbours with thier distance to each other.

Of course the biggest difference in these algorithms is that the scikit-learn one is far more and better optimized and written than mine.

### CNN model

After selecting training cnn model, it turns out that the best model was in epoch XX, with the validation accuracy=:

[WYKRES]

When the best model is tested on the test data, we get the accuracy of the model: **XXX**

It is far more than the accuracy of the best algorithm from scikit-learn library:

| Name | Accuracy |
|------|----------|
| MyCNN | XXX |
| SVC | 0,897 |


## Usage
The fashion-mnist data for the project is downloaded automatically.
To run the project it is required to use Python 3.6 interpreter or newer, with following libraries installed:
1. numpy
2. matplotlib
3. scikit-learn
4. tensorflow
5. keras

You also have to install all of these libraries dependencies.

To train and test models presented in the results section, you have to run the main.py file from this repository.

If you want to skip the training process and get just the test results from the best, trained models, you have to comment line 136 in knn.py and 37 in cnn.py - they are responsible for training the new models, if they're commented, the models will be loaded from files in the repository.
