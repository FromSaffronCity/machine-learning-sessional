# convolutional-neural-network-from-scratch  

This repository contains all the programs coded for the assignment on *Convolutional Neural Network (CNN) Implementation from Scratch* **(Offline-3)**. The following model components of a convolutional neural network are implemented from scratch for image classification tasks in this assignment.  

- Convolution Layer  
- Activation Layer  
- Max Pooling Layer  
- Flattening Layer  
- Fully Connected Layer  
- Softmax Layer  

The forward and backward functions are implemented for each of these components so that any model architecture containing these components can be trained with backpropagation algorithm. **[MNIST](http://yann.lecun.com/exdb/mnist/)** and **[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)** datasets are used in this assignment for image classification.  



## navigation  

- `inputdir/` folder contains input model description file `model.txt` providing architecture of the CNN model.  
- `spec/` folder contains tasks specification for this particular assignment.  
- `src/` folder contains a Jupyter notebook (with `.ipynb` extension) and a Python script (with `.py` extension obviously) both containing implementation of convolutional neural network from scratch and code demonstrating image classification using this implementation.  



## `model.txt` description  

The input file `model.txt` contains description of model architecture which will be trained and later tested for image classification tasks. This file is located inside `inputdir/` folder.  

```
Conv 6 5 1 2    # Convolution Layer with 6 output filters, 5 kernel size, 1 stride, 2 padding
ReLU			# ReLU Activation Layer
Pool 2 2		# Max Pooling Layer with 2 kernel size, 2 stride
Conv 12 5 1 0	# Convolution Layer with 12 output filters, 5 kernel size, 1 stride, 0 padding
ReLU			# ReLU Activation Layer
Pool 2 2		# Max Pooling Layer with 2 kernel size, 2 stride
Conv 100 5 1 0  # Convolution Layer with 100 output filters, 5 kernel size, 1 stride, 0 padding
ReLU			# ReLU Activation Layer
Flatten			# Flattening Layer
FC 10			# Fully Connected Layer with 10 output dimension
Softmax			# Softmax Layer
```



## getting started  

In order to run the Python script, place `cnn_from_scratch.py` file inside a workspace folder. Create or place `inputdir/` folder inside the same workspace folder and place `model.txt` file inside `inputdir/`. You may need to install some Python modules beforehand. Run `cnn_from_scratch.py` inside the workspace folder for running the main program. You can set whether to use **MNIST** or **CIFAR-10** dataset by changing the value of `use_mnist` variable and configure other hyperparameters inside the Python script before running the main program. `outputdir/` folder will be created inside the aforementioned workspace folder once the model training is completed. This folder will contain `.csv` files presenting model performance stats for validation and test sets.  



## model performance for MNIST dataset  

The model was trained on `5000` training samples (`500` samples from each class) in each epoch. Then, the model was validated on `500` validation samples (`50` samples from each class) after each epoch. And finally, the best model (picked based on validation macro F1 score) was tested on `500` testing samples (`50` samples from each class). We set the number of training samples in single batch to `32`, number of training epochs to `10`, and learning rate to `0.001`.  



### validation  

| Epoch | CE Loss            | Accuracy | F1 Score            |
| ----- | ------------------ | -------- | ------------------- |
| 1     | 1080.53269699127   | 0.25     | 0.2205189516365987  |
| 2     | 991.8722843337455  | 0.392    | 0.3638412020922     |
| 3     | 887.8236990036569  | 0.504    | 0.48758759124211776 |
| 4     | 760.9422298225697  | 0.622    | 0.6123688858936503  |
| 5     | 625.1948829164891  | 0.71     | 0.7043708242336805  |
| 6     | 509.425751783077   | 0.76     | 0.7567537263250593  |
| 7     | 423.18191937681075 | 0.796    | 0.7918893455179207  |
| 8     | 362.99098067395215 | 0.814    | 0.8115469031050193  |
| 9     | 320.3539615091214  | 0.826    | 0.825522759690996   |
| 10    | 288.4852366871264  | 0.858    | 0.8573268857651335  |



### test  

| CE Loss            | Accuracy | F1 Score            |
| ------------------ | -------- | ------------------- |
| 1085.4704546905032 | 0.23     | 0.20250234666333272 |



## model performance for CIFAR-10 dataset  

The model was trained on `5000` training samples (`500` samples from each class) in each epoch. Then, the model was validated on `500` validation samples (`50` samples from each class) after each epoch. And finally, the best model (picked based on validation macro F1 score) was tested on `500` testing samples (`50` samples from each class). We set the number of training samples in single batch to `32`, number of training epochs to `10`, and learning rate to `0.001`.  



### validation  

| Epoch | CE Loss            | Accuracy | F1 Score            |
| ----- | ------------------ | -------- | ------------------- |
| 1     | 1143.7400965790039 | 0.11     | 0.09056537133717127 |
| 2     | 1136.4174656730415 | 0.12     | 0.10105536001763307 |
| 3     | 1129.9323534978148 | 0.136    | 0.11638990863679743 |
| 4     | 1122.64513895273   | 0.144    | 0.12560818396920126 |
| 5     | 1113.930552693876  | 0.19     | 0.17401483335027573 |
| 6     | 1103.2684067829482 | 0.2      | 0.18322457188806082 |
| 7     | 1090.5999898703353 | 0.218    | 0.2009464073985583  |
| 8     | 1075.891488475983  | 0.224    | 0.2058528654386992  |
| 9     | 1059.6252799214892 | 0.25     | 0.23084888722128089 |
| 10    | 1043.0351998524284 | 0.252    | 0.23489475054096495 |



### test  

| CE Loss            | Accuracy | F1 Score          |
| ------------------ | -------- | ----------------- |
| 1149.5501991406443 | 0.114    | 0.097090178186342 |

