# Frame LEvel Speech Classification

This work is a part of the second homework assignment for introduction to deep learning(CMU-11785) at [Class Link] (https://deeplearning.cs.cmu.edu/F20/index.html). In this challenge, we are asked to apply vanilla neural networks (MLP) to identify the phoneme state label for each frame in a dataset of audio recordings(utterances). The details of the challenge can be found at [kaggle] (https://www.kaggle.com/c/11-785-fall-20-homework-1-part-2/overview)

## DATA 

The data comes from librispeech that are read aloud and labelled using the original text. The training and the test data comprise of speech recodings and frame-level phoneme state labels. The data can be downloaded from [data] (https://www.kaggle.com/c/11-785-fall-20-homework-1-part-2/data). The data must be in the data directory. Please look at the directory structure below.


### DATA LOADING SCHEME

To process data efficientl, built-in numpy functions are used to load the data from memory. A class (data) is created that holds the training, development and testing data. Another class(myDataset) is created that uses the above class and pre-process the data by concatenating the frames in each clip and padds them with a user input context size from the top and the bottom.

## DEPENDENCIES

* python 3.6[python package](https://www.python.org/downloads/)
* torch [pytorch package] (https://github.com/pytorch/pytorch)
* numpy [numpy package] (https://numpy.org/install/)
* matplotlib [module link](https://matplotlib.org/) 

## MODEL ARCHITECTURE

In this work, a simple feed-forward neural network (MLP) is used with ReLU activations and Batchnorm layers. The input size is (2*context_size+1)*13 and the output size is fixed to 346. The architecture is flexible and takes an array with elements as the number of neurons in each layer. 

## DIRECTORY STRUCTURE

some directories are currently empty (i.e, saved_model, output, data). only data needs to be filled and the rest will be created automically if they do not exit. 

hw1_p2
|
|	README.txt
|	__init__.py
|	configs.py
|	data_utility.py
|	plot_utility.py
|	model.py
|	train.py
|	test.py
|	
|__saved_model
|	.pt 
|
|__output
|	.PNG 
|
|__data
|
|	train.npy
|	train_labels.npy
|	dev.npy
|	dev_labels.npy
|	test.npy
|	train_labels.npy


## HYPER-PARAMATERS 

Hyperparameters are set in configs.py 
* epochs 		: 10
* context_size  : 18
* num_workers   : 8
* weight_decay  : 0.0
* lr 			: 1e-4
* bins 			: 13
* output_size 	: 346
* input_size 	: (2*context_size+1)*bins
* model size    : [input_size,800,750,730,680,640,600,580,540,400,360,output_size]

## RUN

### TRAINING
1) set the hyperparameters in configs.py
2) inside train.py, change the model_size. (my best model is deep6_MLP, take a look at configs.py)
3) python train.py

### TESTING
1) pick the best saved model from saved_model
2) update the path to the model insdie test.py
1) python test.py

## Questions?
shamsbasir@gmail.com

Note: some of the codes are built on top the functions provided in the recitions 

