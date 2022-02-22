# CSE 151B PA3: How to run the code
There are a number of flags that can be used when running the code from the command line. You can see all of them in the 'main.py' file. The most important flag is the '--model' flag which specifies which model to run. The options are 'baseline', 'custom', 'vgg', and 'resnet'. For example, the command 'python main.py --model baseline' will run the baseline CNN model for the default number of epochs (25). Once the model goes through all the epochs, plots are created for the loss and accuracy values for each epoch for the training and validation sets. As another example, we ran our baseline model using: 'python main.py --model baseline --gamma 0.001', where the gamma flag specifies the multiplicative factor of the learning rate decay. The 'main.py' file is the file used to run the models and produce plots. The 'data.py' file is responsible for splitting the training dataset into training and validation sets, normalizing the test set, and creating the dataloaders to be used when training the model. The 'model.py' file holds the classes for each model. Finally, the 'engine.py' file contains functions for preparing, training, and testing the model.