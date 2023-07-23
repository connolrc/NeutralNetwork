Modules used: 
I used matplotlib to plot my data, NumPy for various data structure manipulation, math for math, 
and pickle for saving NeuralNetwork class objects to use for other runs. 


HW4_DataOrganizer.py: 
I wrote code in this file to create the training and test sets from the MINST data for this assignment. 


functions.py: 
Contains logistical, menial functions that I have written across the assignments for convenience. 


hw4problem1.py: 
This file is where I completed the Problem 1 tasks. It is a slightly modified version of the problem1.py file from my HW3 submission. It contains the NeuralNetwork class, which is what I used 
to implement this neural network. It has many methods, the driving one being run(), 
which runs the epochs and training. There are methods for calculating preactivation and activation
values, and the adjustWeights() method is quite important as well. I kept it generally well-
commented, so it should be descriptive. To run this, you will want to comment out or uncomment out 
the "main" code at the very bottom to fit your needs. The filepaths referenced for the .txt files
are written under the assumption that in your code editor, your workspace folder open contains a 
folder called HW4 in which all the immediate files referenced and created through this program 
reside, but I also opened my training and test set files from my previous HW3 folder. For example, "HW3/trainset.txt" is used to open the training set .txt file, while case 1 training output results are saved to "HW4/case1_trainset_output.txt". 

hw4problem2.py: 
This file is where I completed the Problem 2 tasks. It is the same cases as problem1.py, but 
moderately modified to suite the operations of an autoencoder. It is also a modified version of
my HW3 file problem2.py, which in turn is a modified version of HW3's problem1. To be honest, I fell
behind with code comments on this file, so some of it might be a bit confusing, but since it's still
descended from problem1.py it should be mostly clear. Run it in the same way as the others.  

testing.py: 
I made a lot of the graphs in here. 