HW3_DataOrganizer.py: 
I wrote code in this file to create the training and test sets from the MINST data for this assignment. 


functions.py: 
Contains logistical, menial functions that I wrote mostly last assignment for modularity and  
convenience. I modified some of them to more specifically fit this assignment. 


problem1.py: 
This file is where I completed the Problem 1 tasks. It contains the NeuralNetwork class, which is 
what I used to implement this neural network. It has many methods, the driving one being run(), 
which runs the epochs and training. There are methods for calculating preactivation and activation
values, and the adjustWeights() method is quite important as well. I kept it generally well-
commented, so it should be descriptive. To run this, you will want to comment out or uncomment out 
the "main" code at the very bottom to fit your needs. The filepaths referenced for the .txt files
are written under the assumption that in your code editor, your workspace folder open contains a 
folder called HW3 in which all the immediate files referenced and created through this program 
reside. For example, "HW3/trainset.txt" is used to open the training set .txt file. 

problem2.py: 
This file is where I completed the Problem 2 tasks. It is the same cases as problem1.py. 