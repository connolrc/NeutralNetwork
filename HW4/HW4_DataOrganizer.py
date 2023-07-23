# HW3_DataOrganizer.py
# sort data from MNIST dataset into desired sets 

# TEXT FILE KEY: 
# training set: 400 each of 0 and 1 images, randomized
# test set: 100 each of 0 and 1 images
# challenge set: 100 each of 2-9 images

import linecache as lc
import numpy as np
from functions import *
import random

def writeFileSet(imagesFile, labelsFile, imagesList, labelsList): 
    for line in imagesList: 
        imagesFile.write(str(line))
    for line in labelsList: 
        labelsFile.write(str(line)+"\n")
        
    imagesFile.close()
    labelsFile.close()

# images = open("MNISTnumImages5000_balanced.txt", "r")
# labels = open("MNISTnumLabels5000_balanced.txt", "r")

# ~500 images each
images0 = open("MNIST dataset/images0.txt", "r")  
images1 = open("MNIST dataset/images1.txt", "r")
images2 = open("MNIST dataset/images2.txt", "r")
images3 = open("MNIST dataset/images3.txt", "r")
images4 = open("MNIST dataset/images4.txt", "r")
images5 = open("MNIST dataset/images5.txt", "r")
images6 = open("MNIST dataset/images6.txt", "r")
images7 = open("MNIST dataset/images7.txt", "r")
images8 = open("MNIST dataset/images8.txt", "r")
images9 = open("MNIST dataset/images9.txt", "r")

# make a list of the image files to more easily iterate through them to make sets
images = [images0, images1, images2, images3, images4, images5, images6, images7, images8, images9]
images = [img.readlines() for img in images]

# take 400 images from each image txt file, 
# & put them in a single file, trainset.txt, of 4,000 images total. 
# randomize the order. keep track of labels in trainset_labels.txt.

# create files to save the training set and its labels
trainset04_f = open("HW4/problem2/trainset04.txt", "w")
trainset04_labels_f = open("HW4/problem2/trainset04_labels.txt", "w")

# initialize lists for training set and its labels
trainset04 = []
trainset04_labels = []

# create files to save the test set for numbers 0-4 and its labels
testset04_f = open("HW4/problem2/testset04.txt", "w")
testset04_labels_f = open("HW4/problem2/testset04_labels.txt", "w")

# intialize lists for test set for numbers 0-4 and its labels
testset04 = []
testset04_labels = []

# create files to save the test set for numbers 5-9 and its labels
testset59_f = open("HW4/problem2/testset59.txt", "w")
testset59_labels_f = open("HW4/problem2/testset59_labels.txt", "w")

# intialize lists for test set for numbers 5-9 and its labels
testset59 = []
testset59_labels = []

# make training set of 400 digits each of 0-4, 
# and test set of 100 digits each of 0-4
for i in range(0, 5):  
    idxs = list(np.random.choice(range(500), 400, replace=False))
    for idx in range(500): 
        if idx in idxs: 
            trainset04 += [images[i][idx]]
        else: 
            testset04 += [images[i][idx]]
    trainset04_labels += [i] * 400
    testset04_labels += [i] * 100
    
# make test set of 100 digits each of 5-9
for i in range(5, 10): 
    idxs = list(np.random.choice(range(500), 100, replace=False))
    for idx in idxs: 
        testset59 += [images[i][idx]]
    testset59_labels += [i] * 100

# randomize training set
zip_trainset04 = list(zip(trainset04, trainset04_labels))
random.shuffle(zip_trainset04)
trainset04, trainset04_labels = zip(*zip_trainset04)
trainset04, trainset04_labels = list(trainset04), list(trainset04_labels)

writeFileSet(trainset04_f, trainset04_labels_f, trainset04, trainset04_labels)
writeFileSet(testset04_f, testset04_labels_f, testset04, testset04_labels)
writeFileSet(testset59_f, testset59_labels_f, testset59, testset59_labels)
