# HW3_DataOrganizer.py
# sort data from MNIST dataset into desired sets 

# TEXT FILE KEY: 
# training set: 400 each of 0 and 1 images, randomized
# test set: 100 each of 0 and 1 images
# challenge set: 100 each of 2-9 images

import linecache as lc
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
images0 = open("MNIST dataset\\images0.txt", "r")  
images1 = open("MNIST dataset\\images1.txt", "r")
images2 = open("MNIST dataset\\images2.txt", "r")
images3 = open("MNIST dataset\\images3.txt", "r")
images4 = open("MNIST dataset\\images4.txt", "r")
images5 = open("MNIST dataset\\images5.txt", "r")
images6 = open("MNIST dataset\\images6.txt", "r")
images7 = open("MNIST dataset\\images7.txt", "r")
images8 = open("MNIST dataset\\images8.txt", "r")
images9 = open("MNIST dataset\\images9.txt", "r")

# make a list of the image files to more easily iterate through them to make sets
images = [images0, images1, images2, images3, images4, images5, images6, images7, images8, images9]

# take 400 images from each image txt file, 
# & put them in a single file, trainset.txt, of 4,000 images total. 
# randomize the order. keep track of labels in trainset_labels.txt.

# create files to save the training set and its labels
trainset_f = open("HW3\\trainset.txt", "w")
trainset_labels_f = open("HW3\\trainset_labels.txt", "w")

# initialize lists for training set and its labels
trainset = []
trainset_labels = []

# create files to save the test set and its labels
testset_f = open("HW3\\test_set.txt", "w")
testset_labels_f = open("HW3\\test_set_labels.txt", "w")

# intialize lists for test set and its labels
testset = []
testset_labels = []

# create and fill array with a nested array for each image
# track labels in corresponding array
images_full = []
images_full_labels = []

for i in range(0, 10):  
    images_full += images[i].readlines(); 
    images_full_labels += [i] * 500
    
    # create training set, 400 of each digit (4,000 total)
    # (will randomize after for loop, when list is filled)
    trainset += images_full[i * 500 : i * 500 + 400]
    trainset_labels += [i] * 400
    
    # create test set, remaining 100 of each digit (1,000) total
    testset += images_full[i * 500 + 400 : ]
    testset_labels += [i] * 100

# randomize training set
zip_trainset = list(zip(trainset, trainset_labels))
random.shuffle(zip_trainset)
trainset, trainset_labels = zip(*zip_trainset)
trainset, trainset_labels = list(trainset), list(trainset_labels)

# save training set and labels to .txt files
writeFileSet(trainset_f, trainset_labels_f, trainset, trainset_labels)

# save test set and labels to .txt files
writeFileSet(testset_f, testset_labels_f, testset, testset_labels)
