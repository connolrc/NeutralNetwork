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

images = open("MNISTnumImages5000_balanced.txt", "r")
labels = open("MNISTnumLabels5000_balanced.txt", "r")

# ~500 images each
images0 = open("images0.txt", "w+")  
images1 = open("images1.txt", "w+")
images2 = open("images2.txt", "w+")
images3 = open("images3.txt", "w+")
images4 = open("images4.txt", "w+")
images5 = open("images5.txt", "w+")
images6 = open("images6.txt", "w+")
images7 = open("images7.txt", "w+")
images8 = open("images8.txt", "w+")
images9 = open("images9.txt", "w+")

row = 1

# fill txt files for 0, 1, 7, and 9
for line in images: 
    label = lc.getline("MNISTnumLabels5000_balanced.txt", row)
    
    if label.rstrip() == "0": 
        images0.write(line)
    elif label.rstrip() == "1": 
        images1.write(line)
    elif label.rstrip() == "2": 
        images2.write(line)
    elif label.rstrip() == "3": 
        images3.write(line)
    elif label.rstrip() == "4": 
        images4.write(line)
    elif label.rstrip() == "5": 
        images5.write(line)
    elif label.rstrip() == "6": 
        images6.write(line)
    elif label.rstrip() == "7": 
        images7.write(line)
    elif label.rstrip() == "8": 
        images8.write(line)
    elif label.rstrip() == "9": 
        images9.write(line)
    
    row += 1
    
images0.close()
images1.close()
images2.close()
images3.close()
images4.close()
images5.close()
images6.close()
images7.close()
images8.close()
images9.close()

#### MAKE THE TRAINING SET ####
# take 400 points from images0 and images1, 
# & put them in a single file, training_set, of 800 points. 
# randomize the order. keep track of labels in training_set_labels.
training_set = open("training_set.txt", "w")
training_set_labels = open("training_set_labels.txt", "w")

training_set_list = []
training_set_labels_list = []

images0 = open("images0.txt", "r")
images1 = open("images1.txt", "r")

images0_list_full = images0.readlines()
images0_list = images0_list_full[0:400]

images1_list_full = images1.readlines()
images1_list = images1_list_full[0:400]

training_set_list = images0_list + images1_list
training_set_labels_list = [0] * 400 + [1] * 400

zipped01 = list(zip(training_set_list, training_set_labels_list))
random.shuffle(zipped01)
training_set_list_rand, training_set_labels_list_rand = zip(*zipped01)
training_set_list_rand, training_set_labels_list_rand = list(training_set_list_rand), list(training_set_labels_list_rand)
    
writeFileSet(training_set, training_set_labels, training_set_list_rand, training_set_labels_list_rand)
    

#### MAKE THE TEST SET ####
# take the remaining 100 points from the 0 and 1 files, 
# and make another file with these 200 images, 
# again keeping track of which ones are 0 or 1.
# # (it is not necessary to randomize the order here)

test_set = open("test_set.txt", "w")
test_set_labels = open("test_set_labels.txt", "w")

test_set_list = images0_list_full[400:] + images1_list_full[400:]
test_set_labels_list = [0] * len(images0_list_full[400:]) + [1] * len(images1_list_full[400:])

writeFileSet(test_set, test_set_labels, test_set_list, test_set_labels_list)

test_set.close()
test_set_labels.close()


#### MAKE CHALLENGE SET ####
# select 100 points each from all the other sets (2 through 9), 
# and combine them into a single set, 
# making sure to keep track of which inputs are which digit.
# don't need to randomize this set. 

challenge_set = open("challenge_set.txt", "w")
challenge_set_labels = open("challenge_set_labels.txt", "w")

images2 = open("images2.txt", "r")
images3 = open("images3.txt", "r")
images4 = open("images4.txt", "r")
images5 = open("images5.txt", "r")
images6 = open("images6.txt", "r")
images7 = open("images7.txt", "r")
images8 = open("images8.txt", "r")
images9 = open("images9.txt", "r")

images2_list = images2.readlines()
images3_list = images3.readlines()
images4_list = images4.readlines()
images5_list = images5.readlines()
images6_list = images6.readlines()
images7_list = images7.readlines()
images8_list = images8.readlines()
images9_list = images9.readlines()

images2_list = random.sample(images2_list, 100)
images3_list = random.sample(images3_list, 100)
images4_list = random.sample(images4_list, 100)
images5_list = random.sample(images5_list, 100)
images6_list = random.sample(images6_list, 100)
images7_list = random.sample(images7_list, 100)
images8_list = random.sample(images8_list, 100)
images9_list = random.sample(images9_list, 100)

challenge_set_list = images2_list + images3_list + images4_list + images5_list + images6_list + images7_list + images8_list + images9_list 
challenge_set_labels_list = [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100 + [6] * 100 + [7] * 100 + [8] * 100 + [9] * 100

writeFileSet(challenge_set, challenge_set_labels, challenge_set_list, challenge_set_labels_list)