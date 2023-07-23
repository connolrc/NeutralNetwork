import math
import random
import matplotlib.pyplot as plt
from functions import *

###############################
#### FUNCTION DEFINITIONS ####
#############################

# Simulate the binary threshold neuron shown in Slide 22 of Lecture 2. 


# i : each training step, which in this case is also each training image
# j : each individual input, each pixel of the image i
# s : s(t), net input to the neuron at time step t where xj(t) 
#       is the jth pixel value of the current input image
# y : y(t), the neuron's output. for the training epochs, it is equal to
#       z(t) 
# z : z(t), the teaching input. the labels text file is used as this!
# eta : learning rate, coefficient of weight adjustment 
# epochs : number of times to pass through the whole training set
def trainEpoch(x, z, w0, eta = .001, epochs = 41):
    numImages = len(x)     
    # list of net inputs for each image in the set, for every epoch
    s = [0.0] * numImages * epochs    
    # tracks index for s, since it's not 2D (because it's not cumulative across epochs)
    s_index = 0
    
    # pass original, randomly generated weights 
    # to w for training and updating
    w = w0
    
    # run through eps amount of epochs
    for eps in range(epochs): 
        
        # for each image in the training set...
        for i in range(numImages): 
            # Problem 1.2   
            
            # calculate this image's net input s[i] by adding together
            # each pixel's contribution to the net output s[i]
            for j in range(len(x[i])):
                s[s_index] += w[j] * x[i][j]   
                         
            # increment s_index before next image
            s_index += 1
            
            # Problem 1.3
            # post-synaptically gated Hebb Rule shown on Slide 42, Lecture 2
            # adjust the weights of the neuron 
            # no need to calculate more weights when this is the last image in the set
            #if i != numImages - 1: 
            w = [(w[j] + eta * z[i] * (x[i][j] - w[j])) for j in range(len(x[i]))]
        
    return s, w


# i : each training step, which in this case is also each training image
# j : each individual input, each pixel of the image i
# s : s(t), net input to the neuron at time step t where xj(t) 
#       is the jth pixel value of the current input image
# y : y(t), the neuron's output, determined by s(t) and theta. 
# theta : the values which to run the tests with when determining y(t). 
#       input as a range. (ex: runTest(set, weights, range(0, 41, 1)))
def runTest(x, w, theta_stop, theta_start = 0, theta_step = 1): 
    numImages = len(x)
    # list of net inputs for each image in the set, for every epoch
    s = [0.0] * numImages * math.ceil((theta_stop - theta_start) / theta_step)
    # tracks index for s, since it's not 2D (because it's not cumulative across epochs)
    s_index = 0
    
    # list of the program's attempted labels of the test set, based on its learning
    y = []
    
    # lambda function to determine the output, y(t)
    # returns 1 if net input value is greater than theta, or s(t) > theta
    # returns 0 otherwise
    yt = lambda st, th : 1 if st > th else 0
    
    # for each value of theta in the given range...
    for theta in range(theta_start, theta_stop, theta_step): 
        # for each image in the training set...
        for i in range(numImages): 
            # calculate this image's net input s[i] by adding together
            # each pixel's contribution to the net output s[i]
            for j in range(len(x[i])):
                s[s_index] += w[j] * x[i][j]   
                         
            # add the algorithm's decision on what number it's seeing 
            # to the output list
            y += [yt(s[s_index], theta)]
            
            # increment s_index before next image
            s_index += 1
            
    return y, s
    

###############################
#### "main" section below ####
#############################

### PROBLEM 1.3 ###

# get a 2D list of 1x784 lists of training set's images
tr_set = convertFileToList2D("training_set.txt")
# gets a list of the label for each sublist in tr_set
tr_set_labels = convertFileToList1D("training_set_labels.txt")

# initialize as random values from 0.0-0.5  
# original weights for net input, before training and updating
w0 = [random.uniform(0.0, 0.5) for xj in tr_set[0]]

# s(t) for training set
tr_s, tr_w = trainEpoch(tr_set, tr_set_labels, w0, 0.0075, 101)
writeListToFile("training_net_input.txt", tr_s)
writeListToFile("training_weights.txt", [tr_w])


### PROBLEM 1.4 ###

# read weights into 2D list
# (the [0] is only because I put tr_w in brackets in the 
# previous writeListToFile() call in order to write it 
# into the text file all on one line, like other images)
weight = convertFileToList2D("training_weights.txt")[0]

# open test set image file and labels file
test_set = convertFileToList2D("test_set.txt")
test_set_labels = convertFileToList1D("test_set_labels.txt")

test_y, test_s = runTest(test_set, weight, 41)
testRes = TestResults(test_y, test_set_labels)
testRes.printBestTestMetrics()

writeListToFile("test_output.txt", test_y, "%d")


### PROBLEM 1.5 ####
plt.plot(list(range(41)), testRes.recall, label = "Recall")
plt.plot(list(range(41)), testRes.precision, label = "Precision")
plt.plot(list(range(41)), testRes.F1_score, label = "F1 Score")
plt.legend(); 
plt.xlabel("Binary Threshold Theta")
#plt.xticks([0, 10, 20, 30, 40], [0, "", "", "", 40])
plt.title("Figure 1.5.1: Testing Metrics for a Simple Reinforcement Paradigm")
plt.figure()

# false positive rate = 1 - specificity
FPR = [1 - testRes.specificity[t] for t in range(len(testRes.specificity))]
plt.plot(FPR, testRes.recall)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Figure 1.5.2: ROC Curve for a Simple Reinforcement Paradigm")
plt.figure()

#### PROBLEM 1.6 ####
# plot weights before and after training as heat maps

#plt.title("Figure 1.6: Heatmaps of Weights with Post-Synaptic Gated Reinforcement\n")
plt.subplot(1, 2, 1)
plotImage(w0)
#plt.viridis()
plt.title("Figure 1.6.1:\nHeatmap of Weight\nBefore Training", loc = "center")
plt.subplot(1, 2, 2)
plotImage(tr_w)
#plt.bone()
plt.title("Figure 1.6.2:\nHeatmap of Weight\nAfter Training", loc = "center")
#plt.figure()

#### PROBLEM 1.7 ####
# open challenge set files
chal_set = convertFileToList2D("challenge_set.txt")
chal_set_labels = convertFileToList1D("challenge_set_labels.txt")

# run challenge set in runTest(), store results in new TestResults() object
chal_y, chal_s = runTest(chal_set, weight, 21, 20, 1)
writeListToFile("challenge_output.txt", chal_y, "%d")
chalRes = TestResults(chal_y, chal_set_labels)
chalRes.printMetrics()
chalRes.countNumClass()

# display the 2-9 digit classification data in a table
fig, axs = plt.subplots(1, 1)
#axs.axis("tight")
axs.axis("off")
table = axs.add_table(plt.table(chalRes.num_classes, rowLabels = ["    0    ", "    1    "], rowColours = ["lightgray", "lightgray"], 
                                colLabels = [str(i) for i in range(2, 10, 1)], colColours = ["lightgray" for i in range(2, 10, 1)], 
                                loc = "center", cellLoc = "center", colLoc = "center"))
table.scale(1, 2)

plt.title("Figure 1.7: Classifications of Digits 2 through 9", y = .65, weight = "bold", fontsize = 12)
plt.show()
