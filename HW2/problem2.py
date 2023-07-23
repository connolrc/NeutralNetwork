import random
import matplotlib.pyplot as plt
from functions import *

# determines correctness of given test output, 
# according to given list of correct labels

class PercTestResults: 
    
    # constructor
    def __init__(self, test_output, answer_key): 
        self.test_output = test_output.copy()
        self.answer_key = answer_key.copy()
        self.numRuns = len(self.test_output) // len(self.answer_key)
        
        self.true_pos = [0] * self.numRuns       # 1s identified as 1
        self.true_neg = [0] * self.numRuns       # 0s identified as 0
        self.false_pos = [0] * self.numRuns      # 0s identified as 1
        self.false_neg = [0] * self.numRuns      # 1s identified as 0
        
        self.tallyResults()
        self.calcMetrics()
        
    # count up pos and neg (true/false)
    def tallyResults(self): 
        for i in range(len(self.test_output)): 
            if self.test_output[i] == 0: 
                # modulo % since the answer key is 200 lines but the test
                # output is far more, so the answer key's index will loop
                # back around to 0 right before it gets out of bounds
                if self.answer_key[i % len(self.answer_key)] == 0: 
                    # integer division // so each 200 test outputs will all
                    # go into the same index in the tally lists
                    self.true_neg[i // len(self.answer_key)] += 1
                else: 
                    self.false_neg[i // len(self.answer_key)] +=1 
            else: # test_output[i] == 1: 
                if self.answer_key[i % len(self.answer_key)] == 0: 
                    self.false_pos[i // len(self.answer_key)] += 1
                else: 
                    self.true_pos[i // len(self.answer_key)] += 1
        
    def calcMetrics(self): 
        self.recall = []
        self.precision = []
        self.F1_score = []
        self.specificity = []
        for n in range(len(self.test_output) // len(self.answer_key)): 
            # RECALL: selectivity, true positive rate. 
            # fraction of positives identified.
            # how much of what's true is in fact identified as true. 
            # true positives / all positives in data
            # Q11 / (Q11 + Q10)
            self.recall += [self.true_pos[n] / (self.true_pos[n] + self.false_neg[n])]
            
            # PRECISION: positive predictive value
            # fraction of identified positives that are correct. 
            # how much of what's identified as true is in fact true. 
            # true positives / all classified as positive
            # Q11 / (Q11 + Q01)
            if self.true_pos[n] == 0: 
                self.precision += [0]
            else: 
                self.precision += [self.true_pos[n] / (self.true_pos[n] + self.false_pos[n])]
            
            # F1 Score: combines precision and recall into one metric. 
            if self.precision[n] == 0 or self.recall[n] == 0: 
                self.F1_score += [0]
            else:     
                self.F1_score += [2 * (self.precision[n] * self.recall[n]) / (self.precision[n] + self.recall[n])]
    
            # SPECIFICITY: selectivity, true negative rate 
            # fraction of negatives identified
            # true negatives / (false positives + true negatives)
            # Q00 / (Q01 + Q00)
            if self.true_neg[n] == 0: 
                self.specificity += [0]
            else: 
                self.specificity += [self.true_neg[n] / (self.false_pos[n] + self.true_neg[n])]
    
    # FOR THE CHALLENGE SET: 
    # counts the amount of each digit 2-9 counted as 0 and as 1. 
    def countNumClass(self): 
        # declare 2x8 array in which to store counts of each 
        # number's classifications as 0 or 1. 
        # elements initialized to 0 for summation in 
        # subsequent for loop. 
        self.num_classes = [[0] * 8] + [[0] * 8]
        
        # step through output, check if each is a number 2-9. 
        # if so, get its specific value and use that as the col index 
        # with the test's output as the row index to increment the 
        # corresponding element in the 2x8 num_classes array. 
        for t in range(len(self.test_output)): 
            if len(self.answer_key) > 0: 
                if self.answer_key[t % len(self.answer_key)] >= 2:     # unnecessary in this case, but on principle 
                    self.num_classes[self.test_output[t]][self.answer_key[t % len(self.answer_key)] - 2] += 1
        
    def printMetrics(self): 
        for t in range(len(self.true_pos)): 
            self.printNthTestMetrics(t)
            
    def printNthTestMetrics(self, t): 
        print("--------- Test Metrics ---------")
        print("true pos = " + str(self.true_pos[t]))
        print("true neg = " + str(self.true_neg[t]))
        print("false pos = " + str(self.false_pos[t]))
        print("false neg = " + str(self.false_neg[t]))
        print("recall = " + str(self.recall[t]))
        print("precision = " + str(self.precision[t]))
        print("specificity = " + str(self.specificity[t]))
        print("F1 score = " + str(self.F1_score[t]))
        print("\n")
            
    def printBestTestMetrics(self): 
        self.printNthTestMetrics(self.F1_score.index(max(self.F1_score)))
        
    

# class modeling the perceptron, containing methods to train it and run tests, 
# as well as perform analysis of metrics and more. 
class Perceptron: 
    # constructor
    def __init__(self, wi0_lowerbound = 0.0, wi0_upperbound = 0.5, 
                 wi = [random.uniform(0.0, 0.5) for i in range(784)]): 
        self.wi0 = random.uniform(wi0_lowerbound, wi0_upperbound)
        self.x0 = 1
        
        # weights for in case you want to make a perceptron with a specific weight set
        self.wi = wi.copy()
        # save the initial weights for reference later
        self.wi0_init = self.wi0
        self.wi_init = self.wi.copy()
    
    # x : image data, input as 2D array containing vectors of pixel values between 0 and 1
    # k : each training step, which in this case is also each training image
    # j : each individual input, each pixel of the image i
    # wi : list of weights for each pixel
    # wi0 : bias weight, scalar
    # y : y(t), the neuron's output. returned as list. 
    # z : z(t), the teaching input / desired output. 2D in parallel form as x. the labels text file is used as this!
    # eta : learning rate, coefficient of weight adjustment
    # epochs : number of times to pass through the whole training set
    def runEpoch(self, x, z, eta = 0.01, epochs = 41, training = False):               
        m = len(x)     # number of images
        # list of net inputs for each image in the set, for every epoch
        self.y = []
        #s = [0.0] * m * epochs    
        # tracks index for s, since it's not 2D (because it's not cumulative across epochs)
        y_index = 0
        # error fraction
        self.errFrac = []
        # output list to be modified in-loop
        y = [0] * m
        
        # run through eps amount of epochs
        for eps in range(epochs): 
            # for each image in the training set...
            for k in range(m): 
                # create temp numpy matrix to make transpose for y calculation
                wi_temp_transpose = np.matrix(self.wi).transpose()
                xtemp = np.matrix(x[k])[0]
                
                # calculate actual output y
                y[k] = int(np.matmul(xtemp, wi_temp_transpose) + self.wi0 > 0)
                
                # if this is a training epoch ...
                if training == True: 
                    # ... then adjust weights for perceptron
                    # z[k % len(z)] : desired output
                    for j in range(len(x[k])): 
                        self.wi[j] = self.wi[j] + eta * (z[k % len(z)] - y[k]) * x[k][j]
                        #self.wi[j] = self.wi[j] + eta * (y[k] - z[k % len(z)]) * x[k][j]
                    #self.wi = wi
                    self.wi0 = self.wi0 + eta * (z[k % len(z)] - y[k])
                    #self.wi0 = self.wi0 + eta * (y[k] - z[k % len(z)])
                    #     self.wi[j] = self.wi[j] + eta * z[k % len(z)] * x[k][j]
                    # self.wi0 = self.wi0 + eta * z[k % len(z)]
                    
                    # if self.wi != self.wi_init:
                    #     print("\n\nHOLY FUCK")
                
                # update y index for next loop
                y_index += 1

            
            # initialize PercTestResults class object to record results
            # use to calculate error fraction
            epRes = PercTestResults(y, z)
            # this epoch's error fraction
            ep_errFrac = (epRes.false_neg[0] + epRes.false_pos[0]) / m      # m = number of images in set
            self.errFrac.append(ep_errFrac)   
            
            # # check if optimal weight has been reached, end training early if so
            # if ep_errFrac == 0.0: 
            #     break
            
        #self.y += y
        
        self.y = y.copy()
        return PercTestResults(y, z)



###############################
#### "main" section below ####
#############################

# get a 2D list of 1x784 lists of training set's images
tr_set = convertFileToList2D("training_set.txt")
# gets a list of the label for each sublist in tr_set
tr_set_labels = convertFileToList1D("training_set_labels.txt")

# w0 and wi are handled in Perceptron class
# declare Perceptron class instance
Ptron = Perceptron()

# RUN TRAINING on the perceptron using the training set and labels. 
# this method returns a PercTestResults() object
trRes = Ptron.runEpoch(tr_set, tr_set_labels, eta = .001, epochs = 100, training = True)

writeListToFile("training_output_Ptron.txt", trRes.test_output, "%d")
writeListToFile("training_weights_Ptron.txt", [Ptron.wi])
tr_err = Ptron.errFrac.copy()

print("---- original training ----")
trRes.printBestTestMetrics()

# open test set and labels
test_set = convertFileToList2D("test_set.txt")
test_set_labels = convertFileToList1D("test_set_labels.txt")

# RUN TEST 
testRes = Ptron.runEpoch(test_set, test_set_labels, epochs = 100)
writeListToFile("test_output_Ptron.txt", testRes.test_output, "%d")
test_err = Ptron.errFrac.copy()

print("---- original testing ----")
testRes.printBestTestMetrics()

#### PLOT ERROR FRACTIONS ####
plt.plot([i for i in range(len(tr_err))], tr_err, label = "Training Set", color = "xkcd:jungle green")
plt.plot([i for i in range(len(test_err))], test_err, label = "Testing Set", color = "xkcd:mango")
plt.legend()
plt.xlabel("Epoch Number", fontweight = "bold", fontsize = 11)
plt.ylabel("Error Fraction", fontweight = "bold", fontsize = 11)
plt.title("Figure 2.1: Error Fraction Comparison", fontweight = "bold", fontsize = 14)
plt.figure()

#### PLOT BAR GRAPH ####
# declare another Perceptron object to get data before training for
PtronNoTr = Perceptron()
beforeTr = PtronNoTr.runEpoch(tr_set, tr_set_labels)
print("---- beforeTr ----")
beforeTr.printBestTestMetrics()

afterTr = PtronNoTr.runEpoch(tr_set, tr_set_labels, training = True)
print("---- afterTr ----")
afterTr.printBestTestMetrics()

barWidth = 0.33     # for arrangement of paired bar graph
plt.bar([0, 1, 2], [beforeTr.precision[0], beforeTr.recall[0], beforeTr.F1_score[0]], 
        width = barWidth, label = "Pre-Training", color = "xkcd:ruby")
plt.bar([0 + barWidth, 1 + barWidth, 2 + barWidth], [afterTr.precision[0], afterTr.recall[0], afterTr.F1_score[0]], 
        width = barWidth, label = "Post-Training", color = "xkcd:sapphire")
plt.ylabel("Rate", fontweight = "bold", fontsize = 13)
plt.xticks([x + (barWidth / 2) for x in range(3)], ["Precision", "Recall", "F1 Score"], 
           fontweight = "bold", fontsize = 12)
plt.legend()

plt.title("Figure 2.2: Scores Before and After Training", fontweight = "bold", fontsize = 14)

# plt.figure()


#### PLOT CHALLENGE SET RESULTS TABLE ####

# open challenge set files
chal_set = convertFileToList2D("challenge_set.txt")
chal_set_labels = convertFileToList1D("challenge_set_labels.txt")

# run challenge set on Ptron, the first Perceptron object. 
# it has been trained previously, so its weights are tuned. 
chalRes = Ptron.runEpoch(chal_set, chal_set_labels)
writeListToFile("challenge_output", chalRes.test_output)
chalRes.countNumClass()

# display the 2-9 digit classification data in a table
fig, axs = plt.subplots(1, 1)
#axs.axis("tight")
axs.axis("off")
table = axs.add_table(plt.table(chalRes.num_classes, 
                                rowLabels = ["    0    ", "    1    "], rowColours = ["lightgray", "lightgray"], 
                                colLabels = [str(i) for i in range(2, 10, 1)], colColours = ["lightgray" for i in range(2, 10, 1)], 
                                loc = "center", cellLoc = "center", colLoc = "center"))
table.scale(1, 2)

plt.title("Figure 2.3: \n   Perceptron's Classifications of Digits 2 through 9", y = .65, weight = "bold", fontsize = 12)
plt.figure()

#### INITIAL AND FINAL WEIGHTS HEATMAPS ####

# print(Ptron.wi_init == Ptron.wi)
plt.subplot(1, 2, 1)
plotImage(Ptron.wi_init)
#plt.viridis()
plt.title("Figure 2.4.1:\nHeatmap of Weight\nBefore Training", loc = "center")
plt.subplot(1, 2, 2)
plotImage(Ptron.wi)
#plt.bone()
plt.title("Figure 2.4.2:\nHeatmap of Weight\nAfter Training", loc = "center")
#plt.figure()


plt.show()