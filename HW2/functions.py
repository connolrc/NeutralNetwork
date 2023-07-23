from io import TextIOWrapper
import matplotlib.pyplot as plt
import numpy as np
import linecache as lc

# read lines from the given text file
# return list of 28x28 float matrices which are the images I GUESS 
def convertFileToImages(filename): 
    # open the file
    textfile = open(filename, "r")
    
    datalist = []   # initialize master list
    row = 1         # iterate through rows for readDataLine()
    
    for line in textfile: 
        # read the rowth line from the textfile (readDataLine())
        # convert it from 1x784 list to 28x28 matrix (convertListToImage())
        # append to datalist
        datalist += [convertListToImage(readDataLine(filename, row))]
        row += 1
    
    # datalist = np.transpose(datalist)
    return datalist
    

# read line of data (one image) from given row in given file
# return 1x784 list of floats
def readDataLine(filename, row): 
    # convert the .txt file line to a list
    dataline = lc.getline(filename, row).split()
    dataline = [float(x) for x in dataline]
    
    return dataline

# # read the first line from given image file 
# def readDataLine(filename): 
#     # open the file
#     if type(filename) == str: 
#         image_txt = open(filename, "r") 
#     # or bypass this step if it's already a file
#     elif type(filename) == TextIOWrapper: 
#         image_txt = filename
    
#     # convert the .txt file line to a list
#     dataline = image_txt.readline().split()
#     dataline = [float(x) for x in dataline]
    
#     return dataline


# convert a 1x784 list of floats into a 28x28 matrix of floats
# return 28x28 matrix of floats
def convertListToImage(dataline): 
    # convert the 1x784 list to a 28x28 matrix
    image_matrix = np.array(dataline)
    image_matrix = np.reshape(image_matrix, (28, 28))
    # transpose to orient image correctly upright
    image_matrix = np.transpose(image_matrix)
    
    return image_matrix

# generate a plot from the given 28x28 float matrix
def plotImage(image_array): 
    if len(image_array) == 28: 
        plt.imshow(image_array, cmap = "autumn", interpolation = "nearest")
    else: # if it is in 1x784 vector form...
        plt.imshow(convertListToImage(image_array), cmap = "autumn", interpolation = "nearest")
    #plt.show()


# create a list containing the contents of an image file.
# the same as convertFileToImage() except it keeps the 
# pixel values subarray 1x784 instead of converting it to 28x28. 
def convertFileToList2D(filename): 
    # open the file
    textfile = open(filename, "r")
    
    datalist = []   # initialize master list
    row = 1         # iterate through rows for readDataLine()
    
    for line in textfile: 
        # read the rowth line from the textfile (readDataLine())
        # append to datalist
        datalist += [readDataLine(filename, row)]
        row += 1
    
    return datalist


# create a list containing the contents of a label file
def convertFileToList1D(filename): 
    # open the file
    textfile = open(filename, "r")
    
    datalist = []
    row = 1
    for line in textfile: 
        datalist += [int(lc.getline(filename, row)[0])]
        row += 1
    
    return datalist
    
    
# writes the contents of a list into the given file
def writeListToFile(filename, lst, textformat = "%.6f"): 
    # open the file
    textfile = open(filename, "w")
    arr = np.array(lst)
    np.savetxt(textfile, lst, textformat)
    textfile.close()
    # for line in lst: 
    #     textfile.write(str(line))
    
    
# determines correctness of given test output, 
# according to given list of correct labels

class TestResults: 
    
    # constructor
    def __init__(self, test_output, answer_key): 
        self.test_output = test_output
        self.answer_key = answer_key
        
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
                self.F1_score == 0
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
            if self.answer_key[t] >= 2:     # unnecessary in this case, but on principle 
                self.num_classes[self.test_output[t]][self.answer_key[t] - 2] += 1
        
    def printMetrics(self): 
        for t in range(len(self.true_pos)): 
            self.printNthTestMetrics(t)
            
    def printNthTestMetrics(self, t): 
        print("--------- Test No. {:d} ---------" .format(t))
        print("theta = " + str(t))
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
        
    