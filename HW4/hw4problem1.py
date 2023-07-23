# HW4 Problem 1
# Ryan Connolly

import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from functions import *

DEBUG_FLAG = True

# ******************************************
# *** DICTIONARY FOR TERMS AND VARIABLES *** 
# ******************************************
# x : input (X is total amount)
# y : output (Y is total amount, same as X)
# h : hidden layer 
# w : weights (-0.061 < w < 0.061) [784, 100, 10]
#     each w array has num rows equal to num of neurons in next layer, 
#     num cols equal to num of neurons in current layer 
# w0 : bias weights, 2D array in program, num rows equal to total layers, 
#      num cols equal to total num neurons in next layer
# eta : learning factor
# alpha : momentum, velocity
# q : index for image in dataset
# i (l) : number of output neurons (output space dimension, L is total, i or l is iterator)
# j (m) : number of hidden neurons (hidden space dimension, M is total, j or m is iterator)
# k (n) : number of input neurons (input space dimension, N is total, k or n is iterator)
# s : net input / pre-activation values, sjk = wj1 * x1 + wj2 * x2 + ... + wjk * xk + w0j
# f(s) : activation function, f(s) = a * tanh(b * s)
# J : summation of (real y for kth output neuron for ith input - estimated y for the same)^2


# f : lambda function for convenience within larger function
#     according to Le Cun's suggestions, a = 1.7159 and b = 2/3
#     define larger function to perform activation on every neuron
# fprime : the derivative of f
# f = lambda s : 1.7159 * math.tanh((2/3) * s)
# fprime = lambda s : 1.1439 / math.pow(math.cosh((2/3) * s), 2)
f = lambda s : math.tanh(s)
fprime = lambda s : 1 - math.pow(f(s), 2)

# used to normalize a vector of values between -1 and 1
normalize = lambda lst : [((el - min(lst)) / (max(lst) - min(lst))) * 2 - 1 for el in lst]

# CLASS DEFINITION FOR A BACKPROPAGATING NEURAL NETWORK WITH 1 HIDDEN LAYER
class NeuralNetwork: 
    # init (constructor) function
    def __init__(self, total_inputs = 784, total_hidden = 196, total_outputs = 10):
                 # weights_jk = None, weights_ij = None, biases_jk = None, biases_ik = None):
        self.N = total_inputs 
        self.M = total_hidden
        self.L = total_outputs
        self.layer_dims = [self.N, self.M, self.L]

        self.weightsInit()
        self.biasesInit()
        
    
    # method to preload weights into the network from a text file, given file name,
    # rather than randomly generating and training weights
    def preloadWeights(self, file_wjk = None, file_wij = None, file_w0jk = None, file_w0ij = None): 
        if file_wjk != None: 
            self.w[0] = convertFileToList2D(file_wjk)
        if file_wij != None: 
            self.w[1] = convertFileToList2D(file_wij)
        if file_w0jk != None: 
            self.w0[0] = convertFileToList1D(file_w0jk, "float")
        if file_w0ij != None: 
            self.w0[1] = convertFileToList1D(file_w0ij, "float")
    

    # intialize the weights according to how many hidden neurons desired
    # using Xavier initialization, normal distribution between -a and a
    # where a ~ sqrt(6 / Ns + Nt), Ns = total neurons in source layer and
    # Nt = total neurons in target layer
    # OKAY ACTUALLY using a normal distribution -a to a, a = sqrt(3/N)
    # ACTUALLYYYY it's the Gaussian one right now
    # INPUT PARAMETERS (before forming class)
    # layer_shape : list of neurons in each layer
    #               [784, 100, 10]
    # OUTPUT RETURNS (before forming class)
    # w[][][] : 3D array of the weights, accessed in manner of w[layer_num][j][k]
    def weightsInit(self): 
        num_layers = 2
        
        # intialize empty weights array, to be 3D
        self.w = [0.0] * num_layers
        
        # initialize wdelta nested array to hold the 
        # previous weight gradient for momentum calculation
        self.wdelta = [0.0] * num_layers
        
        a = []
        # calculate a for the input-to-hidden weights
        a += [math.sqrt(2 / self.N)]
        # calculate a for the hidden-to-output weights
        a += [math.sqrt(2 / (self.M))]
        
        for idx in range(num_layers): 
            # self.w[idx] = np.random.uniform(-a[idx], a[idx], (self.layer_dims[idx + 1], self.layer_dims[idx]))
            # self.w0[idx] = np.random.uniform(-a[idx], a[idx], self.layer_dims[idx + 1])
            self.w[idx] = np.random.normal(0, a[idx], (self.layer_dims[idx + 1], self.layer_dims[idx]))
            
            # add arrays of 0s into the gradient weight tracker arrays
            self.wdelta[idx] = np.zeros_like(self.w[idx])
            
        # save initial weights for viewing later
        self.w_init = self.w.copy()
        
    
    # function to initialize bias weights
    # RETURN OUTPUT
    # w0[][] : 2D array of bias weights, accessed in manner of bias[layer_num][j] 
    def biasesInit(self): 
        num_layers = 2
        self.w0 = [0.0] * num_layers
        self.w0delta = [0.0] * num_layers
        
        a = []
        # calculate a for the input-to-hidden weights
        a += [math.sqrt(2 / self.N)]
        # calculate a for the hidden-to-output weights
        a += [math.sqrt(2 / (self.M))]
        
        for idx in range(num_layers): 
            self.w0[idx] = np.random.normal(0, a[idx], self.layer_dims[idx + 1])
            # add arrays of 0s into the gradient weight tracker arrays
            self.w0delta[idx] = np.zeros_like(self.w0[idx])
            
        # save initial weights for viewing later
        self.w0_init = self.w0.copy()

    # the preactivation function for the calculating the hidden layer values
    # INPUT PARAMETERS
    # xq : input from input layer x, q signifying specific image in dataset
    # OUTPUT RETURN 
    # sqj : the preactivation values for hidden neurons for current image q
    def preActivFuncHidden(self, xq): 
        # initialize preactivation values for hidden layer, sqj, as list
        sqj = [0.0] * self.M
                
        # calculate summation of the products of input x[k] and weight w[0][j][k], 
        # which corresponding to input x[k] and this hidden neuron h[j]
        for j in range(self.M): 
            for k in range(self.N): 
                sqj[j] += xq[k] * self.w[0][j][k]
                
                if DEBUG_FLAG == True and abs(sqj[j]) >= 1.0: 
                    True
            
            # don't forget to add the bias weight!
            sqj[j] += self.w0[0][j]
            
            if DEBUG_FLAG == True and abs(sqj[j]) >= 1.0: 
                True
            
        return sqj
        
    # activation function for calculating the hidden layer values 
    # f(s) : activation function
    # INPUT PARAMETERS 
    # q : current image in the dataset
    # OUTPUT RETURN
    # hq : the hidden neurons' values for current image q
    def activFuncHidden(self, q):
        # initialize hidden layer neurons as list
        hq = []
        
        for j in range(self.M): 
            hq += [f(self.sj[q][j])]
            
            if DEBUG_FLAG == True and abs(hq[j]) >= 1.0: 
                True
            
        return hq
    
    # the preactivation function for the calculating the hidden layer values
    # INPUT PARAMETERS
    # q : current image in dataset
    # OUTPUT RETURN 
    # sqi : the preactivation values for output neurons for current image q
    def preActivFuncOutput(self, q): 
        # initialize preactivation values for output, sqi, as list
        sqi = [0.0] * self.L 
                
        # calculate summation of the products of hidden neuron h[j] and weight w[1][i][j], 
        # which corresponding to hidden neuron h[j] and this output neuron [i]
        for i in range(self.L): 
            for j in range(self.M): 
                sqi[i] += self.h[q][j] * self.w[1][i][j]
                
                if DEBUG_FLAG == True and abs(sqi[i]) >= 1.0: 
                    True
            
            # don't forget to add the bias weight!
            sqi[i] += self.w0[1][i]
            
            if DEBUG_FLAG == True and abs(sqi[i]) >= 1.0: 
                True
            
        return sqi
        
    # activation function for calculating the hidden layer values 
    # f(s) : activation function
    # INPUT PARAMETERS 
    # q : current image in the dataset
    def activFuncOutput(self, q):
        # initialize hidden layer neurons as list
        yq = []
        
        for i in range(self.L): 
            yq += [f(self.si[q][i])]
            
            if DEBUG_FLAG == True and abs(yq[i]) >= 1.0: 
                True
            
        return yq
                
    
    # adjust weights using backpropagation, momentum 
    # INPUT PARAMETERS
    # q : current image in the dataset 
    def adjustWeights(self, q, backprop = True): 
        # ambda function to determine if the output neuron self.y[q][iy] has a value matching desired output.
        # used to determine those neurons that need weight adjustment (False output means adjustment needed).
        # the function returns False in these cases: 
        # a) output neuron i is the correct answer but its value is < op_H
        # b) output neuron i is a wrong answer but its value is > op_L
        isMatch = lambda iy : (True if (self.ytrue[q] == iy and self.y[q][iy] >= self.op_H) 
                                    or self.op_L <= self.y[q][iy]
                                else False)
        
        # the errors to propagate backwards
        delta_qi = [0.0] * self.L
        # the output y with the binary threshold parameters op_H and op_L enforced
        # y_bin = self.enforceBinary(self.y[q])
        
        # calculate weights wij (for hidden-to-output)
        # for every output neuron...
        for i in range(self.L): 
            if isMatch(i) == False: 
                # calculate delta_q[i] to use for every input neuron j's weight adjustment
                delta_qi[i] = (int(self.ytrue[q] == i) - self.y[q][i]) * fprime(self.si[q][i])
                # for every hidden neuron...
                for j in range(self.M): 
                    # momentum = alpha * the current value stored in wdelta[1][i][j], 
                    # as we are about to calculate the next at t, so this one is t - 1
                    momentum_i = self.alpha * self.wdelta[1][i][j] 
                    # calculate wdelta(t)
                    self.wdelta[1][i][j] = self.eta * delta_qi[i] * self.h[q][j]
                    # update the weight with wdelta + momentum 
                    self.w[1][i][j] += self.wdelta[1][i][j] + momentum_i

                # update bias weight w0
                momentum0_i = self.alpha * self.w0delta[1][i]
                self.w0delta[1][i] = self.eta * delta_qi[i] 
                self.w0[1][i] += self.w0delta[1][i] + momentum0_i

        if backprop == True: 
            # errors to propagate for input-to-hidden weights 
            delta_qj = [0.0] * self.M 

            # calculate weights wjk (for input-to-hidden)
            # for every hidden neuron...
            for j in range(self.M): 
                delta_qj[j] = fprime(self.sj[q][j]) * sum([self.w[1][i][j] * delta_qi[i] for i in range(self.L)])
                # for every input neuron...
                for k in range(self.N): 
                    # momentum = alpha * the current value stored in wdelta[1][i][j], 
                    # as we are about to calculate the next at t, so this one is t - 1
                    momentum_j = self.alpha * self.wdelta[0][j][k] 
                    # calculate wdelta(t)
                    self.wdelta[0][j][k] = self.eta * delta_qj[j] * self.x[q][k]
                    # update the weight with wdelta + momentum 
                    self.w[0][j][k] += self.wdelta[0][j][k] + momentum_j

                # update bias weight w0
                momentum0_j = self.alpha * self.w0delta[0][j]
                self.w0delta[0][j] = self.eta * delta_qj[j] 
                self.w0[0][j] += self.w0delta[0][j] + momentum0_j
            
    # define function for training the neural network
    # INPUT PARAMETERS
    # x : the input dataset for training the neural network
    # ytrue : the answer key for x
    # epochs : the number of times to run through the dataset to train
    # eta : learning rate, coefficient of weight adjustment 
    # alpha : momentum, velocity
    # op_H : operating parameter such that yi >= H is considered
    #        a match for yi = +1
    # op_L : operating parameter such that yi <= L is considered
    #        a match for yi = 0 or -1
    def run(self, x, ytrue, training = False, epochs = 125, eta = .001, alpha = .3, 
            op_H = .75, op_L = .25, stoch = True, backprop = True): 
        self.x = x
        self.ytrue = ytrue
        self.eta = eta
        self.alpha = alpha
        self.op_H = op_H
        self.op_L = op_L
        # number of images in dataset
        self.Q = len(x)
        # binary nested array, each row is for an epoch. 
        # in each row is a binary digit for each stochastically chosen image in the set, 
        # a 1 if the image was classified correctly, and 0 if not
        self.errs = []
        # array of length of epochs parameter, contains the error fraction for each epoch
        self.err_frac = []
        # nested array of J2 at each point in training, sublist for each epoch
        # each sublist contains each image's Jq in epoch
        self.Jq = []
        
        # epoch loop: run through the dataset as many times as given in epochs parameter
        for epoch in range(epochs): 
            if DEBUG_FLAG == True: 
                print("\n")
                print("".center(110, '*'))
                print("*   EPOCH  {:03d}   *".center(110, '*').format(epoch + 1))
                print("".center(110, '*'))
                            
            # initialize hidden and output layer neurons, 
            # and preactivation values s for hidden and output calculation, 
            # nested array of each image's results
            self.h = [0.0] * self.Q
            self.y = [0.0] * self.Q
            self.si = [0.0] * self.Q
            self.sj = [0.0] * self.Q
            
            # add new nested list for this new epoch
            self.errs += [[]]
            self.Jq += [[]]
            
            # for each image in the input dataset x...
            for q in range(self.Q): 
                # use stochastic gradient descent so only a random ~60% of images 
                # will be shown to the network
                random.seed()
                sto_ran = random.random()
                # if DEBUG_FLAG == True and self.ytrue[q] == 8: 
                #     sto_ran = 0.0
                # only randomly decide if this image should be looked at if the 
                # stoch parameter is set to True, otherwise look at all images
                if sto_ran <= .3 or stoch == False: 
                    # generate hidden layer neuron data from input layer data
                    self.sj[q] = self.preActivFuncHidden(self.x[q])
                    self.h[q] = self.activFuncHidden(q)
                    
                    # generate output layer neuron data from hidden layer data
                    self.si[q] = self.preActivFuncOutput(q)
                    self.y[q] = self.activFuncOutput(q)
                    
                    if training == True: 
                        # adjust the weights based on this image
                        self.adjustWeights(q, backprop)
                    
                    # calculate error fraction
                    # errs_q = sum([1 for ii in range(len(self.y[q])) if self.isCorrect(q, ii) == False])
                    # self.errs += [errs_q]                    
                    # self.err_frac += [errs_q / 10]
                    # self.err_frac += [errs_q / ((q + 1) * self.L)]
                    
                    # append self.errs[] with a 1 if qth image was classified correctly, 0 if not
                    self.errs[epoch] += [self.isCorrect(q)]
                    
                    # calculate J2
                    self.Jq[epoch] += [sum(math.pow(int(self.ytrue[q] == ii) - self.y[q][ii], 2) 
                                     for ii in range(10)) * .5]
                                        
                # if this image is not stochastically chosen, we still need to add an element
                # to the Jq and err_frac arrays to reflect the full epoch
                # else: 
                #     if q > 0: # if it's been one image already so there's at least one element,
                #         # add a duplicate of the last element to each array 
                #         # self.err_frac += [self.err_frac[len(self.err_frac) - 1]]
                #         self.Jq += [self.Jq[len(self.Jq) - 1]]
                #     else: # otherwise add 0.0
                #         # self.err_frac += [0.0]
                #         self.Jq += [0.0]
                    
                if DEBUG_FLAG == True and sto_ran <= .0005:
                    print()
                    print("   Image #{:04d}   ".center(108, '-').format(q + 1))
                    # if type(self.y[q]) == list: 
                    #     print("True y: {}                Output y: {}".center(110).format(self.ytrue[q], self.getClass(q)))
                    #     print()
                    #     print("0: {: .6f}         1: {: .6f}         2: {: .6f}         3: {: .6f}         4: {: .6f}           ".center(110).format(self.y[q][0], self.y[q][1], self.y[q][2], self.y[q][3], self.y[q][4]))
                    #     print("5: {: .6f}         6: {: .6f}         7: {: .6f}         8: {: .6f}         9: {: .6f}           ".center(110).format(self.y[q][5], self.y[q][6], self.y[q][7], self.y[q][8], self.y[q][9]))
                    # else: 
                    #     print("True y: {}                           ".center(108).format(self.ytrue[q]))
                    #     print("\n\n")
                    
                # if DEBUG_FLAG == True and type(self.si[q]) == list: 
                #     if any([s >= 1 for s in self.si[q]]): 
                #         True
                
            # append self.err_frac with the error fraction for this epoch 
            self.err_frac += [1 - np.average(self.errs[epoch])]
            
            if DEBUG_FLAG == True and epoch % 1 == 0: # and epoch > 0: 
                print("\n")
                print("".center(110, '*'))
                print("   EPOCH {:03d} STATS   ".center(110, '*').format(epoch + 1))
                # print("  Jq = {:.6f}  ".center(110, '*').format(np.average(self.Jq[len(self.Jq - 1)])))
                print("  err_frac = {:.4f}  ".center(106, '*').format(self.err_frac[int(len(self.err_frac)) - 1]))
                print("".center(110, '*'))
                
                self.plotStats()
                plt.suptitle("Epoch {:03d}" .format(epoch + 1), fontweight="bold")
                self.plotHeatmapsWIJ()
                plt.suptitle("Epoch {:03d}: Weight Heatmaps for Digits" .format(epoch + 1), fontweight="bold")
                self.plotHeatmapsWJK()
                self.showConfusionMatrix()
                # plt.show()
                # if this is not the last epoch...
                if epoch < epochs - 1:
                    plt.pause(10)
                    plt.close("all")
                else: 
                    plt.show()
            
    # define loss function, Mean-Squared Error (J2)
    def MSE(self): 
        # an array of the loss for each epoch
        self.Jq = []
        
        # for every image in the dataset...
        for q in range(self.Q): 
            # the output y with the binary threshold parameters op_H and op_L enforced
            # y_bin = self.enforceBinary(self.y[q])
                
            self.Jq += [0.0]
            # for every output neuron...
            for i in range(self.L): 
        
                # int(self.ytrue[q] == i) so that if the current i is reflective
                # of the correct answer, then the operation will return 1. 
                # otherwise, it will return 0. 
                self.Jq[q] += math.pow(int(self.ytrue[q] == i) - self.y[q][i], 2) 
                
            self.Jq[q] = self.Jq[q] * .5
            
        # overall J across all epochs
        self.J = sum(self.Jq)
        
    # method to return the network's classification of an output set/list y[q]
    # by finding the index of its max element 
    # INPUT PARAMETERS
    # q : the datapoint/image in the set
    # OUTPUT RETURNS
    # y_class : the class, digits 0-9, estimated by the network
    def getClass(self, q): 
        if type(self.y[q]) == list: 
            return self.y[q].index(max(self.y[q]))
        else: # when the qth image was not randomly stochastically chosen, 
              # so the element at y[q] is just 0.0
            return None
    
    # checks if a output y[q][i], given by parameters q and i, is correct
    # correct meaning that it was classified correctly, not necessarily 
    # that it is what digit the datapoint image depicts. 
    def isCorrect(self, q): 
        # if y correctly classified the qth image...
        return self.ytrue[q] == self.getClass(q)
    
    # error fraction calculator
    def errFrac(self, q = None):
        errs = 0
        if q == None: 
            q = self.Q
        total = q * self.L
        
        for qq in range(q): 
            errs += sum([1 for ii in range(self.y[qq]) if self.isCorrect(qq, ii) == False])
            
        return errs / total
        
        
    # calculates error fraction for specific number/digit given in parameter num
    def errFracForNum(self, num): 
        # measurement of how many times an image of num was classified correctly
        correct = 0
        # total amount of images of num that were classified, correct or incorrect
        # (this is not a simple 400, as with stochastic randomness some of the images
        # may not have been considered and classified)
        total = 0
        
        # loop through each image in dataset
        for q in range(self.Q): 
            # check to make sure the qth image wasn't skipped over for stochastic gradient
            # this shouldn't happen for the full, final network run of the data, but for 
            # my testing purposes at least it's necessary
            if type(self.y[q]) == list: 
                # output data that has been exported to a text file and then re-imported
                # will have skipped images as ten-element lists of 0.0 rather than a single, 
                # non-list element 0.0, so this second if checks for that. 
                if self.y[q] != [0.0] * 10: 
                    # check if image y[q] depicts the specified number num
                    if self.ytrue[q] == num: 
                        correct += self.isCorrect(q)
                        total += 1
                    
        return 1 - (correct / total)

    # enforce the binary threshold parameters op_H and op_L on given list lst, 
    # so that all elements of lst >= op_H are set to 1 and all elements of 
    # lst <= op_L are set to 0. the resultant list is returned as a new list. 
    # INPUT PARAMETERS 
    # lst : the given list to enforce the binary thresholds on 
    # OUTPUT RETURN
    # lst_bin : new list for which the binary threshold parameters are enforced
    def enforceBinary(self, lst): 
        lst_bin = []
        
        for el in lst: 
            if el >= self.op_H: 
                lst_bin += [1]
            elif el <= self.op_L: 
                lst_bin += [0]
            else: 
                lst_bin += [el]
                
        return lst_bin
    
    def plotHeatmapsWJK(self): 
        feats = [int(el) for el in np.random.choice(self.M, 20, replace=False)]
        plt.figure()
        for ii in range(20): 
                    # if the header neuron is an untouched one or if the index is a duplicate in the list,
                    # change it
                    # while len(self.h[int(feats[ii])]) <= 1:
            while type(self.h[feats[ii]]) != list and sum([int(np.isin(feats, [feats[ii]])[idx]) for idx in range(len(feats))]) != 1: 
                feats[ii] = np.random.choice(self.M)
            plt.subplot(4, 5, ii + 1)
            plotImage(self.w[0][feats[ii]], cmap="bone")    # plotImage() function from my functions.py file
            plt.title("Hidden Neuron " + str(feats[ii] + 1), fontweight="bold", fontsize=9)
            plt.xticks(range(0, 28, 7), fontsize=6)
            plt.yticks(range(0, 28, 7), fontsize=6)
        plt.suptitle("Sample Input-to-Hidden Weights", fontweight="bold", fontsize=11)
    
    # plot hidden-to-output layer weights for each number as heat map images
    def plotHeatmapsWIJ(self): 
        plt.figure()
        for ii in range(self.L):
            plt.subplot(2, 5, ii + 1)
            # plotImage() function from my functions.py file
            plotImage(self.w[1][ii], cmap="winter")
            plt.title("Number " + str(ii))

        plt.suptitle("Weight Heatmaps for Digits", fontweight="bold")
        
    # for hel in range(196): 
    #     plt.subplot(14, 14, hel + 1)
    #     plt.imshow(np.reshape(self.w[0][hel], (28, 28)))
    #     # plt.title("\n\n\n" + str(hel))
    #     plt.xticks([])
    #     plt.yticks([])
        
    # plot input-to-hidden layer weights for each number as heat map images
    # def plotHeatmapsWIJ(self): 
    #     plt.figure()
    #     for ii in range(self.L):
    #         plt.subplot(5, 2, ii + 1)
    #         # plotImage() function from my functions.py file
    #         plotImage(self.w[1][ii], cmap="bone")
        
    #     plt.suptitle("Weight Heatmaps for Digits", loc="center", fontweight="bold")
    
    # plot the J2 loss and error fraction 
    def plotStats(self): 
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot([x for x in range(1, len(self.Jq) + 1)], [np.average(sublist) for sublist in self.Jq], 
                 color="xkcd:shamrock green")
        plt.xlabel("Epochs", fontweight="bold", fontsize=10)
        # total images shown (duplicates included) / amount of images (duplicates not included)
        plt.xticks([i for i in range(0, 10, len(self.err_frac) + 1)])
        plt.title("J2 Loss Function", fontweight="bold", fontsize=11)
        plt.subplot(2, 1, 2)
        plt.plot([x for x in range(1, len(self.err_frac) + 1)], self.err_frac, color="xkcd:fire engine red")
        plt.xlabel("Epochs", fontweight="bold", fontsize=10)
        plt.ylabel("Error Fraction", fontweight="bold", fontsize=10)
        # total images shown (duplicates included) / amount of images (duplicates not included)
        plt.xticks([i for i in range(0, 10, len(self.err_frac) + 1)])
        plt.title("Error Fraction", fontweight="bold", fontsize=11)
        
        
    # construct confusion matrix
    def makeConfusionMatrix(self): 
        self.conmat = np.zeros((10, 10), dtype=int)
        # row : the actual, real digit
        # column : the digit the network classified this input as
        
        for q in range(self.Q): 
            # check to make sure the qth image wasn't skipped over for stochastic gradient
            # this shouldn't happen for the full, final network run of the data, but for 
            # my testing purposes at least it's necessary
            if type(self.y[q]) == list: 
                # output data that has been exported to a text file and then re-imported
                # will have skipped images as ten-element lists of 0.0 rather than a single, 
                # non-list element 0.0, so this second if checks for that. 
                if self.y[q] != [0.0] * 10: 
                    self.conmat[self.ytrue[q]][self.getClass(q)] += 1
                
    # display the confusion matrix as a table using matplotlib
    def showConfusionMatrix(self): 
        self.makeConfusionMatrix()
        
        fix, axs = plt.subplots(1, 1)
        axs.axis("off")
        table = axs.add_table(plt.table(self.conmat, rowLabels=["   " + str(i) + "   " for i in range(10)], 
                                        colLabels=[i for i in range(10)], loc="center", cellLoc="center", 
                                        colLoc="center", rowColours=["lightgray" for i in range(10)], 
                                        colColours=["lightgray" for i in range(10)]))
        table.scale(1, 2)
        plt.title("Confusion Matrix", fontweight="bold")
            
        

if __name__=="__main__":
    # get a list of the training set and its labels from .txt files
    trainset = convertFileToList2D("HW3/trainset.txt")
    trainset_labels = convertFileToList1D("HW3/trainset_labels.txt")

    # # initialize NeuralNetwork class
    Cortana = NeuralNetwork()
    Cortana.preloadWeights(file_wjk="HW4/ref_data/trainset2_weights_jk.txt", file_w0jk="HW4/ref_data/trainset2_bias_jk.txt")
    Cortana.run(trainset, trainset_labels, training=True, stoch=True, alpha=.3, eta=.001, backprop=False, epochs=41)

    y_smooth = []
    for y in Cortana.y: 
        if type(y) == list: 
            y_smooth += [y]
        else: 
            y_smooth += [[0.0] * 10]

    writeListToFile("HW4/case1_trainset_output.txt", y_smooth)
    writeListToFile("HW4/case1_trainset_weights_ij.txt", Cortana.w[1])
    writeListToFile("HW4/case1_trainset_bias_jk.txt", Cortana.w0[0])
    writeListToFile("HW4/case1_trainset_weights_jk.txt", Cortana.w[0])
    writeListToFile("HW4/case1_trainset_bias_ij.txt", Cortana.w0[1])
    writeListToFile("HW4/case1_trainset_err_frac.txt", Cortana.err_frac)

    # yyy = []
    # for iii in range(len(Cortana.y)): 
    #     if type(Cortana.y[iii]) == list:
    #         yyy += [Cortana.y[iii]]
    #     else: 
    #         yyy += [[0.0] * 4000]

    testset = convertFileToList2D("HW3/testset.txt")
    testset_labels = convertFileToList1D("HW3/testset_labels.txt")
    Cortana.preloadWeights(file_wij="HW4/case1_trainset_weights_ij.txt", file_w0ij="HW4/case1_trainset_bias_ij.txt")

    # # for i in range(2): 
    # #     for idx in range(len(Cortana.w[i])): 
    # #         Cortana.w[i][idx] = normalize(Cortana.w[i][idx])
    # #     Cortana.w0[i] = normalize(Cortana.w0[i])

    Cortana.run(testset, testset_labels, training=False, stoch=False, alpha=.3, eta=.001, backprop=False, epochs=1)
    # Cortana.showConfusionMatrix()

    y_smooth = []
    for y in Cortana.y: 
        if type(y) == list: 
            y_smooth += [y]
        else: 
            y_smooth += [[0.0] * 10]

    writeListToFile("HW4/case1_testset_output.txt", y_smooth)
    writeListToFile("HW4/case1_testset_weights_jk.txt", Cortana.w[0])
    writeListToFile("HW4/case1_testset_weights_ij.txt", Cortana.w[1])
    writeListToFile("HW4/case1_testset_bias_jk.txt", Cortana.w0[0])
    writeListToFile("HW4/case1_testset_bias_ij.txt", Cortana.w0[1])
    writeListToFile("HW4/case1_testset_err_frac.txt", Cortana.err_frac)

    Gaia = NeuralNetwork()
    Gaia.preloadWeights(file_wjk="HW4/ref_data/trainset2_weights_jk.txt", file_w0jk="HW4/ref_data/trainset2_bias_jk.txt")
    Gaia.run(trainset, trainset_labels, training=True, stoch=True, alpha=.3, eta=.001, backprop=True, epochs=41)

    y_smooth = []
    for y in Gaia.y: 
        if type(y) == list: 
            y_smooth += [y]
        else: 
            y_smooth += [[0.0] * 10]

    writeListToFile("HW4/case2_trainset_output.txt", y_smooth)
    writeListToFile("HW4/case2_trainset_weights_ij.txt", Gaia.w[1])
    writeListToFile("HW4/case2_trainset_bias_jk.txt", Gaia.w0[0])
    writeListToFile("HW4/case2_trainset_weights_jk.txt", Gaia.w[0])
    writeListToFile("HW4/case2_trainset_bias_ij.txt", Gaia.w0[1])
    writeListToFile("HW4/case2_trainset_err_frac.txt", Gaia.err_frac)

    # yyy = []
    # for iii in range(len(Gaia.y)): 
    #     if type(Gaia.y[iii]) == list:
    #         yyy += [Gaia.y[iii]]
    #     else: 
    #         yyy += [[0.0] * 4000]

    testset = convertFileToList2D("HW3/testset.txt")
    testset_labels = convertFileToList1D("HW3/testset_labels.txt")
    Gaia.preloadWeights(file_wij="HW4/case2_trainset_weights_ij.txt", file_w0ij="HW4/case2_trainset_bias_ij.txt")

    # # for i in range(2): 
    # #     for idx in range(len(Gaia.w[i])): 
    # #         Gaia.w[i][idx] = normalize(Gaia.w[i][idx])
    # #     Gaia.w0[i] = normalize(Gaia.w0[i])

    Gaia.run(testset, testset_labels, training=False, stoch=False, alpha=.3, eta=.001, backprop=False, epochs=1)
    # Gaia.showConfusionMatrix()

    y_smooth = []
    for y in Gaia.y: 
        if type(y) == list: 
            y_smooth += [y]
        else: 
            y_smooth += [[0.0] * 10]

    writeListToFile("HW4/case2_testset_output.txt", y_smooth)
    writeListToFile("HW4/case2_testset_weights_jk.txt", Gaia.w[0])
    writeListToFile("HW4/case2_testset_weights_ij.txt", Gaia.w[1])
    writeListToFile("HW4/case2_testset_bias_jk.txt", Gaia.w0[0])
    writeListToFile("HW4/case2_testset_bias_ij.txt", Gaia.w0[1])
    writeListToFile("HW4/case2_testset_err_frac.txt", Gaia.err_frac)






    # # plot training set and test set error fraction over each other
    # # get training set's err_frac from saved output .txt
    # trainset_output = convertFileToList2D("HW4/case2_trainset_output.txt")
    # trainset_err_frac = []
    # trainset_errs = []
    # for epoch in range(41): 
    #     trainset_errs += [[]]
    #     for q in range(len(trainset)):
    #         trainset_errs[epoch] += [trainset_labels[q] == trainset_output[q].index(max(trainset_output[q]))]
            
    #     trainset_err_frac += [1 - np.average(trainset_errs[epoch])]

    # # do the same for the test set
    # # however, no epochs as it is just one epoch because there is no need training. 
    # # also, multiply the array by 41 so it extends it for the graph, as it only has one epoch. 
    # testset_output = convertFileToList2D("HW4/case2_testset_output.txt")
    # testset_err_frac = []
    # testset_errs = []
    # for q in range(len(testset)):
    #     testset_errs += [testset_labels[q] == testset_output[q].index(max(testset_output[q]))]

    # testset_err_frac = [1 - np.average(testset_errs)] * 41

    # plt.figure()
    # plt.subplot(2, 1, 2)
    # # plt.plot([x for x in range(1, len(trainset_err_frac) + 1)], trainset_err_frac, 
    #         #  label="Training Set", color="xkcd:fire engine red")
    # plt.plot([x for x in range(1, len(testset_err_frac) + 1)], testset_err_frac, 
    #          label="Test Set", color="xkcd:shamrock green")
    # # plt.legend()
    # plt.xlabel("Epochs", fontweight="bold", fontsize=10)
    # plt.ylabel("Error Fraction", fontweight="bold", fontsize=10)
    # plt.xticks([i for i in range(0, len(trainset_err_frac) + 1, 10)])
    # plt.title("Figure 1.4: Test Set Error Fraction", fontweight="bold", fontsize=11)
    # plt.ylim(top=.85, bottom=0.0)
    # plt.yticks([0.2, 0.4, 0.6, 0.8])
    # plt.show(block=True)