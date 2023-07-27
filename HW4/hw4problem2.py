# HW4 Problem 2
# Ryan Connolly

import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.axes 
from functions import *
import pickle


DEBUG_FLAG = True

"""
******************************************
*** DICTIONARY FOR TERMS AND VARIABLES *** 
******************************************
x : input (X is total amount)
y : output (Y is total amount, same as X)
h : hidden layer 
w : weights (-0.061 < w < 0.061) [784, 100, 10]
    each w array has num rows equal to num of neurons in next layer, 
    num cols equal to num of neurons in current layer 
w0 : bias weights, 2D array in program, num rows equal to total layers, 
     num cols equal to total num neurons in next layer
eta : learning factor
alpha : momentum, velocity
q : index for image in dataset
i (l) : number of output neurons (output space dimension, L is total, i or l is iterator)
j (m) : number of hidden neurons (hidden space dimension, M is total, j or m is iterator)
k (n) : number of input neurons (input space dimension, N is total, k or n is iterator)
s : net input / pre-activation values, sjk = wj1 * x1 + wj2 * x2 + ... + wjk * xk + w0j
f(s) : activation function, f(s) = a * tanh(b * s)
J : summation of (real y for kth output neuron for ith input - estimated y for the same)^2
"""

# f : lambda function for convenience within larger function
#     according to Le Cun's suggestions, a = 1.7159 and b = 2/3
#     define larger function to perform activation on every neuron
# fprime : the derivative of f
f1 = lambda s : math.tanh(s)
f1prime = lambda s : 1 - math.pow(f1(s), 2)
f2 = lambda s : 1.7159 * math.tanh((2/3) * s)
f2prime = lambda s : 1.1439 / math.pow(math.cosh((2/3) * s), 2)
f3 = lambda s : .4 * math.tan(s + .2) - .08
f3prime = lambda s : 2 / (5 * math.pow(math.cos(s + .2), 2))
# f = f2
# fprime = f2prime

# Extended Linear Unit (modified ReLU)
ELU = lambda s : min(s, 1) if s >= 0 else .05 * (math.pow(math.e, s) - 1)
ELUprime = lambda s : 1 if s >= 0 else .05 * math.pow(math.e, s)

# used to normalize a vector of values between -1 and 1
normalize = lambda lst : [((el - min(lst)) / (max(lst) - min(lst))) * 2 - 1 for el in lst]


def normWeights(lst): 
	"""
 	Full function version for normalizing weight arrays according to overall
	max and min, rather than the max and min of the sublist. 

	Parameters
	---------------------------------------------------------
	lst : a 2D list containing a layer of weights. 
 
	Output
	---------------------------------------------------------
	norm_w : a 2D list containing the normalized layer of weights. 
	"""
 
	o_max = np.nanmax(lst)
	o_min = np.nanmin(lst)
	norm_w = []
	for sublist in lst: 
		norm_w += [[((el - o_min) / (o_max - o_min)) * 2 - 1 for el in sublist]]
	return norm_w


class NeuralNetwork: 
    """
    Backpropagating neural network with 1 hidden layer.     
    """
    
    def __init__(self, total_inputs = 784, total_hidden = 196, total_outputs = 784):
        """
        NeuralNetwork constructor method. Calls weightsInit(). 

        Parameters
        ---------------------------------------------------------------
        total_inputs : the total number of input neurons, which is how 
                       many pixels are in an image. Defaults to 784. 
                       (Ideally a square number, for heat mapping.)
        total_hidden : the total number of neurons in the hidden layer. 
                       Defaults to 196. (Also ideally a square number.)
        total_outputs : the total number of output neurons. Defaults to
                        784. (Also ideally a square number.)
        """
        # weights_jk = None, weights_ij = None, biases_jk = None, biases_ik = None):
        self.N = total_inputs 
        self.M = total_hidden
        self.L = total_outputs
        self.layer_dims = [self.N, self.M, self.L]

        self.weightsInit()
        
    
    def preloadWeights(self, file_wjk = None, file_wij = None, file_w0jk = None, file_w0ij = None): 
        """
        Method to preload weights into the network from a text file, given file name,
        rather than randomly generating and training weights. 

        Parameters
        ------------------------------------------------------------------
        file_wjk : the name of the text file containing the input-to-
                   hidden weights. Defaults to None. 
        file_wij : the name of the text file containing the hidden-to-
                   output weights. Defaults to None. 
        file_w0jk : the name of the text file containing the input-to-
                    hidden weight biases. Defaults to None. 
        file_w0ij : the name of the text file containing the hidden-to-
                    output weight biases. Defaults to None. 
        """
        
        if file_wjk != None: 
            self.w[0] = convertFileToList2D(file_wjk)
        if file_wij != None: 
            self.w[1] = convertFileToList2D(file_wij)
        if file_w0jk != None: 
            self.w0[0] = convertFileToList1D(file_w0jk, "float")
        if file_w0ij != None: 
            self.w0[1] = convertFileToList1D(file_w0ij, "float")
    

    def weightsInit(self): 
        """
        Intialize the weights according to how many hidden neurons desired
        using Xavier initialization: 
            - normal distribution between -a and a, where a ~ sqrt(6 / Ns + Nt)
            - Ns = total neurons in source layer 
            - Nt = total neurons in target layer
            
        OKAY ACTUALLY using a uniform distribution -a to a, a = sqrt(3/N)
        
        ACTUALLYYYY it's the Gaussian one right now
        """
        num_weight_sets = 2
        
        # intialize empty weights array, to be 3D
        self.w = [0.0] * num_weight_sets
        self.w0 = [0.0] * num_weight_sets
        
        # initialize wdelta nested array to hold the 
        # previous weight gradient for momentum calculation
        self.wdelta = [0.0] * num_weight_sets
        self.w0delta = [0.0] * num_weight_sets
        
        a = []
        # calculate a for the input-to-hidden weights
        a += [math.sqrt(3 / self.N)]
        # calculate a for the hidden-to-output weights
        a += [math.sqrt(3 / (self.M))]
        
        for idx in range(num_weight_sets): 
            self.w[idx] = np.random.uniform(-a[idx], a[idx], (self.layer_dims[idx + 1], self.layer_dims[idx]))
            self.w0[idx] = np.random.uniform(-a[idx], a[idx], self.layer_dims[idx + 1])
            # self.w[idx] = np.random.normal(0, a[idx], (self.layer_dims[idx + 1], self.layer_dims[idx]))
            np.random.normal
            # add arrays of 0s into the gradient weight tracker arrays
            self.wdelta[idx] = np.zeros_like(self.w[idx])
            self.w0delta[idx] = np.zeros_like(self.w0[idx])
            
        # save initial weights for viewing later
        self.w_init = [np.ndarray(np.shape(self.w[0])), np.ndarray(np.shape(self.w[1]))]
        self.w0_init = [np.ndarray(np.shape(self.w0[0])), np.ndarray(np.shape(self.w0[1]))]
        np.copyto(self.w_init[0], self.w[0])
        np.copyto(self.w_init[1], self.w[1])
        np.copyto(self.w0_init[0], self.w0[0])
        np.copyto(self.w0_init[1], self.w0[1])

    
    def preActivFuncHidden(self, xq): 
        """
        The preactivation function for the calculating the hidden 
        layer values. 
    
        Parameters
        ----------------------------------------------------------
        xq : input from input layer x, q signifying specific image 
             in dataset. 
        
        Output
        ----------------------------------------------------------
        sqj : the preactivation values for hidden neurons for 
              current image q. 
        """
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
        
    
    def activFuncHidden(self, q, h_idxs = None):
        """
        Activation function for calculating the hidden layer values. 
    
        Terminology
        ------------------------------------------------------------
        f(s) : activation function.
        
        Parameters
        ------------------------------------------------------------
        q : current image in the dataset.
        
        Output
        ------------------------------------------------------------
        hq : the hidden neurons' values for current image q.
        """
        # if dropout is enabled, h_idxs will be passed in as 
        # randomly chosen hidden neurons to tune to this image.
        # if dropout is not enabled, then set h_idxs to simply
        # include all hidden neurons. 
        if h_idxs == None: 
            h_idxs = list(range(self.M))
            
        # initialize hidden layer neurons for this image as list
        hq = [0.0] * self.M
        
        for j in range(self.M): 
            # if j is a hidden neuron to be used for this image
            if h_idxs.count(j) != 0: 
                hq[j] += ELU(self.sj[q][j])
            # if not, leave 0.0 as the list element for this neuron
            
            if DEBUG_FLAG == True and abs(hq[j]) >= 1.0: 
                True
            
        return hq
    

    def preActivFuncOutput(self, q): 
        """
        The preactivation function for the calculating the hidden layer values.
        
        Parameters
        -----------------------------------------------------------------------
        q : current image in dataset.

        Output
        -----------------------------------------------------------------------
        sqi : the preactivation values for output neurons for current image q.
        """
        
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
        
        
    def activFuncOutput(self, q):
        """
        This is the activation function for calculating the 
        output layer values. 
        
        Terminology
        --------------------------------------------------
        f(s) : activation function
            
        Parameters
        --------------------------------------------------
        q : current image in the dataset
        
        Output
        --------------------------------------------------
        yq : the result of the activation function, f(s). 
        """
        # initialize hidden layer neurons as list
        yq = []
        
        for i in range(self.L): 
            yq += [f2(self.si[q][i])]
            
            if DEBUG_FLAG == True and abs(yq[i]) >= 1.0: 
                True
            
        return yq
                
     
    def adjustWeights(self, q, q_idxs): 
        """
        adjust weights using backpropagation, momentum 
        
        Parameters
        -------------------------------------------------------
        q : current image in the dataset in the form of a list. 
        q_idxs : a list of the elements of q for which to use 
                 in the rho calculations. 
        """
        
        # reset rho for this image
        self.rho = [0.0] * self.M
        
        # the errors to propagate backwards
        self.delta_qi = [0.0] * self.L
        # the output y with the binary threshold parameters op_H and op_L enforced
        # y_bin = self.enforceBinary(self.y[q])
        
        # calculate weights wij (for hidden-to-output)
        # for every output neuron...
        for i in range(self.L): 
            # if isMatch(i) == False: 
            # calculate delta_q[i] to use for every input neuron j's weight adjustment
            # now for autoencoding reconstruction, it is x[q][i] - yhat[q][i] (generated y)
            # instead of actual y - generated y 
            self.delta_qi[i] = (self.x[q][i] - self.y[q][i]) * f2prime(self.si[q][i])
            # for every hidden neuron...
            for j in range(self.M): 
                # momentum = alpha * the current value stored in wdelta[1][i][j], 
                # as we are about to calculate the next at t, so this one is t - 1
                self.momentum_i = self.alpha * self.wdelta[1][i][j] 
                # calculate wdelta(t) (add the weight penalty for autoencoding regularization)
                self.wdelta[1][i][j] = self.eta * self.delta_qi[i] * self.h[q][j] + (self.lagrange * self.w[1][i][j])
                # update the weight with wdelta + momentum 
                self.w[1][i][j] += self.wdelta[1][i][j] + self.momentum_i
                
                # crazy experimental tangent function to try to stop explosions!!!!
                # self.w[1][i][j] = f1(self.w[1][i][j])

            # update bias weight w0
            self.momentum0_i = self.alpha * self.w0delta[1][i]
            self.w0delta[1][i] = self.eta * self.delta_qi[i] + (self.lagrange * self.w0[1][i])
            self.w0[1][i] += self.w0delta[1][i] + self.momentum0_i
            
            # crazy experimental tangent function to try to stop explosions!!!!
            # self.w0[1][i] = f1(self.w0[1][i])

        # errors to propagate for input-to-hidden weights 
        self.delta_qj = [0.0] * self.M 

        # calculate weights wjk (for input-to-hidden)
        # for every hidden neuron...
        for j in range(self.M): 
            # calculate rho[j]
            self.rho[j] = np.average([self.h[qq][j] for qq in q_idxs[0 : q_idxs.index(q) + 1]])
                
            # sort of, at least... it's taking its place mathematically
            self.delta_qj[j] = sum([self.w[1][i][j] * self.delta_qi[i] for i in range(self.L)]) 
            
            if self.rho[j] != 0.0 and self.rho[j] != 1.0:  # NOTE: how the hell did it equal 1.0
                self.delta_qj[j] -= (self.gamma * (((1 - self.rho_targ) / (1 - self.rho[j])) - (self.rho_targ / self.rho[j]))) 
            else: # NOTE: maybe try gamma being bigger than 1??
                pass
            
            self.delta_qj[j] *= ELUprime(self.sj[q][j])
            
            # for every input neuron...
            for k in range(self.N): 
                # momentum = alpha * the current value stored in wdelta[1][i][j], 
                # as we are about to calculate the next at t, so this one is t - 1
                self.momentum_j = self.alpha * self.wdelta[0][j][k] 
                # calculate wdelta(t) (add the weight penalty for autoencoding regularization)
                self.wdelta[0][j][k] = self.eta * self.delta_qj[j] * self.x[q][k] + (self.lagrange * self.w[0][j][k])
                # update the weight with wdelta + momentum 
                self.w[0][j][k] += self.wdelta[0][j][k] + self.momentum_j
                
                # crazy experimental tangent function to try to stop explosions!!!!
                # self.w[0][j][k] = f1(self.w[0][j][k])

            # update bias weight w0
            self.momentum0_j = self.alpha * self.w0delta[0][j]
            self.w0delta[0][j] = self.eta * self.delta_qj[j] + (self.lagrange * self.w0[0][j])
            self.w0[0][j] += self.w0delta[0][j] + self.momentum0_j
            
            # crazy experimental tangent function to try to stop explosions!!!!
            # self.w0[0][j] = f1(self.w0[0][j])
            
        # normalize all weights so they don't explode
        # self.w[0] = [normalize(wjk) for wjk in self.w[0]]
        # self.w[1] = [normalize(wij) for wij in self.w[1]]
        self.w[0] = normWeights(self.w[0])
        self.w[1] = normWeights(self.w[1])
        self.w0[0] = normalize(self.w0[0])
        self.w0[1] = normalize(self.w0[1])
        
    
    def run(self, x, ytrue, training = False, epochs = 125, stoch = True, dropout = False, 
            eta = .001, alpha = .3, op_H = .75, op_L = .25, gamma = 1, rho_target = .01, lagrange = .001): 
        """
        Define function for training the neural network.
        
        Parameters
        ----------------------------------------------------------------
        x : the input dataset for training the neural network
        ytrue : the answer key for x. 
        training : boolean flag that when set to False, will not train 
                   the hidden neurons. Defaults to False. 
        epochs : the number of times to run through the dataset to train
        stoch : boolean flag that determines if the network trains 
                stochastically. Defaults to True.
        dropout : boolean flag that determines if the network runs with
                  dropout. Defaults to False. 
        eta : learning rate, coefficient of weight adjustment. 
        alpha : momentum, velocity. 
        op_H : operating parameter such that yi >= H is considered
               a match for yi = +1. 
        op_L : operating parameter such that yi <= L is considered
               a match for yi = 0 or -1. 
        gamma : constant.
        rho_target : constant.
        lagrange : constant.
        """
        # self.x = [ELU(image) for image in x] # NOTE: I haven't started a run since changing this yet
        self.x = [[ELU(pixel) for pixel in image] for image in x]
        self.ytrue = ytrue
        self.eta = eta
        self.alpha = alpha
        self.op_H = op_H
        self.op_L = op_L
        self.gamma = gamma
        self.rho_targ = rho_target
        self.lagrange = lagrange
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
        self.Jnum = [[0.0]] * 10
        
        # initialize rho
        self.rho = []
        
        # epoch loop: run through the dataset as many times as given in epochs parameter
        for epoch in range(epochs): 
            if DEBUG_FLAG == True: 
                print("\n")
                print("".center(110, '*'))
                print("*   EPOCH  {:03d}   *".format(epoch + 1).center(110, '*'))
                print("".center(110, '*'))
                            
            # initialize hidden and output layer neurons, 
            # and preactivation values s for hidden and output calculation, 
            # nested array of each image's results
            self.h = [[0.0]] * self.Q
            self.y = [0.0] * self.Q
            self.si = [[0.0]] * self.Q
            self.sj = [[0.0]] * self.Q
            
            
            
            # add new nested list for this new epoch
            self.errs += [[]]
            self.Jq += [[]]
            
            if stoch == True: 
                # generate uniformly stochastic list of images q to run through the 
                # autoencoder for this epoch
                q_idxs = list(np.random.choice(range(self.Q), random.randint(self.Q // 4, self.Q - 1), replace=False))
            else: 
                # then set it to use all ints 0-Q
                q_idxs = range(self.Q)
            
            for q in q_idxs: 
                # generate hidden layer neuron data from input layer data
                self.sj[q] = self.preActivFuncHidden(self.x[q])
                
                # if dropout is enabled, then randomly choose a subset of 
                # hidden neurons to use to encourage diversity of features
                if dropout == True: 
                    h_idxs = list(np.random.choice(range(self.M), random.randint(self.M // 6, self.M - 1), replace=False))
                    self.h[q] = self.activFuncHidden(q, h_idxs)
                # if not true, then all hidden neurons will be considered
                else: 
                    self.h[q] = self.activFuncHidden(q)
                
                # generate output layer neuron data from hidden layer data
                self.si[q] = self.preActivFuncOutput(q)
                self.y[q] = self.activFuncOutput(q)
                
                if training == True: 
                    # adjust the weights based on this image
                    self.adjustWeights(q, q_idxs)
                
                # calculate error fraction
                # errs_q = sum([1 for ii in range(len(self.y[q])) if self.isCorrect(q, ii) == False])
                # self.errs += [errs_q]                    
                # self.err_frac += [errs_q / 10]
                # self.err_frac += [errs_q / ((q + 1) * self.L)]
                
                # append self.errs[] with a 1 if qth image was classified correctly, 0 if not
                # self.errs[epoch] += [self.isCorrect(q)]
                
                # calculate J2
                self.Jq[epoch] += list(([sum(math.pow(self.x[q][ii] - self.y[q][ii], 2) for ii in range(self.L)) * .5]
                                   + self.lagrange / 2 * np.sum(np.power(self.w[1], 2))) / self.L) 
                
                self.Jnum[self.ytrue[q]] += [sum(math.pow(self.x[q][ii] - self.y[q][ii], 2) for ii in range(self.L)) * .5]
                                        
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
                    
                if DEBUG_FLAG == True and q_idxs.index(q) % 100 == 0:
                    # print(".", end="")
                    print()
                    print("  Image #{:04d}  ".format(q + 1).center(110, '-'))
                    print("  {} out of {}  ---------  {:d}% through epoch  ".format(q_idxs.index(q), len(q_idxs), int(q_idxs.index(q) / len(q_idxs) * 100)).center(110, '-'))
                    
                    

                    
                # if DEBUG_FLAG == True and type(self.si[q]) == list: 
                #     if any([s >= 1 for s in self.si[q]]): 
                #         True

            # # normalize all weights so they don't explode, after every epoch
            # self.w[0] = [normalize(wjk) for wjk in self.w[0]]
            # self.w[1] = [normalize(wij) for wij in self.w[1]]
            # self.w0[0] = normalize(self.w0[0])
            # self.w0[1] = normalize(self.w0[1])
                
            # append self.err_frac with the error fraction for this epoch 
            # self.err_frac += [1 - np.average(self.errs[epoch])]
            
            if DEBUG_FLAG == True:
                print("\n")
                print("".center(110, '*'))
                print("   EPOCH {:03d} STATS   ".format(epoch + 1).center(110, '*'))
                print("  Jq = {:.6f}  ".format(self.getJq()).center(110, '*'))
                # print("  err_frac = {:.4f}  ".center(110, '*').format(self.err_frac[int(len(self.err_frac)) - 1]))
                print("".center(110, '*'))
                
                self.plotStats()
                plt.suptitle("Epoch {:03d}" .format(epoch + 1), fontweight="bold")
                self.plotFeatures()
                plt.suptitle("Epoch {:03d}: Weight Heatmaps for Features" .format(epoch + 1), fontweight="bold")
                plt.savefig("HW4/problem2/saved data/epoch {} features.png".format(epoch + 1))
                self.plotSampleOutputs()
                plt.suptitle("Epoch {:03d}: Original vs Reconstruction Heatmaps" .format(epoch + 1), fontweight="bold")
                plt.savefig("HW4/problem2/saved data/epoch {} sample outputs.png".format(epoch + 1))
                self.plotAllFeatures()
                plt.suptitle("Epoch {:03d}".format(epoch + 1), fontweight="bold")
                plt.savefig("HW4/problem2/saved data/epoch {} all features.png".format(epoch + 1))
                
                # if this is not the last epoch...
                if epoch < epochs - 1:
                    plt.pause(5)
                    # if len(self.Jq) > 1: 
                        # if np.average(self.Jq[-1]) > np.average(self.Jq[-2]): 
                        #     pass
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
        
    # returns the average of the last sublist element of self.Jq, 
    # the up-to-date loss value
    # epoch parameter is default set to -1 to reverse-index the last element
    def getJq(self, epoch = -1):
        return np.average(self.Jq[epoch])
    
    # error fraction calculator
    def errFrac(self, q = None):
        errs = 0
        if q == None: 
            q = self.Q
        total = q * self.L
        
        for qq in range(q): 
            errs += sum([1 for ii in range(self.y[qq]) if self.isCorrect(qq, ii) == False])
            
        return errs / total

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
      
    # plot randomly selecteed 20 hidden neurons as heatmaps
    def plotFeatures(self, colormap="bone"): 
        # generate 20 random, unique numbers to use as indices for the input/output neurons 
        # feats = [i for i in range(20)]
        feats = [int(el) for el in np.random.choice(self.M, min(20, self.M), replace=False)]
        plt.figure()
        for ii in range(min(20, self.M)): 
            # if the header neuron is an untouched one or if the index is a duplicate in the list,
            # change it
            # while len(self.h[int(feats[ii])]) <= 1:
            while len(self.h[feats[ii]]) <= 1 and sum([int(np.isin(feats, [feats[ii]])[idx]) for idx in range(len(feats))]) != 1: 
                feats[ii] = np.random.choice(self.M)
            plt.subplot(4, 5, ii + 1)
            plotImage(self.w[0][feats[ii]], cmap=colormap)    # plotImage() function from my functions.py file
            plt.title("Hidden Neuron " + str(feats[ii] + 1), fontweight="bold", fontsize=9)
            plt.xticks(range(0, 28, 7), fontsize=6)
            plt.yticks(range(0, 28, 7), fontsize=6)
        plt.suptitle("Figure 2.4: Sample Hidden Neurons' Features", fontweight="bold", fontsize=11)
        
    
    # randomly chooses 8 samples from the set to plot the original input image
    # and the reconstructed output image as heatmaps
    def plotSampleOutputs(self, origColor = "bone", outputColor = "bone"): 
        samples = [int(el) for el in np.random.choice(self.Q, 8, replace=False)]
        plt.figure()
        for ii in range(8): 
            # if the output neuron is an untouched one or 
            # if the index is a duplicate in the list, change it
            while self.y[samples[ii]] == 0.0 or len(self.y[samples[ii]]) <= 1 and sum([int(np.isin(samples, [samples[ii]])[idx]) for idx in range(len(samples))]) != 1: 
                samples[ii] = np.random.choice(self.Q)
            # plot original image
            plt.subplot(2, 8, ii + 1)
            plotImage(self.x[samples[ii]], origColor)
            plt.title("Original\nImage #{:04d}" .format(samples[ii]), fontweight="bold", fontsize=9)
            plt.xticks(range(0, 28, 7), fontsize=6)
            plt.yticks(range(0, 28, 7), fontsize=6)
            
            # plot reconstructed image
            plt.subplot(2, 8, ii + 9)
            plotImage(self.y[samples[ii]], outputColor)
            plt.title("Reconstructed\nImage #{:04d}" .format(samples[ii]), fontweight="bold", fontsize=9)
            plt.xticks(range(0, 28, 7), fontsize=6)
            plt.yticks(range(0, 28, 7), fontsize=6)
            
            plt.suptitle("Figure 2.5: Original and Reconstructed Image Comparison", fontweight="bold", fontsize=11)
        
    # plots all the features as heatmaps in one figure as subplots
    # THIS HAS TO BE HARDCODED FOR THE SUBPLOT ARRANGEMENTS
    def plotAllFeatures(self):   
        plt.figure()
        for hel in range(self.M): 
            plt.subplot(int(math.sqrt(self.M)), int(math.sqrt(self.M)), hel + 1)
            plt.imshow(np.transpose(np.reshape(self.w[0][hel], (28, 28))), cmap="bone")
            # plt.title("\n\n\n" + str(hel))
            plt.xticks([])
            plt.yticks([])
            

            
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
        if len(self.Jq) > 1: 
            plt.figure()
            # plt.subplot(2, 1, 1)
            plt.plot([x for x in range(1, len(self.Jq) + 1)], [self.getJq(ep) for ep in range(len(self.Jq))], 
                    color="xkcd:shamrock green")
            plt.xlabel("Epochs", fontweight="bold", fontsize=10)
            plt.ylabel("Loss", fontweight="bold", fontsize=10)
            # total images shown (duplicates included) / amount of images (duplicates not included)
            # plt.xticks([x for x in range(0, len(self.Jq), int(len(self.Jq) / 5))])
            plt.title("Figure 2.3: J2 Loss Function Over Time", fontweight="bold", fontsize=11)
            
            # plt.subplot(2, 1, 2)
            # plt.plot([x for x in range(1, len(self.err_frac) + 1)], self.err_frac, color="xkcd:fire engine red")
            # plt.xlabel("Epochs")
            # # total images shown (duplicates included) / amount of images (duplicates not included)
            # plt.xticks([i for i in range(0, len(self.err_frac), 10)])
            # plt.title("Error Fraction", fontweight="bold")
        
        
            


if __name__=="__main__":
    # get a list of the training set and its labels from .txt files
    trainset04 = convertFileToList2D("HW4/problem2/trainset04.txt")
    trainset04_labels = convertFileToList1D("HW4/problem2/trainset04_labels.txt")
    testset04 = convertFileToList2D("HW4/problem2/testset04.txt")
    testset04_labels = convertFileToList1D("HW4/problem2/testset04_labels.txt")
    testset59 = convertFileToList2D("HW4/problem2/testset59.txt")
    testset59_labels = convertFileToList1D("HW4/problem2/testset59_labels.txt")
    
    # initialize NeuralNetwork class
    Cortana = NeuralNetwork(total_hidden = 64)
    # with open("HW4\problem2\saved data\FINAL\Cortana_normw_big_eta", "rb") as NN_file:
    #     Cortana = pickle.load(NN_file)
    #     Gaia = pickle.load(NN_file)
    # Cortana.preloadWeights("HW3/trainset2_weights_jk.txt", "HW3/trainset2_weights_ij.txt", 
    #                     "HW3/trainset2_bias_jk.txt", "HW3/trainset2_bias_ij.txt")
    Cortana.run(trainset04, trainset04_labels, training=True, stoch=True, dropout=True,
                alpha=.4, eta=.06, rho_target=.01, gamma=3, lagrange=.01)  
    Jq_test04 = Cortana.getJq()
    Jnum_test04 = Cortana.Jnum.copy()

    # yyy : Cortana.y, the network's training output, with sublist lengths 
    # standardized so it can be written to .txt file output more easily
    # (numpy doesn't like "ragged" arrays with subarrays of varied lengths)
    yyy = []
    for iii in range(len(Cortana.y)): 
        if type(Cortana.y[iii]) == list:
            yyy += [Cortana.y[iii]]
        else: 
            yyy += [[0.0] * 784]
            
    # do the same thing for Jq 
    # (it's also ragged since its subarrays have 1 element for every output)
    JJJq = [np.average(sublist) for sublist in Cortana.Jq]
    # for iii in range(len(Cortana.Jq)): 
    #     if type(Cortana.Jq[iii]) == list:
    #         JJJq += [Cortana.Jq[iii]]
    #     else: 
    #         JJJq += [[0.0] * 4000]

    writeListToFile("HW4/problem2/saved data/testset04_output.txt", yyy)
    writeListToFile("HW4/problem2/saved data/testset04_weights_jk.txt", Cortana.w[0])
    writeListToFile("HW4/problem2/saved data/testset04_weights_ij.txt", Cortana.w[1])
    writeListToFile("HW4/problem2/saved data/testset04_bias_jk.txt", Cortana.w0[0])
    writeListToFile("HW4/problem2/saved data/testset04_bias_ij.txt", Cortana.w0[1])
    writeListToFile("HW4/problem2/saved data/testset04_Jq.txt", JJJq)

    # Gaia.run(testset59, testset59_labels, training=False, stoch=False, epochs=1,
    #             alpha=.4, eta=.06, rho_target=.01, gamma=3, lagrange=.01) 
    
    

    # # Cortana.preloadWeights("HW3/trainset_weights_jk.txt", "HW3/trainset_weights_ij.txt", 
    #                     #    "HW3/trainset_bias_jk.txt", "HW3/trainset_bias_ij.txt")
    # # for i in range(2): 
    # #     for idx in range(len(Cortana.w[i])): 
    # #         Cortana.w[i][idx] = normalize(Cortana.w[i][idx])
    # #     Cortana.w0[i] = normalize(Cortana.w0[i])
    # Cortana.run(testset, testset_labels, epochs=1, stoch=False)
    # Jq_test = Cortana.getJq()
    # Cortana.plotFeatures()

    # # MAKE BAR GRAPH FOR TOTAL
    # bar_width = .5
    # plt.figure()
    # plt.bar([1], [Jq_train], width=bar_width, label="Training Set", color="xkcd:heliotrope")
    # plt.bar([1 + bar_width], [Jq_test], width=bar_width, label="Test Set", color="xkcd:deep sky blue")
    # plt.xticks([1 + (bar_width / 2)], ["J2q"], fontweight="bold", fontsize=10)
    # plt.legend()
    # plt.title("Figure 2.1: Total Loss Value for Training and Test Sets", fontweight="bold", fontsize=11)

    # # MAKE BAR GRAPH FOR INDIVIDUAL NUMBERS
    # plt.figure()
    # to10 = list(range(10))
    # plt.bar([i for i in to10], Jnum_train, width=bar_width, label="Training Set", color="xkcd:heliotrope")
    # plt.bar([i + bar_width for i in to10], Cortana.Jnum, width=bar_width, label="Test Set", color="xkcd:deep sky blue")
    # plt.xticks([i + (bar_width / 2) for i in to10], to10, fontweight="bold", fontsize=10)
    # plt.legend()
    # plt.title("Figure 2.2: Loss Value for Each Digit", fontweight="bold", fontsize=11)

    # # MAKE HIDDEN WEIGHT_JK HEATMAPS
    # Cortana.plotFeatures()


    # plt.show(block=True)