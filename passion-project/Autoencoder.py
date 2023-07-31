import random
import pickle
from functions import *

DEBUG_FLAG = True

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

f21 = lambda s : 3.4318 * math.tanh((2/3) * s) - 1      # 2 * f2(s) - 1

f = f2
fprime = f2prime

# Extended Linear Unit (modified ReLU)
ELU = lambda s : min(s, 1) if s >= 0 else .05 * (math.pow(math.e, s) - 1)
ELUprime = lambda s : 1 if s >= 0 else .05 * math.pow(math.e, s)

# used to normalize a vector of values between -1 and 1
normalize = lambda lst : [((el - min(lst)) / (max(lst) - min(lst))) * 2 - 1 for el in lst]

class Autoencoder: 
    """
    Autoencoder neural network with 1 hidden layer.  
    
    Dictionary for Terms and Variables
    --------------------------------------------------------------------------------------------
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
    
    def __init__(self, total_inputs: int = 784, total_hidden: int = 196, total_outputs: int = 784):
        """
        Autoencoder constructor method. Calls weightsInit(). 

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
        
        self.N = total_inputs 
        self.M = total_hidden
        self.L = total_outputs
        self.layer_dims = [self.N, self.M, self.L]

        self.weightsInit()
        
    
    def preloadWeights(self, 
                       file_wjk: str = None, file_wij: str = None, file_w0jk: str = None, file_w0ij: str = None): 
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
        
        TODO: is this even accurate at all lol
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
            # np.random.normal
            # # add arrays of 0s into the gradient weight tracker arrays
            self.wdelta[idx] = np.zeros_like(self.w[idx])
            self.w0delta[idx] = np.zeros_like(self.w0[idx])
            
        # save initial weights for viewing later # TODO: what's going on here semantically
        self.w_init = [np.ndarray(np.shape(self.w[0])), np.ndarray(np.shape(self.w[1]))]
        self.w0_init = [np.ndarray(np.shape(self.w0[0])), np.ndarray(np.shape(self.w0[1]))]
        np.copyto(self.w_init[0], self.w[0])
        np.copyto(self.w_init[1], self.w[1])
        np.copyto(self.w0_init[0], self.w0[0])
        np.copyto(self.w0_init[1], self.w0[1])

    
    def preActivFuncHidden(self, xq: list[float]): 
        """
        The preactivation function for the calculating the hidden 
        layer values. 
    
        Parameters
        ----------------------------------------------------------
        xq : input from input layer x, q signifying specific image 
             in dataset. 
        
        Returns
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
        
    
    def activFuncHidden(self, q: int, h_idxs: list[int] = None):
        """
        Activation function for calculating the hidden layer values. 
    
        Terminology
        ------------------------------------------------------------
        f(s) : activation function.
        
        Parameters
        ------------------------------------------------------------
        q : current image in the dataset.
        h_idxs : optional parameter used if dropout is enabled to
                 provide the sample subset of hidden neurons to
                 run through the activation function. If dropout is
                 disabled, defaults to None and uses all hidden
                 neurons. 
        
        Returns
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
                hq[j] += f1(self.sj[q][j])
            # if not, leave 0.0 as the list element for this neuron
            
            if DEBUG_FLAG == True and abs(hq[j]) >= 1.0: 
                True
            
        return hq
    

    def preActivFuncOutput(self, q: int): 
        """
        The preactivation function for the calculating the hidden layer values.
        
        Parameters
        -----------------------------------------------------------------------
        q : current image in dataset.

        Returns
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
        
        
    def activFuncOutput(self, q: int):
        """
        This is the activation function for calculating the 
        output layer values. 
        
        Terminology
        --------------------------------------------------
        f(s) : activation function
            
        Parameters
        --------------------------------------------------
        q : current image in the dataset
        
        Returns
        --------------------------------------------------
        yq : the result of the activation function, f(s). 
        """
        # initialize hidden layer neurons as list
        yq = []
        
        for i in range(self.L): 
            yq += [f(self.si[q][i])]
            
            if DEBUG_FLAG == True and abs(yq[i]) >= 1.0: 
                True
            
        return yq
                
     
    def adjustWeights(self, q_idxs: list[int]): 
        """
        Adjust weights using backpropagation and momentum. 
        
        Parameters
        -------------------------------------------------------
        q : number of the current image in the dataset, index 
            of self.x. 
        q_idxs : a list of the elements of q for which to use 
                 in the rho calculations. 
        """
        for q in q_idxs: 
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
                self.delta_qi[i] = (self.x[q][i] - self.y[q][i]) * fprime(self.si[q][i])
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
                
            self.w_max[1] += [np.nanmax(self.w[1])]
            self.w_min[1] += [np.nanmin(self.w[1])]

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
                
                self.delta_qj[j] *= fprime(self.sj[q][j])     # TODO: wtf is going on here
                
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
                
                self.features_lines[j].set(data=np.transpose(np.reshape(self.w[0][j], (28, 28))))
                
                # crazy experimental tangent function to try to stop explosions!!!!
                # self.w0[0][j] = f1(self.w0[0][j])
                
            # self.w[0] = self.normWeights(self.w[0])
            # self.w0[0] = normalize(self.w0[0])
                
            self.w_max[0] += [np.nanmax(self.w[0])]
            self.w_min[0] += [np.nanmin(self.w[0])]
                
            # normalize all weights so they don't explode
            # self.w[0] = [normalize(wjk) for wjk in self.w[0]]
            # self.w[1] = [normalize(wij) for wij in self.w[1]]
            # self.w[0] = self.normWeights(self.w[0])
            # self.w[1] = self.normWeights(self.w[1])
            # self.w0[0] = normalize(self.w0[0])
            # self.w0[1] = normalize(self.w0[1])
        
    
    def run(self, x: list[list[float]], ytrue: list[int], training = False, epochs = 125, stoch = True, dropout = False, 
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
        
        self.x = [[f21(pixel) for pixel in image] for image in x]
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
        
        self.plotAllFeatures()
        self.features_plot.suptitle("At Initialization", fontweight="bold", fontsize=6)
        self.features_plot.canvas.draw()
        self.features_plot.canvas.flush_events()
        
        # to make a line graph of it over time to see where it starts getting too bit
        self.w_max = [[], []]
        self.sj_max = []
        self.si_max = []
        self.h_max = []
        self.w_min = [[], []]
        self.sj_min = []
        self.si_min = []
        self.h_min = []
        
        
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
                q_idxs = list(np.random.choice(range(self.Q), random.randint(self.Q // 8, self.Q // 4 * 3), replace=False))
            else: 
                # then set it to use all ints 0-Q
                q_idxs = range(self.Q)
            
            for q in q_idxs: 
                # generate hidden layer neuron data from input layer data
                self.sj[q] = normalize(self.preActivFuncHidden(self.x[q]))
                
                self.sj_max += [max(max(self.sj))]
                self.sj_min += [min(min(self.sj))]
                
                # if dropout is enabled, then randomly choose a subset of 
                # hidden neurons to use to encourage diversity of features
                if dropout == True:     # TODO: maybe try changing the upper bound for randint? smaller sample size might decrease uniformity
                    h_idxs = list(np.random.choice(range(self.M), random.randint(int(self.M * .05), int(self.M * .75)), replace=False))
                    self.h[q] = self.activFuncHidden(q, h_idxs)
               
                # if not true, then all hidden neurons will be considered
                else: 
                    self.h[q] = self.activFuncHidden(q)
                    
                self.h_max += [max(max(self.h))]
                self.h_min += [min(min(self.h))]
                
                # generate output layer neuron data from hidden layer data
                self.si[q] = self.preActivFuncOutput(q)
                
                self.si_max += [max(max(self.si))]
                self.si_min += [min(min(self.si))]
                
                self.y[q] = self.activFuncOutput(q)
                
                # calculate error fraction
                # errs_q = sum([1 for ii in range(len(self.y[q])) if self.isCorrect(q, ii) == False])
                # self.errs += [errs_q]                    
                # self.err_frac += [errs_q / 10]
                # self.err_frac += [errs_q / ((q + 1) * self.L)]
                
                # append self.errs[] with a 1 if qth image was classified correctly, 0 if not
                # self.errs[epoch] += [self.isCorrect(q)]
                
                # calculate J2     TODO: what is going on here bruh there's so much
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
                    
            if training == True: 
                    # adjust the weights based on this image
                    self.adjustWeights(q_idxs)
                    
                    self.features_plot.suptitle("Epoch {}".format(epoch + 1), fontweight="bold", fontsize=6)
                    self.features_plot.canvas.draw()
                    self.features_plot.canvas.flush_events()        
                    
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
                # self.plotFeatures()
                # plt.suptitle("Epoch {:03d}: Weight Heatmaps for Features" .format(epoch + 1), fontweight="bold")
                # plt.savefig("HW4/problem2/saved data/epoch {} features.png".format(epoch + 1))
                self.plotSampleOutputs()
                plt.suptitle("Epoch {:03d}: Original vs Reconstruction Heatmaps" .format(epoch + 1), fontweight="bold")
                # plt.savefig("HW4/problem2/saved data/epoch {} sample outputs.png".format(epoch + 1))
                # self.plotAllFeatures()
                # plt.suptitle("All Features at Epoch {:03d}".format(epoch + 1), fontweight="bold")
                # # plt.savefig("HW4/problem2/saved data/epoch {} all features.png".format(epoch + 1))
                
                # if this is not the last epoch...
                if epoch < epochs - 1:
                    plt.pause(25)
                    # if len(self.Jq) > 1: 
                        # if np.average(self.Jq[-1]) > np.average(self.Jq[-2]): 
                        #     pass
                    plt.close("all")
                else: 
                    plt.show()
            
    
    def normWeights(self, lst): 
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
     
            
    def MSE(self):
        """
        The loss function, Mean-Squared Error (J2). 
        """
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
        

    def getJq(self, epoch: int = -1):
        """
        Returns the average of the last sublist element of self.Jq, the 
        up-to-date loss value. 

        Parameters
        ----------------------------------------------------------------
        epoch : to use as the index of self.Jq. The default is -1, to 
                reverse-index the last element. Default is always used
                in the written code, currently. 
    
        Returns
        ----------------------------------------------------------------
        The average loss value for the given epoch (default the latest).
        """
        return np.average(self.Jq[epoch])
    
    
    def errFrac(self, q: int = None):
        """
        Error fraction calculator. 
        
        Parameters
        --------------------------------------------------------
        q : a given image in the dataset? Defaults to self.Q.
        
        Returns
        --------------------------------------------------------
        errFrac : the error fraction, total number of wrong
                  digit classifications divided by total digits.
        """
        errs = 0
        if q == None: 
            q = self.Q
        total = q * self.L
        
        for qq in range(q): 
            errs += sum([1 for ii in range(self.y[qq]) if self.isCorrect(qq, ii) == False])
            
        return errs / total

    
    def enforceBinary(self, lst: list): 
        """
        Enforce the binary threshold parameters op_H and op_L on given list lst, 
        so that all elements of lst >= op_H are set to 1 and all elements of 
        lst <= op_L are set to 0. The resultant list is returned as a new list. 
        
        Parameters
        --------------------------------------------------------------------------
        lst : the given list to enforce the binary thresholds on. 
        
        Returns
        --------------------------------------------------------------------------
        lst_bin : new list for which the binary threshold parameters are enforced.
        """
        
        lst_bin = []
        
        for el in lst: 
            if el >= self.op_H: 
                lst_bin += [1]
            elif el <= self.op_L: 
                lst_bin += [0]
            else: 
                lst_bin += [el]
                
        return lst_bin      
      
    
    def plotFeatures(self, colormap: str = "bone"): 
        """ 
        Plot randomly selected 20 hidden neurons as heatmaps.
        
        Parameters
        --------------------------------------------------------
        colormap : defines the color scheme of the plots. 
                   Defaults to "bone". 
        """
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
        
    
    def plotSampleOutputs(self, origColor: str = "bone", outputColor: str = "bone"): 
        """
        Randomly chooses 8 samples from the set to plot the original input image
        and the reconstructed output image as heatmaps. 
     
        Parameters
        -------------------------------------------------------------------------
        origColor : the color scheme for the original images. Defaults to "bone".
        outputColor : the color scheme for the output plot. Defaults to "bone". 
        """
        
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
        
    
    def plotAllFeatures(self):   
        """
        Plots all the features as heatmaps in one figure as subplots.
        
        THIS HAS TO BE HARDCODED FOR THE SUBPLOT ARRANGEMENTS.
      
        TODO: Is this true though? I'll have to check. 
        """ 
        plt.ion()
        self.features_plot = plt.figure()
        self.features_axes = [[]] * self.M
        self.features_lines = [[]] * self.M
        for hel in range(self.M): 
            self.features_axes[hel] = self.features_plot.add_subplot(int(math.sqrt(self.M)), int(math.sqrt(self.M)), hel + 1)
            self.features_lines[hel] = self.features_axes[hel].imshow(np.transpose(np.reshape(self.w[0][hel], (28, 28))), cmap="bone")
            # plt.title("\n\n\n" + str(hel))
            plt.xticks([])
            plt.yticks([])
            

    def plotAllOutputs(self, outputColor: str = "bone"): 
        """
        Plots all the outputs as heatmaps in two figures as heatmaps. Hardcoded
        for a set of 2000 images, putting 1000 in each figure.  
     
        Parameters
        -------------------------------------------------------------------------
        outputColor : the color scheme for the output plot. Defaults to "bone". 
        """
        
        plt.figure()
        for q in range(self.Q // 2): 
            if type(self.y[q]) == list:
                plt.subplot(20, 50, q + 1)
                plt.imshow(np.transpose(np.reshape(self.y[q], (28, 28))), cmap=outputColor)
                plt.xticks([])
                plt.yticks([])
        
        plt.suptitle("Output Heatmaps 1-1000", fontweight="bold", fontsize=11)
        
        plt.figure()
        for q in range(self.Q // 2, self.Q): 
            if type(self.y[q]) == list: 
                plt.subplot(20, 50, (q + 1) % 1000)
                plt.imshow(np.transpose(np.reshape(self.y[q], (28, 28))), cmap=outputColor)
                plt.xticks([])
                plt.yticks([])
                
        plt.suptitle("Output Heatmaps 1001-2000", fontweight="bold", fontsize=11)
            
    
    def plotStats(self): 
        """
        Plot the J2 loss and error fraction. 
        """
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
 
    
    def plotMaxes(self): 
        """
        Plots the maxes of si, sj, wjk, wij, and h. 
        """ 
        plt.figure() 
        plt.plot([x for x in range(len(self.w_max[0]))], self.w_max[0], label="wjk")
        plt.plot([x for x in range(len(self.w_max[1]))], self.w_max[1], label="wij")
        plt.plot([x for x in range(len(self.sj_max))], self.sj_max, label="sj")
        plt.plot([x for x in range(len(self.si_max))], self.si_max, label="si")
        plt.plot([x for x in range(len(self.h_max))], self.h_max, label="h")
        
        plt.xlabel("Images Processed", fontweight="bold")
        plt.ylabel("Value", fontweight="bold")
        plt.legend()
        plt.title("Maximum Values of Attributes Per Image", fontweight="bold", fontsize=11)
        
        plt.show()
        
    
    def plotMins(self): 
        """
        Plots the mins of si, sj, wjk, wij, and h. 
        """ 
        plt.figure() 
        plt.plot([x for x in range(len(self.w_min[0]))], self.w_min[0], label="wjk")
        plt.plot([x for x in range(len(self.w_min[1]))], self.w_min[1], label="wij")
        plt.plot([x for x in range(len(self.sj_min))], self.sj_min, label="sj")
        plt.plot([x for x in range(len(self.si_min))], self.si_min, label="si")
        plt.plot([x for x in range(len(self.h_min))], self.h_min, label="h")
        
        plt.xlabel("Images Processed", fontweight="bold")
        plt.ylabel("Value", fontweight="bold")
        plt.legend()
        plt.title("Minimum Values of Attributes Per Image", fontweight="bold", fontsize=11)
        
        plt.show()