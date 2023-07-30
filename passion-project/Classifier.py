import random
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

f = f1
fprime = f1prime

# used to normalize a vector of values between -1 and 1
normalize = lambda lst : [((el - min(lst)) / (max(lst) - min(lst))) * 2 - 1 for el in lst]

class Classifier: 
    """
    Multi-layer feed-forward neural network with back-propagation including momentum. 1 hidden 
    layer. Takes handwritten digit images as input and outputs their classification. 
    
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
    
    # init (constructor) function
    def __init__(self, total_inputs: int = 784, total_hidden: int = 196, total_outputs: int = 10):
        """
        Classifier constructor method. Calls weightsInit(). 

        Parameters
        ---------------------------------------------------------------
        total_inputs : the total number of input neurons, which is how 
                       many pixels are in an image. Defaults to 784. 
                       (Ideally a square number, for heat mapping.)
        total_hidden : the total number of neurons in the hidden layer. 
                       Defaults to 196. (Also ideally a square number.)
        total_outputs : the total number of output neurons. Defaults to
                        10, because there are 10 digits to classify.
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
        a += [math.sqrt(2 / self.N)]
        # calculate a for the hidden-to-output weights
        a += [math.sqrt(2 / (self.M))]
        
        for idx in range(num_weight_sets): 
            # self.w[idx] = np.random.uniform(-a[idx], a[idx], (self.layer_dims[idx + 1], self.layer_dims[idx]))
            # self.w0[idx] = np.random.uniform(-a[idx], a[idx], self.layer_dims[idx + 1])
            self.w[idx] = np.random.normal(0, a[idx], (self.layer_dims[idx + 1], self.layer_dims[idx]))
            self.w0[idx] = np.random.normal(0, a[idx], self.layer_dims[idx + 1])
            
            # add arrays of 0s into the gradient weight tracker arrays
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
                hq[j] += f(self.sj[q][j])
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
                
    
    def adjustWeights(self, q: int, backprop = True): 
        """
        Adjust weights using backpropagation and momentum. 
        
        Parameters
        -------------------------------------------------------
        q : number of the current image in the dataset, index 
            of self.x. 
        backprop : defaults to True. wtf is going on here 
        """
        # lambda function to determine if the output neuron self.y[q][iy] has a value matching desired output.
        # used to determine those neurons that need weight adjustment (False output means adjustment needed).
        # the function returns False in these cases: 
        #   a) output neuron i is the correct answer but its value is < op_H
        #   b) output neuron i is a wrong answer but its value is > op_L
        isMatch = lambda iy : (True if (self.ytrue[q] == iy and self.y[q][iy] >= self.op_H) 
                                    or self.op_L <= self.y[q][iy]
                                else False)
        
        # the errors to propagate backwards
        self.delta_qi = [0.0] * self.L
        # the output y with the binary threshold parameters op_H and op_L enforced
        # y_bin = self.enforceBinary(self.y[q])
        
        # calculate weights wij (for hidden-to-output)
        # for every output neuron...
        for i in range(self.L): 
            if isMatch(i) == False: 
                # calculate delta_q[i] to use for every input neuron j's weight adjustment
                self.delta_qi[i] = (int(self.ytrue[q] == i) - self.y[q][i]) * fprime(self.si[q][i])
                # for every hidden neuron...
                for j in range(self.M): 
                    # momentum = alpha * the current value stored in wdelta[1][i][j], 
                    # as we are about to calculate the next at t, so this one is t - 1
                    momentum_i = self.alpha * self.wdelta[1][i][j] 
                    # calculate wdelta(t)
                    self.wdelta[1][i][j] = self.eta * self.delta_qi[i] * self.h[q][j]
                    # update the weight with wdelta + momentum 
                    self.w[1][i][j] += self.wdelta[1][i][j] + momentum_i

                # update bias weight w0
                momentum0_i = self.alpha * self.w0delta[1][i]
                self.w0delta[1][i] = self.eta * self.delta_qi[i] 
                self.w0[1][i] += self.w0delta[1][i] + momentum0_i

        if backprop == True: 
            # errors to propagate for input-to-hidden weights 
            self.delta_qj = [0.0] * self.M 

            # calculate weights wjk (for input-to-hidden)
            # for every hidden neuron...
            for j in range(self.M): 
                self.delta_qj[j] = fprime(self.sj[q][j]) * sum([self.w[1][i][j] * self.delta_qi[i] for i in range(self.L)])
                # for every input neuron...
                for k in range(self.N): 
                    # momentum = alpha * the current value stored in wdelta[1][i][j], 
                    # as we are about to calculate the next at t, so this one is t - 1
                    momentum_j = self.alpha * self.wdelta[0][j][k] 
                    # calculate wdelta(t)
                    self.wdelta[0][j][k] = self.eta * self.delta_qj[j] * self.x[q][k]
                    # update the weight with wdelta + momentum 
                    self.w[0][j][k] += self.wdelta[0][j][k] + momentum_j

                # update bias weight w0
                momentum0_j = self.alpha * self.w0delta[0][j]
                self.w0delta[0][j] = self.eta * self.delta_qj[j] 
                self.w0[0][j] += self.w0delta[0][j] + momentum0_j
            
    
    def run(self, x: list[list[float]], ytrue: list[int], training = False, epochs = 125, stoch = True, dropout = False, 
            eta = .001, alpha = .3, op_H = .75, op_L = .25, gamma = 1, rho_target = .01, lagrange = .001, backprop = True): 
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
        backprop : TODO: why is this here
        """
        # TODO: in Autoencoder.py self.x calls ELU() on every image pixel first. look into this
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
                if dropout == True:     # TODO: maybe try changing the upper bound for randint? smaller sample size might decrease uniformity
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
                    
                if DEBUG_FLAG == True and q_idxs.index(q) % 100 == 0:
                    print()
                    print("  Image #{:04d}  ".format(q + 1).center(110, '-'))
                    print("  {} out of {}  ---------  {:d}% through epoch  ".format(q_idxs.index(q), len(q_idxs), int(q_idxs.index(q) / len(q_idxs) * 100)).center(110, '-'))
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
            
            if DEBUG_FLAG == True:
                print("\n")
                print("".center(110, '*'))
                print("   EPOCH {:03d} STATS   ".center(110, '*').format(epoch + 1))
                # print("  Jq = {:.6f}  ".center(110, '*').format(np.average(self.Jq[len(self.Jq - 1)])))
                print("  err_frac = {:.4f}  ".center(106, '*').format(self.err_frac[int(len(self.err_frac)) - 1]))
                print("".center(110, '*'))
                
                self.plotStats()
                plt.suptitle("Epoch {:03d}" .format(epoch + 1), fontweight="bold")
                self.plotHeatmapsWij()
                plt.suptitle("Epoch {:03d}: Weight Heatmaps for Digits" .format(epoch + 1), fontweight="bold")
                self.plotHeatmapsWjk()
                self.showConfusionMatrix()
                # plt.show()
                # if this is not the last epoch...
                if epoch < epochs - 1:
                    plt.pause(25)
                    plt.close("all")
                else: 
                    plt.show()
            

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
    
    
    def getJq(self, epoch: int = -1): # TODO: is this the same in Classifier
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
        
        
    def getClass(self, q: int): 
        """
        Returns the network's classification of an image q by finding the
        max of the output neurons in list self.y[q]. 
        
        NOTE: This method does NOT check if the classification is correct, 
        it simply returns it. 
        
        Parameters
        ------------------------------------------------------------------
        q : the datapoint/image in the set
        
        Returns
        ------------------------------------------------------------------
        y_class : the class, digits 0-9, estimated by the network
        """
        if type(self.y[q]) == list: 
            return self.y[q].index(max(self.y[q]))
        else: # when the qth image was not randomly stochastically chosen, 
              # so the element at y[q] is just 0.0
            return None
    
    
    def isCorrect(self, q: int) -> bool: 
        """
        Checks if the network correctly classified image q. Calls
        getClass(q). 
        
        Parameters
        ----------------------------------------------------------
        q : the number of image to check whether the network
            correctly classified it. 
            
        Returns
        ----------------------------------------------------------
        isCorrect : whether image q was classified correctly, as a
                    boolean value. 
        """
        return self.ytrue[q] == self.getClass(q)
    
    
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
        
    
    def errFracForNum(self, num: int): 
        """
        Calculates error fraction for the specific number/digit given in parameter num.
        
        Parameters
        -------------------------------------------------------------------------------
        num : the number/digit for which to calculate the error fraction. 
        
        Returns
        -------------------------------------------------------------------------------
        errFrac : the error fraction for digit num. 
        """
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
    
    
    def plotHeatmapsWjk(self): 
        """
        Plots a sample 20 of the input-to-hidden weights as heatmaps. 
        """
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
    
    
    def plotHeatmapsWij(self): 
        """
        Plot hidden-to-output layer weights for each number as heat map images.
        """
        plt.figure()
        for ii in range(self.L):
            plt.subplot(2, 5, ii + 1)
            # plotImage() function from my functions.py file
            plotImage(self.w[1][ii], cmap="winter")
            plt.title("Number " + str(ii))

        plt.suptitle("Weight Heatmaps for Digits", fontweight="bold")
        
        
    def plotAllFeatures(self):   
        """
        Plots all the features as heatmaps in one figure as subplots.
        
        THIS HAS TO BE HARDCODED FOR THE SUBPLOT ARRANGEMENTS.
      
        TODO: Is this true though? I'll have to check. 
        """
        plt.figure()
        for hel in range(self.M): 
            plt.subplot(int(math.sqrt(self.M)), int(math.sqrt(self.M)), hel + 1)
            plt.imshow(np.transpose(np.reshape(self.w[0][hel], (28, 28))), cmap="bone")
            # plt.title("\n\n\n" + str(hel))
            plt.xticks([])
            plt.yticks([])
            
    
    def plotStats(self): 
        """
        Plot the J2 loss and error fraction.
        """
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
        
        
    def makeConfusionMatrix(self): 
        """
        Constructs a confusion matrix figure.
        """
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
                
    
    # TODO: why is this split into two methods? 
    def showConfusionMatrix(self): 
        """
        Plots a confusion matrix for the output results, which is a 
        table where... well, it's kinda hard to explain. You'll 
        understand when you see it, hopefully. 
        """
        self.makeConfusionMatrix()
        
        fix, axs = plt.subplots(1, 1)
        axs.axis("off")
        table = axs.add_table(plt.table(self.conmat, rowLabels=["   " + str(i) + "   " for i in range(10)], 
                                        colLabels=[i for i in range(10)], loc="center", cellLoc="center", 
                                        colLoc="center", rowColours=["lightgray" for i in range(10)], 
                                        colColours=["lightgray" for i in range(10)]))
        table.scale(1, 2)
        plt.title("Confusion Matrix", fontweight="bold")