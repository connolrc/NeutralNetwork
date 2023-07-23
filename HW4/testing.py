import matplotlib
import matplotlib.pyplot as plt
from functions import *
from hw4problem1 import NeuralNetwork
# # from problem1 import *
# # from HW3 import *

testset04 = convertFileToList2D("HW4/problem2/testset04.txt")
testset04_labels = convertFileToList1D("HW4/problem2/testset04_labels.txt")
testset59 = convertFileToList2D("HW4/problem2/testset59.txt")
testset59_labels = convertFileToList1D("HW4/problem2/testset59_labels.txt")
testout04 = convertFileToList2D("HW4/problem2/saved data/testset04_output.txt")
testout59 = convertFileToList2D("HW4/problem2/saved data/testset59_output.txt")

testsetall = testset04 + testset59
testoutall = testout04 + testout59


samples = np.random.randint(0, 100, 5)


plt.figure()
for num in range(10): 
    for ii in range(5): 
        # plot original image
        plt.subplot(20, 5, num * 10 + ii + 1)
        # num * 10 + ii + 1
        plotImage(testsetall[samples[ii] + (num * 100)], "bone")
        # plt.title("Original\nImage #{:04d}" .format(samples[ii] + (num * 100)), fontweight="bold", fontsize=9)
        # plt.xticks(range(0, 28, 7), fontsize=6)
        # plt.yticks(range(0, 28, 7), fontsize=6)
        plt.xticks([])
        plt.yticks([])

        # plot reconstructed image
        plt.subplot(20, 5, 10 * num + ii + 6)
        # (num + 1) * 5 + (num * 5) + ii + 1
        # 5num + 5 + 5num + ii + 1
        # 10num + ii + 6
        plotImage(testoutall[samples[ii] + (num * 100)], "bone")
        # plt.title("Reconstructed\nImage #{:04d}" .format(samples[ii] + (num * 100)), fontweight="bold", fontsize=9)
        # plt.xticks(range(0, 28, 7), fontsize=6)
        # plt.yticks(range(0, 28, 7), fontsize=6)
        plt.xticks([])
        plt.yticks([])


plt.suptitle("Figure 2.4: Original and Reconstructed Image Comparison", fontweight="bold", fontsize=11)

plt.pause(5)
plt.close("all")

# testset59_Jnum = convertFileToList1D("HW4/problem2/saved data/testset59_Jnum.txt")
# testset59_Jnum = [0.169416, 0.183319, 0.176979, 0.163391, 0.173001]

# # bar_width = .4
# to5 = list(range(0, 5, 1))
# plt.figure()
# plt.bar(to5, testset59_Jnum, label="Loss for each digit in 5-9 test set", color="xkcd:strawberry")
# plt.xticks(to5, list(range(5, 10)))
# plt.xlabel("Digit")
# plt.ylabel("J")
# plt.legend(loc="upper left")
# plt.title("Figure 2.3: Loss for Test Set of Numbers 5-9")
# plt.ylim(top=.24)
# plt.show()
# plt.savefig("HW4/problem2/saved data/FINAL/bar graph 5-9 Jnum.png")

# barwidth = .4
# plt.figure()
# plt.bar(1, 0.172281, width = .4, label = "Training Set")
# plt.bar(2, 0.172546, width = .4, label="Test Set 0-4")
# plt.title("Loss for Training Set and Test Set of 0-4 numbers")
# plt.legend()
# plt.xlabel("Set")
# plt.ylabel("J")
# plt.ylim(top=.2)
# plt.show()


# # wjk = convertFileToList2D("HW3/trainset2_weights_jk.txt")
# # w0jk = convertFileToList1D("HW3/trainset2_bias_jk.txt", "float")

# # M = 196

# # feats = [int(el) for el in np.random.choice(M, 20, replace=False)]
# # plt.figure()
# # for ii in range(20): 
# #     # if the header neuron is an untouched one or if the index is a duplicate in the list,
# #     # change it
# #     # while len(self.h[int(feats[ii])]) <= 1:
# #     #  while len(h[feats[ii]]) <= 1 and sum([int(np.isin(feats, [feats[ii]])[idx]) for idx in range(len(feats))]) != 1: 
# #     while sum([int(np.isin(feats, [feats[ii]])[idx]) for idx in range(len(feats))]) != 1:
# #         feats[ii] = np.random.choice(M)
# #     plt.subplot(4, 5, ii + 1)
# #     plotImage(wjk[feats[ii]], cmap="bone")    # plotImage() function from my functions.py file
# #     plt.title("Hidden Neuron " + str(feats[ii] + 1), fontsize=9)
# #     plt.xticks(range(0, 28, 7), fontsize=6)
# #     plt.yticks(range(0, 28, 7), fontsize=6)
# # plt.suptitle("Figure 2.4: Sample Hidden Neurons' Features", fontsize=11)

# # plt.show()


# matplotlib.rc('font', family='sans-serif')
# matplotlib.rcParams['font.sans-serif'].insert(0, 'Calibri')
# matplotlib.rc('font', size=12)
# matplotlib.rc('axes', titleweight='bold')
# matplotlib.rc('axes', titlesize=16)
# matplotlib.rc('axes', labelweight='bold')
# plt.rc('axes', labelsize=14)
# # matplotlib.rcParams['xaxis.labelsize'] = 44
# matplotlib.rc('figure', autolayout=True)
# # print(matplotlib.rcParams)
# # print(plt.rcParams == matplotlib.rcParams)

# #### generate plots for problem 1 ####

# # case 1
# Cortana_tr = NeuralNetwork()
# Cortana_te = NeuralNetwork()

# # training set
# Cortana_tr.y = convertFileToList2D("HW4/case1_trainset_output.txt")
# Cortana_tr.ytrue = convertFileToList1D("HW3/trainset_labels.txt")
# Cortana_tr.Q = 4000
# Cortana_tr.err_frac = convertFileToList1D("HW4/case1_trainset_err_frac.txt", "float")

# # confusion matrix for case 1 training set
# Cortana_tr.showConfusionMatrix()
# plt.title("Figure 1.1.1: Confusion Matrix for\nTraining Set with No Backpropagation")
# plt.savefig("HW4/plots/conf mat p1 c1 train set.png")

# # test set
# Cortana_te.y = convertFileToList2D("HW4/case1_testset_output.txt")
# Cortana_te.ytrue = convertFileToList1D("HW3/testset_labels.txt")
# Cortana_te.Q = 1000
# Cortana_te.err_frac = convertFileToList1D("HW4/case1_testset_err_frac.txt", "float")

# # confusion matrix for case 1 test set
# Cortana_te.showConfusionMatrix()
# plt.title("Figure 1.2.1: Confusion Matrix for\nTest Set with No Backpropagation")
# plt.savefig("HW4/plots/conf mat p1 c1 test set.png")

# # case 2
# Gaia_tr = NeuralNetwork()
# Gaia_te = NeuralNetwork()

# # training set
# Gaia_tr.y = convertFileToList2D("HW4/case2_trainset_output.txt")
# Gaia_tr.ytrue = convertFileToList1D("HW3/trainset_labels.txt")
# Gaia_tr.Q = 4000
# Gaia_tr.err_frac = convertFileToList1D("HW4/case2_trainset_err_frac.txt", "float")

# # confusion matrix for case 2 training set
# Gaia_tr.showConfusionMatrix()
# plt.title("Figure 1.1.2: Confusion Matrix for\nTraining Set with Backpropagation")
# plt.savefig("HW4/plots/conf mat p1 c2 train set.png")

# # test set
# Gaia_te.y = convertFileToList2D("HW4/case2_testset_output.txt")
# Gaia_te.ytrue = convertFileToList1D("HW3/testset_labels.txt")
# Gaia_te.Q = 1000
# Gaia_te.err_frac = convertFileToList1D("HW4/case2_testset_err_frac.txt", "float")

# # confusion matrix for case 2 test set
# Gaia_te.showConfusionMatrix()
# plt.title("Figure 1.2.2: Confusion Matrix for\nTest Set with Backpropagation")
# plt.savefig("HW4/plots/conf mat p1 c2 test set.png")

# # bar graph of mean error fraction for training and test sets, comparing case 1 and case 2
# bar_width = .2
# to11 = list(range(11))
# plt.figure()
# plt.bar([i - (bar_width * 1.5) for i in to11], [Cortana_tr.errFracForNum(d) for d in range(10)] + [Cortana_tr.err_frac[-1]], 
#         width=bar_width, label="Training set, without\nbackpropagation", color="xkcd:cerulean")
# plt.bar([i - (bar_width * .5) for i in to11], [Cortana_te.errFracForNum(d) for d in range(10)] + [Cortana_te.err_frac[-1]], 
#         width=bar_width, label="Test set, without\nbackpropagation", color="xkcd:lightblue")
# plt.bar([i + (bar_width * .5) for i in to11], [Gaia_tr.errFracForNum(d) for d in range(10)] + [Gaia_tr.err_frac[-1]], 
#         width=bar_width, label="Training set, with\nbackpropagation", color="xkcd:leaf green") # watermelon / coral pink
# plt.bar([i + (bar_width * 1.5) for i in to11], [Gaia_te.errFracForNum(d) for d in range(10)] + [Gaia_te.err_frac[-1]], 
#         width=bar_width, label="Test set, with\nbackpropagation", color="xkcd:light grey green") # light rose / pale salmon
# plt.xticks(to11, to11[0:10] + ["Overall"])
# plt.xlabel("Digit")
# plt.ylabel("Error Fraction")
# plt.legend(loc="upper left")
# plt.title("Figure 1.3: Error Fraction for Training and Test Sets")
# plt.savefig("HW4/plots/bar graph err p1.png")

# # error fraction time series 
# plt.figure()
# plt.plot(range(1, 42), Cortana_tr.err_frac, label="Without Backpropagation", color="xkcd:marigold")
# plt.plot(range(1, 42), Gaia_tr.err_frac, label="With Backpropagation", color="xkcd:strawberry")
# plt.xticks(range(0, 41, 10), range(0, 41, 10))
# # plt.xlim(left=1, right=41)
# plt.xlabel("Epochs")
# plt.yticks([round(y * .2, 1) for y in range(6)], [round(y * .2, 1) for y in range(6)])
# plt.ylabel("Error Fraction")
# plt.legend()
# plt.title("Figure 1.4: Error Fraction over Epochs")
# plt.savefig("HW4/plots/err frac time series p1.png")

# plt.show()