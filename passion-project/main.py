from functions import *
from Autoencoder import *
from Classifier import *
import pickle

if __name__=="__main__": 
    # get a list of the training set and its labels from .txt files
    trainset = convertFileToList2D("passion-project/datasets/trainset.txt")
    trainset_labels = convertFileToList1D("passion-project/datasets/trainset_labels.txt")
    
    # initialize Autoencoder class
    Cortana = Autoencoder(total_hidden=144)
    Cortana.run(trainset, trainset_labels, training=True, dropout=True, gamma=2)
    
    pass