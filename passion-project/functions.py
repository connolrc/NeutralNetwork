# passion-project functions.py

import matplotlib.pyplot as plt
import numpy as np
import linecache as lc
import math


def convertFileToImages(filename: str): 
    """
    Reads lines from the given text file. 
    Returns list of 28x28 float matrices, which are the images. 
    
    Parameters
    ----------------------------------------------------------
    filename : a str that is the name of the text file 
               containing the image data. 
               
    Returns
    ----------------------------------------------------------
    datalist : a list where each element is a 28x28 matrix
               representing an image from the file. 
    """
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
    
    textfile.close()
    # datalist = np.transpose(datalist)
    return datalist
    

def readDataLine(filename: str, row: int): 
    """
    Read line of data (one image) from given row in given file.
    Return 1x784 list of floats. 
    
    Parameters 
    ---------------------------------------------------------------
    filename : the name of the text file containing the image data.
    row : the number row which to read the data of. 
    
    Returns
    ---------------------------------------------------------------
    dataline : the image data from the rowth line of filename, as a
               list of floats. 
    """
    # convert the .txt file line to a list
    dataline = lc.getline(filename, row).split()
    dataline = [float(x) for x in dataline]
    
    return dataline


def convertListToImage(dataline: list[float]):
    """
    By default, convert a 1x784 list of floats representing
    an image into a 28x28 matrix of floats, to be able to 
    plot a heatmap of the image. 
    
    Otherwise, convert to a square matrix if possible, or 
    closest it can. TODO: This last part simply cannot be.

    Parameters
    --------------------------------------------------------
    dataline : a one-dimensional list of floats, by default 
               1x784. Contains an image's data. 
    
    Returns
    --------------------------------------------------------
    image_matrix : a square matrix of floats, 28x28 default. 
    """
    # convert the 1x784 list to a 28x28 matrix
    image_matrix = np.array(dataline)
    if len(image_matrix) == 784: 
        image_matrix = np.reshape(image_matrix, (28, 28))
    else: 
        root = math.sqrt(image_matrix.size)
        if root % 1 == 0:
            dim = int(root)
        else: 
            dim = int(root + 1)
        image_matrix.resize((dim, dim), refcheck=False)
    # transpose to orient image correctly upright
    image_matrix = np.transpose(image_matrix)
    
    return image_matrix


def plotImage(image_array: str, cmap = "bone"): 
    """
    Generate a heatmap plot from the given square float matrix
    representing an image. If the given image is in list form,
    call convertListToImage() first.
    
    Parameters
    -------------------------------------------------------------
    image_array : a matrix of floats representing an image.
                  Square by default, 28x28 by more default.
    cmap : the color scheme for the heatmap. Default of "autumn". 
    """
    if len(np.shape(image_array)) != 1: 
        plt.imshow(image_array, cmap, interpolation = "nearest")
    else: # if it is in 1x784 vector form...
        plt.imshow(convertListToImage(image_array), cmap, interpolation = "nearest")
    #plt.show()


def convertFileToList2D(filename: str) -> list[list[float]]: 
    """
    Create a list containing the contents of an image file. The 
    same as convertFileToImage() except it keeps the pixel 
    values subarray 1x784 instead of converting it to 28x28. 
    
    Parameters
    --------------------------------------------------------------
    filename : the name of the text file containing the images'
               data. 
               
    Returns
    --------------------------------------------------------------
    datalist : two-dimensional list where each element is a
               1x784 vector of floats containing one image's data. 
    """
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


def convertFileToList1D(filename: str, datatype = "int"): 
    """
    Create a list containing the contents of a label text file.
    
    Parameters
    -------------------------------------------------------------
    filename : name of the text file containing the image labels.
    datatype : a str containing the data type as which to store
               the labels. Defaults to "int". 
               
    Returns
    -------------------------------------------------------------
    datalist : a one-dimensional list containing the labels for
               the images. 
    """
    # open the file
    textfile = open(filename, "r")
    
    datalist = []
    row = 1
    for line in textfile: 
        if datatype == "int": 
            datalist += [int(lc.getline(filename, row)[0])]
        else: # "float"
            datalist += [float(lc.getline(filename, row))]
        row += 1
    
    return datalist
    
    
def writeListToFile(filename: str, lst: list, textformat = "%.6f"): 
    """
    Writes the contents of a list into the given text file.
    Essentially a wrapper function for numpy's savetext(). 
    
    Parameters
    ---------------------------------------------------------
    filename : the name of an existing file to store the list 
               in, or what to name the new file created. 
    lst : the list whose data to store in the file. 
    textformat : tells numpy's savetext() how to format the
                 text in the file. Default is with 6 digits 
                 right of the decimal. 
    """
    # open the file
    textfile = open(filename, "w")
    # arr = np.array(lst)
    np.savetxt(textfile, lst, fmt=textformat)
    textfile.close()
    

# # same as above, but for the 3D weights array in Problem 1
# def write3DListToFile(filename, lst, textformat = "%.6f"):
#     # open the file
#     textfile = open(filename, "w")
#     arr = np.array(lst)
#     np.savetxt(textfile, lst, fmt=textformat)
#     textfile.close()


def removeEmptySublists(lst: list[list]): 
    """
    Removes empty sublists, or non-list elements, in a given
    list of at least two dimensions. Used for saving list
    data to a text file in the writeListToFile() function.
    
    Parameters
    --------------------------------------------------------
    lst : a list of at least two dimensions. 
    
    Returns
    --------------------------------------------------------
    lst_copy : a copy of lst with empty sublists and non-
               list elements removed. 
    """
    
    lst = list(lst)
    lst_copy = lst.copy()
    for sublist in lst: 
        if type(sublist) != list: 
            lst_copy.remove(sublist)
        elif len(sublist) == 1: 
            lst_copy.remove(sublist)
    return lst_copy

def absMax(lst: list[list]): 
    """
    Returns the maximum element of a multi-dimensional list. 
    (A single value, not a list of maxes of each sublist.)
    
    Parameters
    --------------------------------------------------------
    lst : a multi-dimensional list of which to find the max.
    
    Returns
    --------------------------------------------------------
    absMax : the max value found in lst. 
    """
    return np.nanmax(np.absolute(lst))

def absMin(lst: list[list]): 
    """
    Returns the minimum element of a multi-dimensional list. 
    (A single value, not a list of mins of each sublist.)
    
    Parameters
    --------------------------------------------------------
    lst : a multi-dimensional list of which to find the min.
    
    Returns
    --------------------------------------------------------
    absMax : the min value found in lst. 
    """
    return np.nanmin(np.absolute(lst))
    
