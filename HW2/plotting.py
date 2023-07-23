import matplotlib.pyplot as plt
from functions import convertFileToImages, plotImage

# creates a list of all the lines in the given file. 
# each line contains the data making up one image. 
datalist = convertFileToImages("images8.txt")

# these are just random indexes I put in to get a good look
plotImage(datalist[0])
plt.figure();
plotImage(datalist[100])
plt.figure(); 
plotImage(datalist[284])
plt.figure();
plotImage(datalist[499])
#plt.figure();
plt.show();