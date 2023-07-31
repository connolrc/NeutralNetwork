# import tkinter as tk
from tkinter import *
from functions import *


class LoggerWindow: 
    
    def __init__(self): 
        self.log = Text(self.root, state="disabled", width=80, height=10, wrap="none", background="black", 
                        foreground="white")
        self.num_lines = 0
        
    def addLine(self, msg): 
        self.log['state'] = 'normal'
        self.log.insert('end', msg)
        
        self.log['state'] = 'disabled'
        
        self.num_lines += 1


class NeuralNetworkGUI: 
    
    def __init__(self): 
        # create root window
        self.root = Tk()

        self.root.title("NeuralNetwork GUI")
        self.root.geometry("900x500")

        self.leftspace = Frame(self.root, width=50, height=5)
        self.leftspace.grid(column=0, row=0)

        self.log = LoggerWindow
        self.log.grid(column=1, row=1, columnspan=3, rowspan=2) # TODO
        
        self.menu = Menu(self.root)
        self.item = Menu(self.menu)
        self.item.add_command(label="New")
        self.menu.add_cascade(label="File", menu=self.item)
        self.root.config(menu=self.menu)

        self.label = Label(self.root, text="Neural Network Menu")
        self.label.grid(column=2, row=3)

        # txt = Entry(self.root, width=10)
        # txt.grid(column=1, row=0)
            
        self.create_autoencoder_button = Button(self.root, text="Create Autoencoder", fg="red", 
                                              command=self.createAutoencoder)
        self.create_autoencoder_button.grid(column=1, row=3)

        self.create_classifier_button = Button(self.root, text="Create Classifier", fg="blue", 
                                             command=self.createClassifier)
        self.create_classifier_button.grid(column=2, row=3)
        
        self.hidden_neurons_spinbox = Spinbox(self.root, from_=1, to=500, increment=1, )
        self.hidden_neurons_spinbox.grid(column=2, row=4)
        
        
    def createAutoencoder(self): 
        pass


    def createClassifier(self): 
        pass


    def writeToLog(self, msg):
        self.log.addLine(msg)

        
    def run(self): 
        self.root.mainloop()
    
    
    
if __name__=="__main__":
    gui = NeuralNetworkGUI()
    gui.run()