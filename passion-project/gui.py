# import tkinter as tk
from tkinter import *
from functions import *


class LoggerWindow: 
    
    def __init__(self, root): 
        self.root = root
        self.log = Text(self.root, state="disabled", width=80, height=10, wrap="none", background="black", 
                        foreground="white", bd=6)
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

        # self.leftspace = Frame(self.root, width=50, height=5)
        # self.leftspace.grid(column=0, row=0)

        self.logger_window = LoggerWindow(self.root)
        self.logger_window.log.grid(column=2, row=1, columnspan=3, rowspan=2, padx=8, pady=8) 
        self.writeToLog(">")
        
        self.menu = Menu(self.root)
        self.item = Menu(self.menu)
        self.item.add_command(label="New")
        self.menu.add_cascade(label="File", menu=self.item)
        self.root.config(menu=self.menu)
            
        # self.create_autoencoder_button = Button(self.root, text="Create Autoencoder", fg="red", 
        #                                       command=self.createAutoencoder)
        # self.create_autoencoder_button.grid(column=1, row=3)

        # self.create_classifier_button = Button(self.root, text="Create Classifier", fg="blue", 
        #                                      command=self.createClassifier)
        # self.create_classifier_button.grid(column=2, row=3)
        
        self.labelframe_network = LabelFrame(self.root, text="Neural Network Type")
        self.labelframe_network.grid(column=0, row=0, rowspan=4, columnspan=2, sticky="nw")
        self.frame_network = Frame(self.labelframe_network, width=75, height=200)
        self.frame_network.grid(column=0, row=1, rowspan=3, columnspan=2)
        
        self.create_or_load = StringVar()
        self.rbutton_create = Radiobutton(self.labelframe_network, text="Create Network", variable=self.create_or_load,
                                          value="create", command=self.updateNetworkFrame)
        self.rbutton_create.grid(column=0, row=0)
        
        self.rbutton_load = Radiobutton(self.labelframe_network, text="Load Network", variable=self.create_or_load, 
                                        value="load", command=self.updateNetworkFrame)
        self.rbutton_load.grid(column=1, row=0)
        
        self.frame_load = Frame(self.labelframe_network)
        self.label_load = Label(self.frame_load, text="This is the loading frame")
        self.label_load.grid(column=0, row=1)
        
        self.frame_create = Frame(self.labelframe_network)
        # self.frame_create.grid(column=0, row=1)
        
        self.label_network_type = Label(self.frame_create, text="Network Type")
        self.label_network_type.grid(column=0, row=1)
        self.listbox_network_type = Listbox(self.frame_create, listvariable=StringVar(value=["Autoencoder", "Classifier"]), height=2)
        self.listbox_network_type.grid(column=0, row=2)
        
        self.label_hidden_neurons = Label(self.frame_create, text="Number of Hidden Neurons")
        self.label_hidden_neurons.grid(column=1, row=1) 
        self.spinbox_hidden_neurons = Spinbox(self.frame_create, from_=1, to=500, increment=1, )
        self.spinbox_hidden_neurons.grid(column=1, row=2)
        
        
        
    def createAutoencoder(self): 
        pass


    def createClassifier(self): 
        pass


    def writeToLog(self, msg):
        self.logger_window.addLine(msg)
        
        
    def updateNetworkFrame(self): 
        if self.create_or_load.get() == "create": 
            # self.frame_network.grid(column=0, row=1)
            self.frame_network.grid_remove()
            self.frame_load.grid_remove()
            self.frame_create.grid(column=0, row=1, columnspan=2, rowspan=3)
        elif self.create_or_load.get() == "load": 
            # self.frame_network.grid(column=0, row=1)
            # self.frame_network = self.frame_load
            self.frame_network.grid_remove()
            self.frame_create.grid_remove()
            self.frame_load.grid(column=0, row=1, columnspan=2, rowspan=3)
        
        # self.root.update()
        self.labelframe_network.update()

        
    def run(self): 
        # self.root.wait_variable(self.create_or_load)
        # self.updateNetworkFrame()
        self.root.mainloop()
    
    
    
if __name__=="__main__":
    gui = NeuralNetworkGUI()
    gui.run()
    