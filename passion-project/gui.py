# import tkinter as tk
from tkinter import *

# create root window
root = Tk()

root.title("NeuralNetwork GUI")
root.geometry("900x500")

leftspace = Frame(root, width=50, height=5)
leftspace.grid(column=0, row=0)

log = Text(root, state="normal", width=80, height=10, wrap="none")
log.grid(column=1, row=1, columnspan=3, rowspan=2)

def writeToLog(msg):
    log.insert('1.0', msg)

menu = Menu(root)
item = Menu(menu)
item.add_command(label="New")
menu.add_cascade(label="File", menu=item)
root.config(menu=menu)

label = Label(root, text="Neural Network Menu")
label.grid(column=2, row=3)

# txt = Entry(root, width=10)
# txt.grid(column=1, row=0)

def clicked(): 
    pass
    
button = Button(root, text="Start Network", fg="red", command=clicked)
button.grid(column=1, row=3)

writeToLog("oh my uwu\noh uwu BEAN!!!!")

root.mainloop()