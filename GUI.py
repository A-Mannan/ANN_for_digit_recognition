#------------------ GUI for handwritten (drawn) digit recognition using ANN model -----------------

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import CENTER, LEFT, font , ROUND
from matplotlib.pyplot import text
from PIL import ImageGrab,Image, ImageOps

model=tf.keras.models.load_model('handwritten.model')

window=tk.Tk()
window.title("Digit Predictor")
main=tk.Frame(window,height=600,width=1050)
main.grid(columnspan=4)

part1=tk.Canvas(main,height=600,width=700,bg="black",highlightthickness=0)
part1.pack(side=LEFT)
part2=tk.Canvas(main,height=600,width=350,bg="#39FF13",highlightthickness=0)
part2.pack(side=LEFT)

programName=tk.Label(part2,text="DIGIT\nPREDICTOR",fg='black',bg='#39FF13',font=("QUARTZO bold",45))
programName.place(x=8,y=50)

button1=tk.Button(part2,text="Predict",bg="black",fg="#9CFF00",height=1,width=10,font=("QUARTZO",15),command=lambda:capture(canvas))
button1.place(x=33,y=263)
button2=tk.Button(part2,text="Clear",bg="black",fg="#9CFF00",height=1,width=10,font=("QUARTZO",15),command=lambda:clear())
button2.place(x=186,y=263)

canvas=tk.Canvas(part1,bg="black",height=500,width=500,highlightbackground="#39FF13")
canvas.place(relx=0.5,rely=0.5,anchor=CENTER)

label2=tk.Label(part2,text="Draw a digit...",bg="black",fg="#9CFF00",height=5,width=23,font=("KG Take On The World Regular",25),highlightbackground = "red", highlightcolor= "red")
label2.place(x=24,y=347)

current_x, current_y = 0,0

def locate_xy(event):
    global current_x, current_y
    current_x, current_y = event.x, event.y

def addLine(event):
    global current_x, current_y
    canvas.create_line((current_x,current_y,event.x,event.y),width=30,fill = 'white',capstyle=ROUND,smooth=True)
    current_x, current_y = event.x, event.y

canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>',addLine)

def capture(widget):
    x=window.winfo_rootx()+widget.winfo_x()
    y=window.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()
    img=ImageGrab.grab().crop((x,y,x1,y1))
    img=np.array(img)#, order='C')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(28,28),interpolation = cv2.INTER_CUBIC)
    img=img.reshape(1,28,28)
    # plt.imshow(img)
    # plt.show()
    # # print(ImageGrab.grab().crop((x,y,x1,y1)))
    if img is not None:
        result=np.argmax(model.predict(img)) 
        label2.config(text=f'You have drawn\n" {result} "')


def clear():
    canvas.delete('all')
    label2.config(text='Draw a digit...')

window.mainloop()