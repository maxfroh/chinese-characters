import tkinter as tk
import threading
from chinese_mnist_ui import *

def run_model(points):
    pass

def main():
    points = set()
    app = GUI()
    
    while(app.is_alive()):
        if(app.canvas != None):
            curr_points = app.canvas.get_points()
            if(curr_points != None and curr_points != points):
                points.update(curr_points)
                run_model(points)

if __name__ == '__main__':
    main()