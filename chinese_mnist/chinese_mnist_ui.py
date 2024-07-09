from tkinter import *
from tkinter import ttk
from PIL import Image, ImageDraw

window = Tk()
points: list[tuple[int, int]] = []
line_stack: list[list[tuple[int, int]]] = []

LINE_WIDTH = 4
DIM = 64 * 4

def left(point: tuple[int,int]) -> tuple[int,int]:
    return (point[0] - 1, point[1]) if point[0] - 1 >= 0 else point

def right(point: tuple[int,int]) -> tuple[int,int]:
    return (point[0] + 1, point[1]) if point[0] + 1 < DIM else point

def up(point: tuple[int,int]) -> tuple[int,int]:
    return (point[0], point[1] - 1) if point[1] - 1 >= 0 else point

def down(point: tuple[int,int]) -> tuple[int,int]:
    return (point[0], point[1] + 1) if point[1] + 1 < DIM else point

def start_drawing(event):
    global points
    points.append((event.x, event.y))

def draw(event):
    global points
    points.append((event.x, event.y))
    canvas.create_line(points, width=LINE_WIDTH)
    
def finish_drawing(event):
    global points, line_stack
    line_stack.append(points)
    print(line_stack)
    points = []
    
def undo(event):
    global points, line_stack
    if(len(line_stack) > 0):
        line_stack.pop()
        canvas.delete("all")
        for points in line_stack:
            canvas.create_line(points, width=LINE_WIDTH)
    points = []
    
def save(event):
    global line_stack
    image_points = set()
    img = Image.new(mode="1", size=(DIM, DIM), color=255)
    for points in line_stack:
        for point in points:
            image_points.add(point)
            image_points.add(left(point))
            image_points.add(right(point))
            image_points.add(up(point))
            image_points.add(down(point))
            
    for point in image_points:
        drawer = ImageDraw.Draw(img)
        drawer.point(point, fill=0)
    img.show()
            
def main():
    window.geometry(f"{DIM+40}x{DIM+40}")
    #window.iconwindow()
    label = Label(text="Draw a Chinese number | 回中文数字")
    label.pack()
    
    global canvas
    canvas = Canvas(window, bd=4, bg="white", width=DIM, height=DIM)
    canvas.pack()
    
    canvas.bind('<ButtonPress-1>', start_drawing)
    canvas.bind('<B1-Motion>', draw)
    canvas.bind('<ButtonRelease-1>', finish_drawing)
    window.bind('<Control-z>', undo)
    window.bind('<Control-d>', save)
    
    window.mainloop()

if __name__ == '__main__':
    main()