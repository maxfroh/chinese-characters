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

# from https://stackoverflow.com/a/29402598 (Luke Taylor on Stack Overflow)
def line(x0, y0, x1, y1):
        "Bresenham's line algorithm"
        points_in_line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points_in_line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points_in_line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points_in_line.append((x, y))
        return points_in_line

def start_drawing(event):
    global points
    points.append((event.x, event.y))

def draw(event):
    global points
    old_x, old_y = points[-1]
    points.extend(line(old_x, old_y, event.x, event.y))
    points.append((event.x, event.y))
    canvas.create_line(points, width=LINE_WIDTH)
    
def finish_drawing(event):
    global points, line_stack
    line_stack.append(points)
    points = []
    save(event)
    
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
    # img = Image.new(mode="1", size=(DIM, DIM), color=255)
    for points in line_stack:
        for point in points:
            image_points.update(
                {
                    point, 
                    left(point), 
                    right(point), 
                    up(point), 
                    down(point)
                }
            )
    # for point in image_points:
    #     drawer = ImageDraw.Draw(img)
    #     drawer.point(point, fill=0)
    # img.show()
            
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