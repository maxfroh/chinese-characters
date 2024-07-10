from tkinter import *
import threading
from PIL import Image, ImageDraw

class GUI(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.root = None
        self.canvas = None
        self.start()
    
    def callback(self):
        self.root.quit()
        
    def run(self):
        self.root = Tk()
        self.root.geometry(f"{CMNISTCanvas.DIM+40}x{CMNISTCanvas.DIM+40}")
        label = Label(text="Draw a Chinese number | 回中文数字")
        label.pack()
        
        self.canvas = CMNISTCanvas(self.root, bd=4, bg="white", width=CMNISTCanvas.DIM, height=CMNISTCanvas.DIM)
        self.canvas.pack()
        
        self.canvas.bind('<ButtonPress-1>', self.canvas._start_drawing)
        self.canvas.bind('<B1-Motion>', self.canvas._draw)
        self.canvas.bind('<ButtonRelease-1>', self.canvas._finish_drawing)
        self.root.bind('<Control-z>', self.canvas._undo)
        self.root.bind('<Control-d>', self.canvas._save)
        
        self.root.mainloop()
        del self.root

class CMNISTCanvas(Canvas):
    LINE_WIDTH = 4
    DIM = 64 * 4
    
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.points:list[tuple[int, int]] = []
        self.line_stack:list[list[tuple[int,int]]] = []
        self.image_points:set[tuple[int,int]] = None
        
    def get_points(self):
        return self.image_points
    
    def _left(self, point: tuple[int,int]) -> tuple[int,int]:
        return (point[0] - 1, point[1]) if point[0] - 1 >= 0 else point

    def _right(self, point: tuple[int,int]) -> tuple[int,int]:
        return (point[0] + 1, point[1]) if point[0] + 1 < CMNISTCanvas.DIM else point

    def _up(self, point: tuple[int,int]) -> tuple[int,int]:
        return (point[0], point[1] - 1) if point[1] - 1 >= 0 else point

    def _down(self, point: tuple[int,int]) -> tuple[int,int]:
        return (point[0], point[1] + 1) if point[1] + 1 < CMNISTCanvas.DIM else point
    
    # from https://stackoverflow.com/a/29402598 (Luke Taylor on Stack Overflow)
    def _calculate_line_points(self, x0, y0, x1, y1):
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

    def _start_drawing(self, event):
        self.points.append((event.x, event.y))

    def _draw(self, event):
        old_x, old_y = self.points[-1]
        self.points.extend(self._calculate_line_points(old_x, old_y, event.x, event.y))
        self.points.append((event.x, event.y))
        super().create_line(self.points, width=CMNISTCanvas.LINE_WIDTH)
        
    def _finish_drawing(self, event):
        self.line_stack.append(self.points)
        self.points = []
        self._save(event)
        
    def _undo(self, event):
        if(len(self.line_stack) > 0):
            self.line_stack.pop()
            super().delete("all")
            for points in self.line_stack:
                super().create_line(points, width=CMNISTCanvas.LINE_WIDTH)
        self.points = []
        
    def _save(self, event):
        image_points = set()
        for points in self.line_stack:
            for point in points:
                image_points.update(
                    {
                        point, 
                        self._left(point), 
                        self._right(point), 
                        self._up(point), 
                        self._down(point)
                    }
                )
        self.image_points = image_points
