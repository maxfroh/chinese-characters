import torch, cv2
import numpy as np
from torch import Tensor
from chinese_mnist_ui import *
from chinese_mnist_data import *
import keyboard

BASE_DIM = 64

def points_to_tensor(points:set[tuple[int,int]]) -> Tensor:
    """
    Shrinks a set of points (representing a `BASE_DIM*4` x `BASE_DIM*4` image)
    to be 1/16 the size (`BASE_DIM` x `BASE_DIM` image) and converts to a tensor.
    """
    large_array = np.zeros((BASE_DIM * 4, BASE_DIM * 4))
    for point in points:
        if(0 <= point[0] < BASE_DIM * 4 and 0 <= point[1] < BASE_DIM * 4):
            large_array[point[1]][point[0]] = 1
    resized_array = cv2.resize(large_array, dsize=(BASE_DIM, BASE_DIM), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized_array).float()
    return tensor  

def run_model(model, points):
    drawing_tensor = points_to_tensor(points)
    logits = model(drawing_tensor)
    pred_probabilities = nn.Softmax(dim=0)(logits)
    pred = pred_probabilities.argmax().item()
    pred = ChineseMNISTDataset.map[pred]
    return pred

def main():
    model = ChineseMNISTNN()
    model.load_state_dict(torch.load('chinese_mnist_model_0.001_v2.pth'))
    model.eval()
    print(model)
    points = set()
    app = GUI()
    
    # manual save
    keyboard.add_hotkey('ctrl+d', lambda: run_model(points))
    
    while(app.is_alive()):
        if(app.canvas != None):
            curr_points = app.canvas.get_points()
            if(curr_points != None and curr_points != points):
                points = curr_points
                result = run_model(model, points)
                print(f"Guess: {result}")
                app.set_guess(result)

if __name__ == '__main__':
    main()