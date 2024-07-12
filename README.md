# Chinese Character Recognition

## Datasets used
[Chinese MNIST Dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist)

## Goals
- [X] Train model on Chinese MNIST dataset
- [X] Implement UI to draw characters
    - [X] Implement shrinking (256x256 => 64x64)
    - [ ] Convert to pygame for better performance? (branch)
- [X] Create mapper between drawing and tensor
- [ ] Implement guessing app
- [ ] Repeat for characters ?!


## Models
### Chinese MNIST
- Batch size: 64
- Epochs: 80

Results:
| Learning Rate | Accuracy (%)      | Loss               | Time (s)          |
| ------------- | ----------------- | ------------------ | ----------------- |
|0.1            | 6.768922805786133 | 13.175267219543457 | 439.2638692855835 | 
|0.01           | 20.70690155029297 | 2.7304136753082275 | 512.9765141010284 |
|0.001          | 85.59519958496094 | 5.0016565322875980 | 842.9604597091675 |
|0.0001         | 82.69422912597656 | 1.2334425449371338 | 480.1158037185669 |