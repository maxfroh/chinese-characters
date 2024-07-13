# Chinese Character Recognition

## Datasets Used
[Chinese MNIST Dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist)


## Goals
- [X] Train model on Chinese MNIST dataset
    - [ ] Investigate if the model can be changed for performance
    - [ ] See if the 六 diagnosis is a feature of the model
- [X] Implement UI to draw characters
    - [X] Implement shrinking (256x256 => 64x64)
    - [ ] Convert to pygame for better performance? (branch)
    - [ ] Investigate modifying drawing interface/shrinking technique for accuracy
- [X] Create mapper between drawing and tensor
- [X] Implement guessing app
- [ ] Repeat for characters
    - [ ] New branch with new model


## Models
### Chinese MNIST


#### Results:
##### ChineseMNISTNN v1
- Batch size: 64
- Epochs: 80
- `Sequential(Linear(4096->512),RELU,Linear(512->512),RELU,Linear(512->15))`

| Learning Rate | Accuracy (%)      | Loss               | Time (s)          |
| ------------- | ----------------- | ------------------ | ----------------- |
|0.1            | 6.768922805786133 | 13.175267219543457 | 439.2638692855835 | 
|0.01           | 20.70690155029297 | 2.7304136753082275 | 512.9765141010284 |
|0.001          | 85.59519958496094 | 5.0016565322875980 | 842.9604597091675 |
|0.0001         | 82.69422912597656 | 1.2334425449371338 | 480.1158037185669 |

##### ChineseMNISTNN v2
- Batch size: 64
- Epochs: 100
- `Sequential(Linear(4096->2048),RELU,Linear(2048->2048),RELU,Linear(2048->2048),RELU,Linear(2048->15))`

| Learning Rate | Accuracy (%)      | Loss               | Time (s)           |
| ------------- | ----------------- | ------------------ | ------------------ |
|0.01           | 6.26875638961792  | 2.7096245288848877 | 3522.8918516635895 |
|0.001          | 87.32910919189453 | 1.6266231536865234 | 673.1165864467621  |
|0.0001         | 87.16238403320312 | 1.4460784196853638 | 676.9658582210541  |

\* note: the longer time for learning rate `0.01` is due to the computer closing

#### Chosen model: `chinese_mnist_model_0.001.pth`

- ~85.6% acurracy
