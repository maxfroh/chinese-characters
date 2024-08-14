# Chinese Character Recognition

## Datasets Used
[Chinese MNIST Dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist)

Chinese MNIST Dataset - Extended (by me) (link not yet made)

[CASIA Online and Offline Databases](https://nlpr.ia.ac.cn/databases/handwriting/Home.html)

## About

This project's goal is to learn basic CNN concepts by creating an interactive app that allows users to draw Chinese numerals (and later characters) and then tries to guess what they have drawn. Information about the models I have made is below.

The numeral recognizer model uses a version of the Chinese MNIST dataset to which I have applied a variety of transformations (scaling, rotating) in order to increase sample size. I am currently working on extending the app to recognize characters using the CASIA database, both online and offline versions.

## Models
### Chinese MNIST


#### Results:
##### ChineseMNISTNN v1
- Dataset: Chinese MNIST
- Batch size: 64
- Epochs: 80
-
    ```python
    Linear(4096, 512) -> ReLU() -> Linear(512,512) -> ReLU() -> Linear(512,15)
    ```

| Learning Rate | Accuracy (%)      | Loss               | Time (s)          |
| ------------- | ----------------- | ------------------ | ----------------- |
|0.1            | 6.768922805786133 | 13.175267219543457 | 439.2638692855835 | 
|0.01           | 20.70690155029297 | 2.7304136753082275 | 512.9765141010284 |
|0.001          | 85.59519958496094 | 5.0016565322875980 | 842.9604597091675 |
|0.0001         | 82.69422912597656 | 1.2334425449371338 | 480.1158037185669 |

##### ChineseMNISTNN v2
- Dataset: Chinese MNIST
- Batch size: 64
- Epochs: 100
- 
    ```python 
    Linear(4096,2048) -> RELU() -> Linear(2048,2048) -> RELU() -> 
    Linear(2048,2048) -> RELU() -> Linear(2048,15) 
    ```

| Learning Rate | Accuracy (%)      | Loss               | Time (s)           |
| ------------- | ----------------- | ------------------ | ------------------ |
|0.01           | 6.26875638961792  | 2.7096245288848877 | 3522.8918516635895 |
|0.001          | 87.32910919189453 | 1.6266231536865234 | 673.1165864467621  |
|0.0001         | 87.16238403320312 | 1.4460784196853638 | 676.9658582210541  |

\* note: the longer time for learning rate `0.01` is due to the computer closing

##### ChineseMNISTNN v3
- Dataset: Chinese MNIST Extended
- Batch size: 32
- Epochs: 60
- 
    ```python 
    Conv2d(1, 32, 3) -> BatchNorm2d(32) -> ReLU() -> MaxPool2d(3) -> Dropout(0.3) ->
    Conv2d(32, 64, 3) -> BatchNorm2d(64) -> ReLU() -> MaxPool2d(3) -> Dropout(0.3) ->
    Conv2d(64, 64, 3) -> BatchNorm2d(64) -> ReLU() -> MaxPool2d(3) -> Dropout(0.3) ->
    Flatten() ->
    Linear(64, 512) -> ReLU() -> Linear(512, 512) -> ReLU() -> Linear(512, 15)
    ```

| Learning Rate | Accuracy (%)      | Loss               | Time (s)           |
| ------------- | ----------------- | ------------------ | ------------------ |
| 0.0001        | 97.79994358829714 | 0.0804631245949774 | 6745.724561929703  |


* Takeaways: Larger batch sizes/smaller learning rates may be more effective, but would require more epochs.

#### Chosen model: `chinese_mnist_model_0.0001_32_v3.pth`

- ~97.8% acurracy
