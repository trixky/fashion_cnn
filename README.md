# fashion_cnn

An [online](https://trixky.github.io/fashion_cnn/) [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN) model for classifying fashion items.

The model is trained on the [MNIST fashion dataset](https://www.tensorflow.org/datasets/catalog/fashion_mnist) and implemented using the [TensorFlow.js](https://www.tensorflow.org/js) library.

> The model is loaded on the client side and all calculations are made on the device.

<img src="https://raw.githubusercontent.com/trixky/fashion_cnn/main/.demo/screenshots.gif"  width="442">

## Setup

```bash
npm run install
npm run dev #localhost:5173
```

## Model caracteristics

- dataset: MNIST (fashion) (10 000 samples)
- input layer: 784 (28x28 pixels)
- convolutional layers (4):
    * 16(28x28)[5x5 filter] + stride[2] & max pool[2x2]
    * 32(14x14)[5x5 filter] + stride[2] & max pool[2x2]
- hidden layers (1): 128 neurons (ReLU activation)
- output layer: 10 (Softmax activation)
- optimizer: Adam
- loss: categoricalCrossentropy
- metrics: accuracy
- epochs: 10
- batch size: 512
