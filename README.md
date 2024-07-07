# digit_classifier

An [online](https://trixky.github.io/digit_classifier/) digit classifier using the [multi-layer perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP) model.

The model is trained on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) and implemented using the [TensorFlow.js](https://www.tensorflow.org/js) library.

> The model is loaded on the client side and all calculations are made on the device.

<img src="https://raw.githubusercontent.com/trixky/digit_classifier/main/.demo/screenshots.gif"  width="442">

## Setup

```bash
npm run install
npm run dev #localhost:5173
```

## Model caracteristics

- dataset: MNIST (10 000 samples)
- input layer: 784 neurons (28x28 pixels)
- hidden layers (2): 32 neurons (ReLU activation) + 16 neurons (ReLU activation)
- output layer: 10 neurons (Softmax activation)
- optimizer: Adam
- loss: categoricalCrossentropy
- metrics: accuracy
- epochs: 50
- batch size: 512
