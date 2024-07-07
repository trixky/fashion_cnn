import * as tf from "@tensorflow/tfjs";

const model = tf.sequential();

// --------------------------------------- Convolutional Neural Network ---------------------------------------
model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 16,
    kernelSize: 3, // Square Filter of 3 by 3. Could also specify rectangle eg [2, 3].
    strides: 1,
    padding: 'same',
    activation: 'relu'
}));

model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    strides: 1,
    padding: 'same',
    activation: 'relu'
}));

model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

// --------------------------------------- Multi Layer Perceptron ---------------------------------------
model.add(tf.layers.flatten());


model.add(tf.layers.dense({units: 128, activation: 'relu'}));


model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

export default model;