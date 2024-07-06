import * as tf from "@tensorflow/tfjs";

const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [784],
    activation: 'relu',
    units: 32
}));

model.add(tf.layers.dense({
    activation: 'relu',
    units: 16
}));

model.add(tf.layers.dense({
    activation: 'softmax',
    units: 10
}));

export default model;