import * as tf from "@tensorflow/tfjs";
import model from './model';

export default async function train(inputs: number[][], outputs: number[], callBack: (epoch: number, accuracy: number) => void): Promise<tf.Sequential> {
    // Shuffle the two arrays in the same way so inputs still match outputs indexes.
    tf.util.shuffleCombo(inputs, outputs);
    // inputs feature Array is 1 dimensional.
    const inputTensor = tf.tensor2d(inputs);
    // Output feature Array is 1 dimensional.
    const outputTensor = tf.oneHot(tf.tensor1d(outputs, 'int32'), 10);

    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    const result = await model.fit(inputTensor, outputTensor, {
        shuffle: true,
        validationSplit: 0.2,
        batchSize: 512,
        epochs: 50,
        callbacks: {
            onEpochEnd: (epoch: number, logs: tf.Logs | any) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
                callBack(epoch, logs.acc);
            }
        }
    });

    inputTensor.dispose();
    outputTensor.dispose();

    return model;
}