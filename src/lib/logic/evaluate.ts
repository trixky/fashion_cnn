import * as tf from "@tensorflow/tfjs";
import normalize from "./normalize";

export default function evaluate(model: tf.Sequential, inputs: number[][], outputs: number[]): {
    answer: number,
    expected: number,
    input: number[],
} {
    const offset = Math.floor(Math.random() * inputs.length);
    const newInput = normalize([inputs[offset]], 0, 255);
    const inputTensor = newInput.reshape([1, 28, 28, 1]);

    const answer = tf.tidy(() => {
        // expandDims adds a dimension to the tensor
        const prediction = model.predict(inputTensor) as tf.Tensor<tf.Rank>;
        prediction.print();
        // squeeze is the opposite of expandDims
        return prediction.squeeze().argMax()
    })

    console.log(`Answer: ${answer.dataSync()[0]}, Expected: ${outputs[offset]}`);

    return {
        answer: answer.dataSync()[0],
        expected: outputs[offset],
        input: inputs[offset],
    }
}