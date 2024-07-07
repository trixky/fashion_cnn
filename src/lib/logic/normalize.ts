import * as tf from '@tensorflow/tfjs';

export default function normalize(inputs: number[][], min: number, max: number): tf.Tensor<tf.Rank> {
    const normalized = tf.tidy(function () {
        const MIN_VALUES = tf.scalar(min);
        const MAX_VALUES = tf.scalar(max);

        const INPUT_TENSOR = tf.tensor2d(inputs);
        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(INPUT_TENSOR, MIN_VALUES);

        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);


        return NORMALIZED_VALUES;
    });

    return normalized;
}