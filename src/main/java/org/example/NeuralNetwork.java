package org.example;

import java.util.Random;

public class NeuralNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double[] biasesHidden;
    private double[] biasesOutput;
    private Random random;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Initialize weights and biases
        weightsInputHidden = new double[inputSize][hiddenSize];
        weightsHiddenOutput = new double[hiddenSize][outputSize];
        biasesHidden = new double[hiddenSize];
        biasesOutput = new double[outputSize];
        random = new Random();

        // Initialize weights and biases randomly
        initializeWeightsAndBiases();
    }

    private void initializeWeightsAndBiases() {
        // Initialize weights randomly
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = random.nextDouble() - 0.5; // Random value between -0.5 and 0.5
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenOutput[i][j] = random.nextDouble() - 0.5; // Random value between -0.5 and 0.5
            }
        }

        // Initialize biases to zeros or small random values
        for (int i = 0; i < hiddenSize; i++) {
            biasesHidden[i] = 0.0;
        }
        for (int i = 0; i < outputSize; i++) {
            biasesOutput[i] = 0.0;
        }
    }

    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1 / (1 + Math.exp(-x[i]));
        }
        return result;
    }

    public double[] predict(double[] input) {
        // Forward propagation
        double[] hiddenLayerOutput = sigmoid(calculateHiddenLayerOutput(input));
        return sigmoid(calculateOutput(hiddenLayerOutput));
    }

    private double[] calculateHiddenLayerOutput(double[] input) {
        double[] hiddenLayerOutput = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; j++) {
                sum += input[j] * weightsInputHidden[j][i];
            }
            hiddenLayerOutput[i] = sum + biasesHidden[i];
        }
        return hiddenLayerOutput;
    }

    private double[] calculateOutput(double[] hiddenLayerOutput) {
        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenLayerOutput[j] * weightsHiddenOutput[j][i];
            }
            output[i] = sum + biasesOutput[i];
        }
        return output;
    }

    public void train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double[] input = inputs[i];
                double[] target = targets[i];

                // Forward propagation
                double[] hiddenLayerOutput = sigmoid(calculateHiddenLayerOutput(input));
                double[] output = sigmoid(calculateOutput(hiddenLayerOutput));

                // Backpropagation
                // Calculate output layer error
                double[] outputError = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    outputError[j] = (target[j] - output[j]) * output[j] * (1 - output[j]);
                }

                // Calculate hidden layer error
                double[] hiddenError = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < outputSize; k++) {
                        sum += outputError[k] * weightsHiddenOutput[j][k];
                    }
                    hiddenError[j] = sum * hiddenLayerOutput[j] * (1 - hiddenLayerOutput[j]);
                }

                // Update weights and biases
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < outputSize; k++) {
                        weightsHiddenOutput[j][k] += learningRate * outputError[k] * hiddenLayerOutput[j];
                    }
                }
                for (int j = 0; j < inputSize; j++) {
                    for (int k = 0; k < hiddenSize; k++) {
                        weightsInputHidden[j][k] += learningRate * hiddenError[k] * input[j];
                    }
                }
                for (int j = 0; j < outputSize; j++) {
                    biasesOutput[j] += learningRate * outputError[j];
                }
                for (int j = 0; j < hiddenSize; j++) {
                    biasesHidden[j] += learningRate * hiddenError[j];
                }
            }
        }
    }

    public static void main(String[] args) {


        int inputSize = 2;
        int hiddenSize = 3;
        int outputSize = 1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize, hiddenSize, outputSize);

        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] targets = {{0}, {1}, {1}, {0}};

        // Train the neural network
        int epochs = 10;
        double learningRate = 0.1;
        neuralNetwork.train(inputs, targets, epochs, learningRate);

        // Test the neural network
        System.out.println("Testing neural network predictions:");
        for (double[] input : inputs) {
            double[] predicted = neuralNetwork.predict(input);
            System.out.println("Input: " + java.util.Arrays.toString(input) + " => Predicted: " + java.util.Arrays.toString(predicted));
        }
    }

}

