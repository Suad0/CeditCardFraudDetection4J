package org.example;

public class LogisticRegressionModel {

    private double[][] coefficients;

    public LogisticRegressionModel() {
    }

    public void train(double[][] X_train, double[] y_train, double learningRate, int iterations) {
        int numFeatures = X_train[0].length;
        coefficients = new double[1][numFeatures + 1]; // Include bias term

        // Add bias term to features (append 1 to each row of X_train)
        double[][] augmentedX_train = new double[X_train.length][numFeatures + 1];
        for (int i = 0; i < X_train.length; i++) {
            augmentedX_train[i][0] = 1; // Bias term
            System.arraycopy(X_train[i], 0, augmentedX_train[i], 1, numFeatures);
        }

        // Gradient Descent
        for (int iter = 0; iter < iterations; iter++) {
            double[] predictions = predictProbability(augmentedX_train);

            double[] errors = new double[y_train.length];
            for (int i = 0; i < y_train.length; i++) {
                errors[i] = predictions[i] - y_train[i];
            }

            for (int j = 0; j < numFeatures + 1; j++) { // Include bias term
                double gradient = 0.0;
                for (int i = 0; i < X_train.length; i++) {
                    gradient += errors[i] * augmentedX_train[i][j];
                }
                coefficients[0][j] -= learningRate * gradient / X_train.length;
            }
        }
    }


    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    private double[] predictProbability(double[][] X) {
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double linearCombination = 0.0;
            for (int j = 0; j < X[0].length; j++) {
                linearCombination += coefficients[0][j] * X[i][j];
            }
            predictions[i] = sigmoid(linearCombination);
        }
        return predictions;
    }


    public double[] predict(double[][] X_test) {
        double[][] augmentedX_test = new double[X_test.length][X_test[0].length + 1];
        for (int i = 0; i < X_test.length; i++) {
            augmentedX_test[i][0] = 1; // Bias term
            System.arraycopy(X_test[i], 0, augmentedX_test[i], 1, X_test[0].length);
        }
        return predictProbability(augmentedX_test);
    }


}