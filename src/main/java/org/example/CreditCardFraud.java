package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class CreditCardFraud {

    private List<double[]> data;

    public CreditCardFraud(String filename) throws IOException {
        this.data = readData(filename);
    }
    

    private List<double[]> readData(String filename) throws IOException {
        List<double[]> records = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean headerSkipped = false;
            while ((line = in.readLine()) != null) {
                if (!headerSkipped) {
                    headerSkipped = true;
                    continue; // Skip the header line
                }
                String[] parts = line.split(",");
                double[] values = new double[parts.length];
                for (int i = 0; i < parts.length - 1; i++) { // Exclude the last part (Class)
                    values[i] = Double.parseDouble(parts[i]);
                }
                // Parse the last part (Class) as an integer
                int classValue = Integer.parseInt(parts[parts.length - 1].replaceAll("\"", ""));
                values[parts.length - 1] = classValue;
                records.add(values);
            }
        }
        return records;
    }


    public void dataPreparation() {
        int numRows = data.size();
        int numCols = data.get(0).length;

        double[] classColumn = new double[numRows];
        for (int i = 0; i < numRows; i++) {
            classColumn[i] = data.get(i)[numCols - 1];
        }

        double[] fraud = new double[numRows];
        double[] valid = new double[numRows];
        int fraudCount = 0;
        int validCount = 0;
        for (int i = 0; i < numRows; i++) {
            if (classColumn[i] == 1.0) {
                fraud[fraudCount++] = data.get(i)[numCols - 2];
            } else {
                valid[validCount++] = data.get(i)[numCols - 2];
            }
        }
        fraud = trimArray(fraud, fraudCount);
        valid = trimArray(valid, validCount);

        double outlierFraction = (double) fraudCount / validCount;
        System.out.println(outlierFraction);
        System.out.println("Fraud Cases: " + fraudCount);
        System.out.println("Valid Transactions: " + validCount);

        System.out.println("###########################");

        System.out.println("Amount details of the fraudulent transaction");
        printStatistics(fraud);

        System.out.println();
        System.out.println("###########################");
        System.out.println();

        System.out.println("Details of valid transaction");
        printStatistics(valid);
    }

    private double[] trimArray(double[] array, int count) {
        double[] trimmedArray = new double[count];
        System.arraycopy(array, 0, trimmedArray, 0, count);
        return trimmedArray;
    }

    private void printStatistics(double[] array) {
        double sum = 0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (double num : array) {
            sum += num;
            if (num < min) {
                min = num;
            }
            if (num > max) {
                max = num;
            }
        }
        double mean = sum / array.length;
        double variance = 0;
        for (double num : array) {
            variance += Math.pow(num - mean, 2);
        }
        variance /= array.length;
        double stdDev = Math.sqrt(variance);
        System.out.println("Mean: " + mean);
        System.out.println("Standard Deviation: " + stdDev);
        System.out.println("Min: " + min);
        System.out.println("Max: " + max);
    }

    public void regressionModel() {
        int numRows = data.size();
        int numCols = data.get(0).length;

        double[][] X = new double[numRows][numCols - 1];
        double[] y = new double[numRows];

        for (int i = 0; i < numRows; i++) {
            double[] row = data.get(i);
            System.arraycopy(row, 0, X[i], 0, numCols - 1);
            y[i] = row[numCols - 1];
        }

        double splitRatio = 0.8;
        int splitIndex = (int) (numRows * splitRatio);

        double[][] X_train = new double[splitIndex][];
        double[][] X_test = new double[numRows - splitIndex][];
        double[] y_train = new double[splitIndex];
        double[] y_test = new double[numRows - splitIndex];

        Random rand = new Random();
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            indices.add(i);
        }
        for (int i = 0; i < splitIndex; i++) {
            int index = rand.nextInt(indices.size());
            int chosenIndex = indices.remove(index);
            X_train[i] = X[chosenIndex];
            y_train[i] = y[chosenIndex];
        }
        for (int i = 0; i < numRows - splitIndex; i++) {
            int chosenIndex = indices.get(i);
            X_test[i] = X[chosenIndex];
            y_test[i] = y[chosenIndex];
        }

        // Creating an instance of LogisticRegressionModel
        LogisticRegressionModel model = new LogisticRegressionModel();

        // Training the logistic regression model
        model.train(X_train, y_train, 100, 10);

        // Making predictions on X_test
        double[] predictions = model.predict(X_test);

        // Evaluate the model (you can implement this part)
    }

    public static void main(String[] args) {
        try {
            CreditCardFraud fraudDetection = new CreditCardFraud("./datasets/creditcard.csv");
            fraudDetection.dataPreparation();
            fraudDetection.regressionModel();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}