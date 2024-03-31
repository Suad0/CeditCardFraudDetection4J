package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RandomForest {

    private List<DecisionTree> trees;
    private int numTrees;

    public RandomForest(int numTrees) {
        this.numTrees = numTrees;
        this.trees = new ArrayList<>();
    }

    public void train(double[][] X_train, int[] y_train, int numFeatures, int maxDepth) {
        Random random = new Random();

        for (int i = 0; i < numTrees; i++) {

            int[] sampledIndices = new int[X_train.length];
            for (int j = 0; j < X_train.length; j++) {
                sampledIndices[j] = random.nextInt(X_train.length);
            }

            // Random feature selection
            List<Integer> selectedFeatures = new ArrayList<>();
            for (int j = 0; j < numFeatures; j++) {
                selectedFeatures.add(j);
            }

            DecisionTree tree = new DecisionTree(maxDepth);
            tree.train(X_train, y_train);
            trees.add(tree);
        }
    }

    public int[] predict(double[][] X_test) {
        int[] predictions = new int[X_test.length];
        for (int i = 0; i < X_test.length; i++) {
            int[] treePredictions = new int[numTrees];
            for (int j = 0; j < numTrees; j++) {
                treePredictions[j] = trees.get(j).predict(X_test[i]);
            }
            predictions[i] = majorityVote(treePredictions);
        }
        return predictions;
    }

    private int majorityVote(int[] predictions) {
        // Simple majority voting
        int[] counts = new int[2];
        for (int pred : predictions) {
            counts[pred]++;
        }
        return counts[1] > counts[0] ? 1 : 0;
    }

    public static void main(String[] args) {
        // Beispielcode zur Verwendung des RandomForest-Modells
        double[][] X_train = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        int[] y_train = {0, 1, 0, 1};

        RandomForest forest = new RandomForest(10);
        forest.train(X_train, y_train, 2, 3);

        double[][] X_test = {{2, 3}, {6, 7}};
        int[] predictions = forest.predict(X_test);

        for (int pred : predictions) {
            System.out.println(pred);
        }
    }
}
