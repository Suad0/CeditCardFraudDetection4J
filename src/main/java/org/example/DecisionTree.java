package org.example;

import java.util.HashMap;
import java.util.Map;

public class DecisionTree {

    private Node root;
    private int maxDepth;

    public DecisionTree(int maxDepth) {
        this.maxDepth = maxDepth;

    }

    public void train(double[][] X_train, int[] y_train) {
        this.root = buildTree(X_train, y_train, 0);
    }

    public int predict(double[] instance) {
        return classify(instance, root);
    }

    private Node buildTree(double[][] X_train, int[] y_train, int depth) {
        if (depth >= maxDepth || allSame(y_train)) {
            return new Node(mostCommonClass(y_train));
        }

        Split bestSplit = findBestSplit(X_train, y_train);

        if (bestSplit == null) {
            return new Node(mostCommonClass(y_train));
        }

        Node leftChild = buildTree(bestSplit.leftX, bestSplit.leftY, depth + 1);
        Node rightChild = buildTree(bestSplit.rightX, bestSplit.rightY, depth + 1);

        return new Node(bestSplit.featureIndex, bestSplit.threshold, leftChild, rightChild);
    }

    private boolean allSame(int[] array) {
        int first = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] != first) {
                return false;
            }
        }
        return true;
    }

    private int mostCommonClass(int[] array) {
        Map<Integer, Integer> classCounts = new HashMap<>();
        for (int value : array) {
            classCounts.put(value, classCounts.getOrDefault(value, 0) + 1);
        }
        int mostCommonClass = -1;
        int maxCount = Integer.MIN_VALUE;
        for (Map.Entry<Integer, Integer> entry : classCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mostCommonClass = entry.getKey();
            }
        }
        return mostCommonClass;
    }


    private Split findBestSplit(double[][] X_train, int[] y_train) {
        int numFeatures = X_train[0].length;
        int numRows = X_train.length;
        double bestGini = Double.MAX_VALUE;
        int bestFeatureIndex = -1;
        double bestThreshold = 0;

        int[] leftY = new int[numRows];
        double[][] leftX = new double[numRows][numFeatures];
        int leftIndex = 0;

        int[] rightY = new int[numRows];
        double[][] rightX = new double[numRows][numFeatures];
        int rightIndex = 0;

        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
                double threshold = X_train[rowIndex][featureIndex];

                leftIndex = 0;
                rightIndex = 0;

                for (int i = 0; i < numRows; i++) {
                    if (X_train[i][featureIndex] <= threshold) {
                        leftY[leftIndex] = y_train[i];
                        System.arraycopy(X_train[i], 0, leftX[leftIndex], 0, numFeatures);
                        leftIndex++;
                    } else {
                        rightY[rightIndex] = y_train[i];
                        System.arraycopy(X_train[i], 0, rightX[rightIndex], 0, numFeatures);
                        rightIndex++;
                    }
                }

                double gini = calculateGini(leftY, rightY);
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeatureIndex = featureIndex;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeatureIndex == -1) {
            return null;
        }

        // Construct the Split object based on the best split found
        return new Split(bestFeatureIndex, bestThreshold, leftX, leftY, rightX, rightY);
    }


    private double calculateGini(int[]... groups) {
        int totalInstances = 0;
        for (int[] group : groups) {
            totalInstances += group.length;
        }

        double gini = 0;
        for (int[] group : groups) {
            double groupSize = group.length;
            if (groupSize == 0) {
                continue;
            }
            double score = 0;
            for (int classLabel : group) {
                double p = (double) countOccurrences(classLabel, group) / groupSize;
                score += p * p;
            }
            gini += (1.0 - score) * (groupSize / totalInstances);
        }
        return gini;
    }

    private int countOccurrences(int value, int[] array) {
        int count = 0;
        for (int v : array) {
            if (v == value) {
                count++;
            }
        }
        return count;
    }

    private int classify(double[] instance, Node node) {
        if (node.isLeaf()) {
            return node.classLabel;
        }

        if (instance[node.featureIndex] <= node.threshold) {
            return classify(instance, node.leftChild);
        } else {
            return classify(instance, node.rightChild);
        }
    }

    private static class Node {
        private int featureIndex;
        private double threshold;
        private int classLabel;
        private Node leftChild;
        private Node rightChild;

        public Node(int classLabel) {
            this.classLabel = classLabel;
        }

        public Node(int featureIndex, double threshold, Node leftChild, Node rightChild) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.leftChild = leftChild;
            this.rightChild = rightChild;
        }

        public boolean isLeaf() {
            return leftChild == null && rightChild == null;
        }
    }

    private static class Split {
        private int featureIndex;
        private double threshold;
        private double[][] leftX;
        private int[] leftY;
        private double[][] rightX;
        private int[] rightY;

        public Split(int featureIndex, double threshold, double[][] leftX, int[] leftY, double[][] rightX,
                     int[] rightY) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.leftX = leftX;
            this.leftY = leftY;
            this.rightX = rightX;
            this.rightY = rightY;
        }
    }

    public static void main(String[] args) {
        // Beispiel Daten
        double[][] X_train = {
                {1, 2},
                {2, 3},
                {3, 4},
                {4, 5},
                {5, 6},
                {6, 7},
                {7, 8},
                {8, 9}
        };

        int[] y_train = {0, 0, 1, 1, 0, 1, 0, 1}; // Dummy-Klassenlabels

        // Trainieren des Entscheidungsbaums
        DecisionTree decisionTree = new DecisionTree(3); // Max. Tiefe: 3
        decisionTree.train(X_train, y_train);

        // Testdaten
        double[][] X_test = {
                {1, 2},
                {5, 6},
                {8, 9}
        };

        // Vorhersagen treffen
        for (double[] instance : X_test) {
            int prediction = decisionTree.predict(instance);
            System.out.println("Vorhersage für " + java.util.Arrays.toString(instance) + ": " + prediction);
        }
    }
}
