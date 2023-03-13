// Author: Patrick Moehrke
// License: MIT

import std.math;
import std.file;
import std.random;
import std.stdio;
import std.array;
import std.conv;
import std.csv;
import std.algorithm;
import std.string;

struct Neuron {
    float[] weights;
    float bias;
}

struct Data {
    float[][] X;
    int[] y;
}

Data process_data(float[][] data) {
    Data df;
    int rows = data.length.to!int;
    int cols = data[0].length.to!int;
    df.X = new float[][](rows, cols-1);
    df.y = new int[](rows);

    foreach (i; 0..rows) {
        foreach (j; 0..cols) {
            if (j == cols-1) {
                df.y[i] = data[i][j].to!int;
	    } else {
                df.X[i][j] = log_transform(data[i][j].to!float);
	    }
	}
    }
    return(df);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

int transform_output(float x) {
    if(x > 0.5) return 1;
    return(0);
}

float dot_product(float[] a, float[] b) {
    float sum = 0.0;
    foreach (i; 0..a.length) {
        sum += a[i] * b[i];
    }
    return sum;
}

float feedforward(float[] inputs, Neuron neuron) {
    float net = dot_product(inputs, neuron.weights) + neuron.bias;
    float output = sigmoid(net);
    return output;
}

float log_transform(float x) {
    return(log(x+1.0f));
}

float[][] read_csv(string filename) {
    auto file_data = readText(filename);
    auto csv_data = file_data.chomp.splitter('\n').map!(a => a.splitter(',').array).array;
    float[][] data_float = csv_data.map!(a => a.map!(b => b.canFind('.') ? b.to!float : b == "bad\r" ? 0.0f : 1.0f).array).array;
    return data_float;
}

Neuron train(float[][] X, int[] y, float learning_rate, int n_epochs) {
    int rows = X.length.to!int;
    int n_features = X[0].length.to!int;
    Neuron neuron;
    neuron.weights = new float[](n_features);
    foreach (i; 0..n_features) {
        neuron.weights[i] = uniform(-1.0, 1.0);
    }
    neuron.bias = uniform(-1.0, 1.0);

    foreach (epoch; 0..n_epochs) {
        foreach (i; 0..rows) {
            float output = feedforward(X[i], neuron);
            float error = y[i] - output;
            foreach (j; 0..neuron.weights.length) {
                neuron.weights[j] += learning_rate * error * output * (1 - output) * X[i][j];
            }
            neuron.bias += learning_rate * error * output * (1 - output);
        }
    }
    return(neuron);
}

int[] predict(float[][] X, Neuron neuron) {
    int rows = X.length.to!int;
    int[] pred = new int[](rows);
    foreach(i; 0..rows){
        pred[i] = transform_output(feedforward(X[i], neuron));
    }
    return(pred);
}

int[][] confusion_matrix(int[] actual, int[] predicted) {
    assert(actual.length == predicted.length);
    int[][] cm = [
        [0, 0],
        [0, 0],
    ];
    foreach (i; 0..actual.length) {
	if (actual[i] == predicted[i]) {
	    if (actual[i].to!int == 0) {
                cm[0][0] += 1;
	    } else {
                cm[1][1] += 1;
	    }
	} else {
	    if (actual[i].to!int == 0) {
                cm[1][0] += 1;
	    } else {
                cm[0][1] += 1;
	    }
	}
    }
    return(cm);
}

void main() {
    // Define the input data
    float[][] raw_data = read_csv("../data/wine.csv");

    // Extract the input data and expected output
    // TODO: split data into training and testing
    Data data = process_data(raw_data);

    // Train the model
    Neuron model = train(data.X, data.y, 0.1, 1000);

    // Print the final neuron weights and bias
    writeln("Final weights: ", model.weights);
    writeln("Final bias: ", model.bias);

    // Generate predictions
    int[] predicted_output = predict(data.X, model);

    // Display the confusion matrix
    writeln("Confusion Matrix: ");
    confusion_matrix(data.y, predicted_output).writeln;
}

