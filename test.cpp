#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <tuple>
#include <bits/stdc++.h>
#include "process.h"  // Assuming this file exists and provides utility functions
using namespace std;

class LinearLayer
{
public:
    int inputSize;
    int outputSize;
    float learningRate = 0.01f; // Default learning rate
    vector<vector<float>> weights;
    vector<float> biases;
    vector<vector<float>> lastInput; // Store input for backpropagation
    vector<vector<float>> outputs;

    LinearLayer(int inputSize, int outputSize)
    {
        this->inputSize = inputSize;
        this->outputSize = outputSize;
        this->initializeParameters();
    }

    void initializeParameters()
    {
        this->weights.resize(inputSize, vector<float>(outputSize));
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                this->weights[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.01f; // Small random weights
            }
        }
        this->biases.resize(outputSize, 0.0f); // Initialize biases to zero
    }

    vector<vector<float>> forward(const vector<vector<float>> &inputs)
    {
        this->lastInput = inputs;

        assert(!inputs.empty() && inputs[0].size() == inputSize);
        outputs.resize(inputs.size(), vector<float>(outputSize, 0.0f));

        for (size_t i = 0; i < inputs.size(); i++)
        {
            for (size_t j = 0; j < outputSize; j++)
            {
                for (size_t k = 0; k < inputSize; k++)
                {
                    outputs[i][j] += inputs[i][k] * weights[k][j];
                }
                outputs[i][j] += biases[j];  // Adding bias
            }
        }

        return outputs;
    }

    vector<vector<float>> backward(const vector<vector<float>> &y_true, const vector<vector<float>> &y_pred, double loss)
    {
        // Calculate the gradient of the loss with respect to predictions (dL/dy_pred)
        auto grad_loss = loss_gradient(y_true, y_pred);

        // Derivative of sigmoid
        auto grad_sigmoid = sigmoid_derivative(y_pred);

        vector<vector<float>> grad_output(y_pred.size(), vector<float>(y_pred[0].size()));
        for (size_t i = 0; i < grad_loss.size(); ++i)
        {
            grad_output[i][0] = grad_loss[i][0] * grad_sigmoid[i]; // Chain rule
        }

        auto inputTranspose = transpose(lastInput);

        // Compute the gradient of the loss with respect to the weights and biases
        vector<vector<float>> weight_grad(inputTranspose.size(), vector<float>(grad_output[0].size(), 0.0f));
        for (size_t i = 0; i < inputTranspose.size(); ++i)
        {
            for (size_t j = 0; j < grad_output[0].size(); ++j)
            {
                for (size_t k = 0; k < grad_output.size(); ++k)
                {
                    weight_grad[i][j] += inputTranspose[i][k] * grad_output[k][j];
                }
            }
        }

        // Update weights and biases
        for (size_t i = 0; i < weights.size(); ++i)
        {
            for (size_t j = 0; j < weights[i].size(); ++j)
            {
                weights[i][j] -= learningRate * weight_grad[i][j]; // Gradient descent update
            }
        }

        // Update biases
        for (size_t i = 0; i < biases.size(); ++i)
        {
            biases[i] -= learningRate * grad_output[i][0]; // Gradient descent update for bias
        }

        return grad_output;
    }

    // Matrix multiplication (A * B)
    vector<vector<float>> matmul(const vector<vector<float>> &A, const vector<vector<float>> &B)
    {
        vector<vector<float>> C(A.size(), vector<float>(B[0].size(), 0.0f));
        for (size_t i = 0; i < A.size(); ++i)
        {
            for (size_t j = 0; j < B[0].size(); ++j)
            {
                for (size_t k = 0; k < A[0].size(); ++k)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    vector<vector<float>> transpose(const vector<vector<float>> &mat)
    {
        vector<vector<float>> result(mat[0].size(), vector<float>(mat.size()));
        for (size_t i = 0; i < mat.size(); i++)
        {
            for (size_t j = 0; j < mat[i].size(); j++)
            {
                result[j][i] = mat[i][j];
            }
        }
        return result;
    }

    vector<vector<float>> loss_gradient(const vector<vector<float>> &y_true, const vector<vector<float>> &y_pred)
    {
        vector<vector<float>> grad(y_true.size(), vector<float>(1));
        for (size_t i = 0; i < y_true.size(); ++i)
        {
            grad[i][0] = y_pred[i][0] - y_true[i][0]; // Gradient: y_pred - y_true
        }
        return grad;
    }

    // Sigmoid derivative function (for backpropagation)
    vector<float> sigmoid_derivative(const vector<vector<float>> &output)
    {
        vector<float> derivative(output.size());
        for (size_t i = 0; i < output.size(); ++i)
        {
            derivative[i] = output[i][0] * (1 - output[i][0]); // Sigmoid derivative
        }
        return derivative;
    }
};

// Sigmoid activation function
vector<float> sigmoid(const vector<float> &input)
{
    vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i] = 1.0f / (1.0f + exp(-input[i])); // Apply sigmoid
    }
    return output;
}

// ReLU activation function
vector<vector<float>> relu(const vector<vector<float>> &input)
{
    vector<vector<float>> output = input;
    for (size_t i = 0; i < input.size(); ++i)
    {
        for (size_t j = 0; j < input[i].size(); ++j)
        {
            output[i][j] = max(0.0f, input[i][j]); // Apply ReLU
        }
    }
    return output;
}

// Function to calculate binary cross-entropy loss
double binaryCrossEntropy(const vector<vector<float>> &y_true, const vector<vector<float>> &y_pred)
{
    double loss = 0.0;
    size_t size = y_true.size();
    for (size_t i = 0; i < size; i++)
    {
        loss += y_true[i][0] * log(y_pred[i][0]) + (1 - y_true[i][0]) * log(1 - y_pred[i][0]);
    }
    return -loss / size;
}

int main()
{
    process<int> p;
    string path = "../pima-indians-diabetes.csv";
    auto data = p.CSV2VEC(path); // Assuming CSV2VEC method
    auto X_raw = p.vector2DSlice(data, 0, data.size(), 0, 8); // Adjusted column size
    auto Y = p.vector2DSlice(data, 0, data.size(), 8, 9); // Assuming column 8 for Y
    auto X = p.fit_transform(X_raw);

    const float trainRatio = 0.8;
    const int dataSize = X.size();
    const int trainSize = static_cast<int>(trainRatio * dataSize);
    const int testSize = dataSize - trainSize;

    vector<int> indices(dataSize);
    iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., dataSize-1
    shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    vector<vector<float>> X_train, y_train, X_test, y_test;
    for (int i = 0; i < trainSize; ++i)
    {
        X_train.push_back(X[indices[i]]);
        y_train.push_back(Y[indices[i]]);
    }
    for (int i = trainSize; i < dataSize; ++i)
    {
        X_test.push_back(X[indices[i]]);
        y_test.push_back(Y[indices[i]]);
    }

    LinearLayer layer1(8, 12); // Layer 1: 8 inputs -> 12 neurons
    LinearLayer layer2(12, 1); // Layer 2: 12 neurons -> 1 output

    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch)
    {
        auto y_pred_train = layer1.forward(X_train);
        y_pred_train = relu(y_pred_train); // Apply ReLU after layer 1
        y_pred_train = layer2.forward(y_pred_train);
        y_pred_train = sigmoid(y_pred_train); // Apply sigmoid on output

        double loss = binaryCrossEntropy(y_train, y_pred_train);
        cout << "Epoch " << epoch << ", Loss: " << loss << endl;

        // Backpropagation
        layer2.backward(y_train, y_pred_train, loss);
        layer1.backward(y_train, y_pred_train, loss);
    }

    return 0;
}
