#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <tuple>
#include <bits/stdc++.h>

#include "process.h"
using namespace std;

class Model
{
private:
    LinearLayer ff;
    Sigmoid sig;
    vector<double> output_vals;

public:
    Model();
    vector<double> feedForward(vector<double> input);
    void backPropagate(vector<double> exp_vals);
};
Model::Model()
{
    ff = LinearLayer(2, 1, 0.15); // linear layer with input size 2, output size 1, and learning rate 0.15
    sig = Sigmoid();
}
vector<double> Model::feedForward(vector<double> input)
{
    input = ff.feedForward(input);
    output_vals = sig.feedForward(input);
    return output_vals;
}
void Model::backPropagate(vector<double> exp_vals)
{
    vector<double> grad;
    assert(exp_vals.size() == output_vals.size());

    // compute derivative of error with respect to network's output
    for (int out = 0; out < exp_vals.size(); out++)
    {
        grad.push_back((output_vals[out] - exp_vals[out]));
    }

    grad = sig.backPropagate(grad);
    ff.backPropagate(grad);
}
class Sigmoid
{
private:
    vector<double> output_vals;

public:
    Sigmoid() {}
    vector<double> feedForward(const vector<double> &input);
    vector<double> backPropagate(vector<double> grad);
};
vector<double> Sigmoid::feedForward(const vector<double> &input)
{
    output_vals = vector<double>();
    for (int in = 0; in < input.size(); in++)
    {
        output_vals.push_back(1.0 / (1.0 + exp(-input[in])));
    }
    return output_vals;
}
vector<double> Sigmoid::backPropagate(vector<double> grad)
{
    assert(grad.size() == output_vals.size());
    for (int out = 0; out < output_vals.size(); out++)
    {
        grad[out] *= output_vals[out] * (1.0 - output_vals[out]);
    }
    return grad;
}
class LinearLayer
{
private:
    vector<vector<double>> weights;
    vector<double> output_vals;
    vector<double> input_vals;
    int output_dim;
    int input_dim;
    double eta;

public:
    LinearLayer() {}
    LinearLayer(int input_size, int output_size, double lr);
    vector<double> feedForward(const vector<double> &input);
    vector<double> backPropagate(const vector<double> &grad);
};
LinearLayer::LinearLayer(int input_size, int output_size, double lr)
{
    assert(input_size > 0);
    assert(output_size > 0);
    output_dim = output_size;
    input_dim = input_size;
    eta = lr;

    // generate random weights
    for (int out = 0; out < output_size; out++)
    {
        weights.push_back(vector<double>());
        for (int input = 0; input < input_size + 1; input++)
        {                                                        // we create an extra weight (one more than input_size) for our bias
            weights.back().push_back((double)rand() / RAND_MAX); // random value between 0 and 1
        }
    }
}
vector<double> LinearLayer::feedForward(const vector<double> &input)
{
    assert(input.size() == input_dim);
    output_vals = vector<double>();
    input_vals = input; // store the input vector

    // perform matrix multiplication
    for (int out = 0; out < output_dim; out++)
    {
        double sum = 0.0;
        for (int w = 0; w < input_dim; w++)
        {
            sum += weights[out][w] * input[w];
        }
        sum += weights[out][input_dim]; // account for the bias
        output_vals.push_back(sum);
    }
    return output_vals;
}
vector<double> LinearLayer::backPropagate(const vector<double> &grad)
{
    assert(grad.size() == output_dim);
    vector<double> prev_layer_grad;

    // calculate partial derivatives with respect to input values
    for (int input = 0; input < input_dim; input++)
    {
        double g = 0.0;
        for (int out = 0; out < output_dim; out++)
        {
            g += (grad[out] * weights[out][input]);
        }
        prev_layer_grad.push_back(g);
    }

    // change weights using gradient
    for (int out = 0; out < output_dim; out++)
    {
        for (int input = 0; input < input_dim; input++)
        {
            weights[out][input] -= (eta * grad[out] * input_vals[input]);
        }
        weights[out][input_dim] -= eta * grad[out];
    }

    // return computed partial derivatives to be passed to preceding layer
    return prev_layer_grad;
}

int main()
{
    process<int> p;
    string path = "../pima-indians-diabetes.csv";
    auto data = p.CSV2VEC(path); // Call the CSV2VEC method on the instance
    auto X_raw = p.vector2DSlice(data, 0, size(data), 0, 7);
    auto Y = p.vector2DSlice(data, 0, size(data), 8, 8);
    auto X = p.fit_transform(X_raw);
    const float trainRatio = 0.8;
    const int dataSize = size(X);

    const int trainSize = static_cast<int>(trainRatio * dataSize);
    const int testSize = dataSize - trainSize;
    std::vector<int> indices(dataSize);
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., dataSize-1
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    std::vector<int> trainIndices(indices.begin(), indices.begin() + trainSize);
    std::vector<int> testIndices(indices.begin() + trainSize, indices.end());
    vector<vector<float>> X_train;
    vector<vector<float>> y_train;
    vector<vector<float>> X_test;
    vector<vector<float>> y_test;

    for (int i = 0; i < size(trainIndices); i++)
    {
        X_train.push_back(X[i]);
        y_train.push_back(Y[i]);
    }
    for (int i = 0; i < size(testIndices); i++)
    {
        X_test.push_back(X[i]);
        y_test.push_back(Y[i]);
    }
    int epoch = 200;
    int batch_size = 32;

    // train
    for (int i = 0; i < epoch; i++)
    {
        int start = 0;
        int end = batch_size;
        for (int j = 0; j < size(X_train); j += batch_size)
        {

            int end = j + batch_size;
            if (end > size(X_train))
            {
                end = size(X_train);
            }

            vector<vector<float>> Xbatch;
            Xbatch = p.vector2DSlice(X_train, start, end, 0, size(X_train[0]));
            auto ybatch = p.vector2DSlice(y_train, start, end, 0, size(y_train[0]));
            

            cout << "Batch start: " << start << ", end: " << end << endl;
            start = end + 1;
        }

        cout << "batch " << i + 1 << endl;
    }

    return 0; // Return 0 to indicate successful execution
}
