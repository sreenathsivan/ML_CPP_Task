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
#include <algorithm>
#include <cstdlib>
using namespace std;
class LinearLayer
{
public:
    int inputSize = 0;
    int outputSize = 0;
    float learningRate = 0.001f; // Default learning rate
    vector<vector<float>> weights;
    vector<float> biases;
    vector<vector<float>> lastInput; // Store input for backpropagation
    vector<vector<float>> outputs;
    vector<vector<float>> dw;
    vector<vector<float>> dz;

    LinearLayer(int inputSize, int outputSize)
    {
        this->inputSize = inputSize;
        this->outputSize = outputSize;
        this->initializeParameters();
    }
    vector<vector<float>> forward(const vector<vector<float>> inputs)
    {

        this->lastInput = inputs;
        assert(!inputs.empty() && inputs[0].size() == inputSize);
        this->outputs.resize(inputs.size(), vector<float>(this->outputSize, 0.0f)); // 614X12   //614X1
        // cout << "Output matrix size" << size(this->outputs) << "X" << size(this->outputs[0]) << endl;
        // cout << "Input matrix size" << size(inputs) << "X" << size(inputs[0]) << endl;
        // cout << "weight matrix size" << size(this->weights) << "X" << size(this->weights[0]) << endl;
        // cout << "biase matrix size" << size(this->biases) << endl;
        // exit(0);
        vector<vector<float>> biase_broadcast(this->outputs.size(), vector<float>(this->biases.size()));
        // cout << "biase_broadcast matrix size" << size(biase_broadcast) << "X" << size(biase_broadcast[0]) << endl;

        for (int i = 0; i < size(inputs); i++)
        {
            for (int j = 0; j < this->weights[0].size(); j++)
            {
                this->outputs[i][j] = 0;
                for (int k = 0; k < this->weights.size(); k++)
                {
                    this->outputs[i][j] += inputs[i][k] * this->weights[k][j];
                }
            }
        }

        for (int i = 0; i < size(biase_broadcast); i++)
        {
            for (int j = 0; j < biase_broadcast[0].size(); j++)
            {
                biase_broadcast[i][j] = this->biases[i];
            }
        }
 
        for (int i = 0; i < size(this->outputs); i++)
        {
            for (int j = 0; j < this->outputs[0].size(); j++)
            {
                this->outputs[i][j] += biase_broadcast[i][j];
            }
        }
        // this->outputs=matmul(inputs,this->weights);
        return this->outputs;
    }
    vector<vector<float>> backward(double loss, vector<vector<float>> y_train, vector<vector<float>> y_pred)
    {

        auto dz2 = loss_gradient(y_train, y_pred); //(y^-y) error of the output layer
        // cout << size(dz2) << endl;
        // cout << "shape of grad out " << dz2.size() << "X" << dz2[0].size() << endl;
        // exit(0);

        auto inputTranspose = transpose(this->lastInput);
        // cout << "shape of grad out " << this->lastInput.size() << "X" << this->lastInput[0].size() << endl;

        // Calculate the gradient of the loss with respect to weight (dL/dw)
        vector<vector<float>> dw2(inputTranspose.size(), vector<float>(dz2[0].size()));

        for (int i = 0; i < size(inputTranspose); i++)
        {
            for (int j = 0; j < dw2[0].size(); j++)
            {
                for (int k = 0; k < dz2.size(); k++)
                {
                    dw2[i][j] = inputTranspose[i][k] * dz2[k][j];
                }
                // cout << dw2[i][j] << "\t";
            }

            // cout << endl;
        }
        this->dw = dw2;
        this->dz = dz2;
        // cout << "shape of this->weights " << this->weights.size() << "X" << this->weights[0].size() << endl;
        // exit(0);

        return dz2;
    }
    void update_weight_and_biase()
    {
        // weight w1 updation
        //  Update weights and biases for the output layer

        


        for (size_t i = 0; i < this->weights.size(); ++i)
        {
            for (size_t j = 0; j < this->weights[i].size(); ++j)
            {
                // std::cout << "dw: " << dw[i][j] << std::endl;
                this->weights[i][j] -= this->learningRate * this->dw[i][j]; // Gradient descent update
                // this->weights[i][j] = 0;
            }
        }

        // update biase
        // cout << "shape of dz2 " << this->dz.size() << "X" << this->dz[0].size() << endl;
        auto columnsum_dz2 = columnWiseSum(this->dz);
        auto db2 = columnsum_dz2;

        for (int i = 0; i < size(this->biases); i++)
        {
            this->biases[i] -= this->learningRate * db2[i];
        }

        // weight2 updation

        // weight updation
        // for (size_t i = 0; i < this->weights.size(); ++i)
        // {
        //     for (size_t j = 0; j < this->weights[i].size(); ++j)
        //     {
        //         this->weights[i][j] -= this->learningRate * dw1[i][j]; // Gradient descent update
        //     }
        // }
        // auto columnsum_dz1 = columnWiseSum(a);
        // auto db1 = columnsum_dz1;

        // for (int i = 0; i < size(this->biases); i++)
        // {
        //     this->biases[i] -= this->learningRate * db1[i];
        // }
    }

    vector<vector<float>> elemWiseMul(vector<vector<float>> mat1, vector<vector<float>> mat2)
    {
        vector<vector<float>> output(mat1.size(), vector<float>(mat1[0].size()));
        for (int i = 0; i < size(mat1); i++)
        {
            for (int j = 0; j < size(mat1[0]); j++)
            {
                output[i][j] = mat1[i][j] * mat2[i][j];
                // output[i][j]=1;
            }
        }
        return output;
    }

    vector<float> columnWiseSum(const vector<vector<float>> &matrix)
    {
        int rows = matrix.size();
        int cols = matrix[0].size();
        vector<float> result(cols, 0); // Initialize a vector of size 'cols' with 0s

        // Loop through each column
        for (int j = 0; j < cols; ++j)
        {
            // Sum up each column
            for (int i = 0; i < rows; ++i)
            {
                result[j] += matrix[i][j];
            }
        }
        return result;
    }

    void backward1(vector<vector<float>> dz2, vector<vector<float>> output_weights, vector<vector<float>> z1)
    {
        // dz2 is the error of the output layer
        // z1 is pre activation of the current layer ,ie, z1=w.x^T
        //  grad_output is the error from the output layer

        // For hidden layers, compute the error term
        // δ (l) =((W (l+1) ) T δ (l+1) )⊙σ ′ (z (l) )

        auto a = matmul(dz2, transpose(output_weights)); //((W (l+1) ) T δ (l+1) )
        // a is the error of the first hidden layer

        //  cout << "shape of a " << a.size() << "X" << a[0].size() << endl;
        //  exit(0);

        // Apply the ReLU derivative (gradient of ReLU)

        vector<vector<float>> grad_relu_input(z1.size(), vector<float>(z1[0].size())); // σ ′ (z (l)
        for (int i = 0; i < z1.size(); i++)
        {
            for (int j = 0; j < z1[0].size(); j++)
            {
                grad_relu_input[i][j] = (z1[i][j] > 0 ? 1.0f : 0.0f);
            }
        }

        // auto grad_input = matmul(a, grad_relu_input);
        // Calculate the gradient of the loss with respect to weight (dL/dw1)
        auto dz1 = elemWiseMul(a, grad_relu_input);
        this->dz = dz1;
        auto dw1 = matmul(transpose(this->lastInput), dz1); // δ (l)*a^(l-1)^T
        this->dw = dw1;
    }

    void initializeParameters()
    {
        // cout << "initializeParameters" << endl;
        this->weights.resize(this->inputSize, vector<float>(this->outputSize));
        this->biases.resize(this->outputSize);
        // cout << "biase matrix size" << size(this->biases) << endl;

        for (int i = 0; i < this->inputSize; i++)
        {
            for (int j = 0; j < this->outputSize; j++)
            {
                this->weights[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.0001f; // Small random weights
                // cout << this->weights[i][j] << "\t";
            }

            // cout << "\n";
        }
        // cout << "weight matrix size" << size(this->weights) << "X" << size(this->weights[0]) << endl;
        for (int i = 0; i < (this->outputSize); i++)
        {
            this->biases[i] = 0.0f; // Initialize biases to zero
        }
    }

    // Matrix multiplication (A * B)
    vector<vector<float>> matmul(const vector<vector<float>> &A, const vector<vector<float>> &B)
    {
        // Assert that the number of columns in A is equal to the number of rows in B
        assert(A[0].size() == B.size() && "Matrix dimensions do not match for multiplication.");

        // Initialize the result matrix C with the correct dimensions
        vector<vector<float>> C(A.size(), vector<float>(B[0].size(), 0.0f));

        // Perform matrix multiplication
        for (size_t i = 0; i < A.size(); ++i) // Iterate over rows of A
        {
            for (size_t j = 0; j < B[0].size(); ++j) // Iterate over columns of B
            {
                for (size_t k = 0; k < A[0].size(); ++k) // Iterate over columns of A and rows of B
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }

    vector<vector<float>> transpose(vector<vector<float>> codes)
    {
        vector<vector<float>> vect1(size(codes[0]), std::vector<float>(size(codes), 0));

        for (int i = 0; i < size(codes); i++)
        {

            for (int j = 0; j < size(codes[0]); j++)
            {
                vect1[j][i] = codes[i][j];
            }
        }
        return vect1;
    }
    vector<vector<float>> loss_gradient(const vector<vector<float>> y_true, const vector<vector<float>> y_pred)
    {
        vector<vector<float>> grad(y_true.size(), vector<float>(1));

        for (size_t i = 0; i < y_true.size(); ++i)
        {
            grad[i][0] = y_pred[i][0] - y_true[i][0]; // Gradient: y_pred - y_true
        }

        return grad;
    }
    // Sigmoid derivative function (for backpropagation)
    std::vector<float> sigmoid_derivative(const vector<vector<float>> &output)
    {
        std::vector<float> derivative(output.size());
        for (size_t i = 0; i < output.size(); ++i)
        {
            derivative[i] = output[i][0] * (1 - output[i][0]); // Sigmoid derivative
        }
        return derivative;
    }
};
// ReLU activation function
vector<vector<float>> relu(const vector<vector<float>> input)
{
    vector<vector<float>> output = input;
    for (size_t i = 0; i < output.size(); ++i)
    {
        for (size_t j = 0; j < output[i].size(); ++j)
        {
            output[i][j] = max(0.0f, output[i][j]); // Apply ReLU
        }
    }
    return output;
}
// Function to calculate binary cross-entropy loss
double binaryCrossEntropy(const vector<vector<float>> y_true, const vector<vector<float>> y_pred)
{
    double loss = 0.0;
    size_t size = y_true.size();
    double epsilon = 0;

    for (size_t i = 0; i < size; i++)
    {
        if (y_pred[i][0] == 0)
        {
            epsilon = 1e-10;
        }
        if (1 - y_pred[i][0] == 0)
        {
            epsilon = 1e-10;
        }

        loss += y_true[i][0] * log(y_pred[i][0] + epsilon) + (1 - y_true[i][0]) * log(1 - y_pred[i][0] + epsilon);
    }

    return -loss / size;
}
// Sigmoid activation function
vector<float> sigmoid(const vector<float> input)
{
    vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i] = 1.0f / (1.0f + exp(-input[i])); // Apply sigmoid
    }
    return output;
}

int main()
{
    // // Define a network with 8 input neurons, 2 hidden layers (3 and 2 neurons), and 1 output neuron
    // NeuralNetwork nn({8, 3, 3, 1});

    process<float> p;
    string path = "../pima-indians-diabetes.csv";
    auto data = p.CSV2VEC(path); // Call the CSV2VEC method on the instance
    auto X_raw = p.vector2DSlice(data, 0, size(data), 0, 7);
    auto Y = p.vector2DSlice(data, 0, size(data), 8, 8);
    // auto X=X_raw;
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

    // vector<vector<float>> X_val;
    // vector<vector<float>> y_val;
    // vector<vector<float>> X_train_new;
    // vector<vector<float>> y_train_new;

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

    // Now split the train + validation set into train and validation sets (80% train, 20% validation)
    // const float trainRatio = 0.8;
    // const int dataSize = size(X_train);
    // const int trainSize = static_cast<int>(trainRatio * dataSize);
    // const int valSize = dataSize - trainSize;

    // std::vector<int> indices(dataSize);
    // std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., dataSize-1
    // std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    // std::vector<int> trainIndices(indices.begin(), indices.begin() + trainSize);
    // std::vector<int> valIndices(indices.begin() + trainSize, indices.end());

    // for (int i = 0; i < size(trainIndices); i++)
    // {
    //     X_train_new.push_back(X_train[i]);
    //     y_train_new.push_back(X_train[i]);
    // }
    // for (int i = 0; i < size(valIndices); i++)
    // {
    //     X_val.push_back(X_train[i]);
    //     y_val.push_back(X_train[i]);
    // }

    // X_train=X_train_new;
    // y_train=y_train_new;

    int epoch = 200;
    int batch_size = 32;

    // train

    LinearLayer input(8, 12);
    LinearLayer output_layer(12, 1);
    std::vector<float> loss_values;
    std::vector<float> train_loss;
    float initial_learning_rate = 0.001f;
float decay_rate = 0.9f;  // Decay rate (e.g., 0.9 means reduce by 10% each time)
int decay_step = 50;  

    for (int i = 0; i < epoch; i++)
    {
    //        if (epoch % decay_step == 0 && epoch > 0) {
    //     input.learningRate *= decay_rate;  // Apply decay to input layer
    //     output_layer.learningRate *= decay_rate;  // Apply decay to output layer
    // }

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

            Xbatch = p.vector2DSlice(X_train, start, end, 0, size(X_train[0]) - 1);

            auto ybatch = p.vector2DSlice(y_train, start, end, 0, size(y_train[0])-1 );
            auto input_for_layer2 = input.forward(Xbatch);
            auto activated_output = relu(input_for_layer2); // Apply ReLU
            auto output = output_layer.forward(activated_output);
            vector<vector<float>> y_pred;
            for (const auto &out : output)
            {
                y_pred.push_back(sigmoid(out));
            }
            auto actual_output = ybatch;
            auto loss = binaryCrossEntropy(actual_output,(y_pred));
            // cout << loss << endl;
            loss_values.push_back(loss);
            // Backward pass
            auto W1_grad = output_layer.backward(loss, ybatch, y_pred);
            input.backward1(W1_grad, output_layer.weights, input_for_layer2);
            output_layer.update_weight_and_biase();
            input.update_weight_and_biase();

            // cout << "Batch start: " << start << ", end: " << end << endl;
            start = end ;
        }
        train_loss.push_back(loss_values[size(loss_values) - 1]);
        cout << "epoch " << i + 1 << "epoch loss " << loss_values[size(loss_values) - 1] << endl;
    }

        // Evaluate final model on the test set
    vector<vector<float>> X_test_pred = input.forward(X_test);
    auto activated_test_output = relu(X_test_pred);
    auto test_output = output_layer.forward(activated_test_output);
    vector<vector<float>> test_pred;
    for (const auto &out : test_output)
    {
        test_pred.push_back(sigmoid(out));
    }
    auto test_loss = binaryCrossEntropy(y_test, test_pred);
    cout << "Test Loss: " << test_loss << endl;
   

    FILE *gnuplot = popen("gnuplot -persistent", "w");

    if (gnuplot)
    {
        // Send gnuplot commands to plot the array directly
        fprintf(gnuplot, "plot '-' with lines\n");

        // Send data points directly to gnuplot
        for (size_t i = 0; i < train_loss.size(); ++i)
        {
            fprintf(gnuplot, "%zu %f\n", i + 1, train_loss[i]); // Plot index vs value
        }

        // End the data input
        fprintf(gnuplot, "e\n");

        // Close the pipe to gnuplot
        fclose(gnuplot);
    }
    else
    {
        std::cerr << "Error: Couldn't open gnuplot." << std::endl;
    }

    // exit(0);

    // for (int i = 0; i < 1000; i++)
    // {
    //     auto input_for_layer2 = input.forward(X_train);
    //     auto activated_output = relu(input_for_layer2); // Apply ReLU
    //     auto output = output_layer.forward(activated_output);
    //     vector<vector<float>> y_pred;
    //     for (const auto &out : output)
    //     {
    //         y_pred.push_back(sigmoid(out));
    //     }
    //     auto actual_output = y_train;
    //     auto loss = binaryCrossEntropy(actual_output, y_pred);
    //     cout << loss << endl;
    //     loss_values.push_back(loss);
    //     // Backward pass
    //     auto W1_grad = output_layer.backward(loss, y_train, y_pred);
    //     input.backward1(W1_grad, output_layer.weights, input_for_layer2);
    //     output_layer.update_weight_and_biase();
    //     input.update_weight_and_biase();
    // }
    // for (int j = 0; j < size(loss_values); j++)
    // {
    //     cout << loss_values[j] << "\t";
    // }
    // cout << "\n";

    return 0; // Return 0 to indicate succestsful execution
}