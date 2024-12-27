#include <iostream>
#include <vector>
#include <cmath>

// Sigmoid function
std::vector<float> sigmoid(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i])); // Apply sigmoid
    }
    return output;
}

// Sigmoid derivative function (for backpropagation)
std::vector<float> sigmoid_derivative(const std::vector<float>& output) {
    std::vector<float> derivative(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        derivative[i] = output[i] * (1 - output[i]); // Sigmoid derivative
    }
    return derivative;
}

// Binary cross-entropy loss function
double binaryCrossEntropy(const std::vector<float>& y_true, const std::vector<float>& y_pred) {
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        loss += y_true[i] * std::log(y_pred[i]) + (1 - y_true[i]) * std::log(1 - y_pred[i]);
    }
    return -loss / y_true.size(); // Average loss
}

// Calculate the gradient of the loss with respect to predictions (dL/dy_pred)
std::vector<float> loss_gradient(const std::vector<float>& y_true, const std::vector<float>& y_pred) {
    std::vector<float> grad(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        grad[i] = y_pred[i] - y_true[i]; // Gradient: y_pred - y_true
    }
    return grad;
}

// Forward pass function: computes the output of a layer given the input
std::vector<float> forward(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases) {
    std::vector<float> output(weights[0].size(), 0.0f);
    for (size_t i = 0; i < weights[0].size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            output[i] += input[j] * weights[j][i];
        }
        output[i] += biases[i]; // Add bias term
    }
    return output;
}

// Backpropagation function: updates weights and biases
void backpropagate(
    const std::vector<float>& input,
    const std::vector<float>& y_true,
    const std::vector<float>& y_pred,
    std::vector<std::vector<float>>& weights,
    std::vector<float>& biases,
    float learning_rate
) {
    // Compute the gradient of the loss with respect to the predictions (dL/dy_pred)
    std::vector<float> grad_loss = loss_gradient(y_true, y_pred);

    // Compute the gradient of the loss with respect to the weights and biases
    std::vector<float> grad_sigmoid = sigmoid_derivative(y_pred); // Derivative of sigmoid
    std::vector<float> grad_output(y_pred.size());
    
    for (size_t i = 0; i < grad_loss.size(); ++i) {
        grad_output[i] = grad_loss[i] * grad_sigmoid[i]; // Chain rule
    }

    // Update weights and biases for the output layer
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learning_rate * grad_output[j] * input[i]; // Gradient descent update
        }
    }

    // Update biases for the output layer
    for (size_t j = 0; j < biases.size(); ++j) {
        biases[j] -= learning_rate * grad_output[j]; // Gradient descent update
    }
}

int main() {
    // Sample input data (2 features, 1 sample)
    std::vector<float> input = {0.5f, 0.8f};
    std::vector<float> y_true = {1}; // True label for binary classification

    // Initialize weights (input layer -> output layer) and biases
    std::vector<std::vector<float>> weights = {
        {0.1f, 0.2f}, // Weights for input 1
        {0.3f, 0.4f}  // Weights for input 2
    };
    std::vector<float> biases = {0.5f, 0.6f}; // Biases for each output neuron

    float learning_rate = 0.01f;

    // Forward pass: Get predicted output
    std::vector<float> output = forward(input, weights, biases);
    std::vector<float> y_pred = sigmoid(output); // Apply sigmoid

    // Compute the loss (binary cross-entropy)
    double loss = binaryCrossEntropy(y_true, y_pred);
    std::cout << "Initial Loss: " << loss << std::endl;

    // Backpropagation: Update weights and biases
    backpropagate(input, y_true, y_pred, weights, biases, learning_rate);

    // Forward pass after backpropagation: Get new predicted output
    output = forward(input, weights, biases);
    y_pred = sigmoid(output); // Apply sigmoid

    // Compute the new loss
    loss = binaryCrossEntropy(y_true, y_pred);
    std::cout << "New Loss After Backpropagation: " << loss << std::endl;

    // Output updated weights and biases
    std::cout << "Updated Weights:" << std::endl;
    for (const auto& row : weights) {
        for (const auto& w : row) {
            std::cout << w << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Updated Biases:" << std::endl;
    for (const auto& b : biases) {
        std::cout << b << " ";
    }
    std::cout << std::endl;

    return 0;
}
