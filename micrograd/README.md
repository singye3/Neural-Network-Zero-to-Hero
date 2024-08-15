# Custom Autograd Engine with Neural Network Implementation

## Overview

This project implements a custom autograd engine using Python and an object-oriented approach. It demonstrates the creation of a simple multi-layer perceptron (MLP) neural network from scratch without using any external libraries for automatic differentiation. The project also includes a gradient checking function and a comparison with PyTorch's autograd engine for verification.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating Values](#creating-values)
  - [Defining Operations](#defining-operations)
  - [Building a Neural Network](#building-a-neural-network)
  - [Training the Neural Network](#training-the-neural-network)
  - [Gradient Checking](#gradient-checking)
- [Implementation Details](#implementation-details)
  - [Value Class](#value-class)
  - [Graph Visualization](#graph-visualization)
  - [Neuron, Layer, and MLP Classes](#neuron-layer-and-mlp-classes)
- [Comparison with PyTorch](#comparison-with-pytorch)
- [License](#license)

## Project Structure

```
.
├── autograd.py
├── neural_network.py
├── main.py
└── README.md
```

- `autograd.py`: Contains the implementation of the `Value` class and the graph visualization functions.
- `neural_network.py`: Contains the implementation of the `Neuron`, `Layer`, and `MLP` classes.
- `main.py`: Contains example usage, training loops, and gradient checking.
- `README.md`: Documentation for the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   Note: `requirements.txt` should include `graphviz`, `numpy`, `matplotlib`, and `torch` (for comparison purposes).

## Usage

### Creating Values

To create a value, instantiate the `Value` class with the desired data and optional parameters for children, operation, and label:

```python
from autograd import Value

x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
```

### Defining Operations

You can perform arithmetic operations between `Value` instances, and these will automatically record the computational graph:

```python
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')

# Forward pass
x1w1 = x1 * w1; x1w1.label = 'x1w1'
x2w2 = x2 * w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
n = x1w1x2w2 + b; n.label = 'n'

# Activation
o = (2 * n).exp() - 1 / ((2 * n).exp() + 1); o.label = 'o'

# Backward pass
o.backward()
```

### Building a Neural Network

Create a neural network using the `Neuron`, `Layer`, and `MLP` classes:

```python
from neural_network import MLP

# Define a multi-layer perceptron with 3 inputs, two hidden layers with 4 neurons each, and 1 output
n = MLP(3, [4, 4, 1])
```

### Training the Neural Network

Train the network using a simple loop and gradient descent:

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]  # Desired targets

for k in range(20):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # Backward pass
    loss.backward()

    # Update parameters
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(f'epoch {k} loss {loss.data}')
```

### Gradient Checking

To check the correctness of the gradients computed by the autograd engine, use the `lol` function:

```python
from autograd import lol

lol()
```

## Implementation Details

### Value Class

The `Value` class is the core of the autograd engine. It represents a node in the computational graph and stores the data, gradient, and the operation that produced the node. The class provides methods for basic arithmetic operations, activation functions, and backward propagation.

### Graph Visualization

The `trace` and `draw_dot` functions allow you to visualize the computational graph. These functions use `graphviz` to generate a visual representation of the graph, showing the nodes and edges, along with the data and gradients:

```python
from autograd import draw_dot

# Example of drawing the computational graph
draw_dot(o)
```

### Neuron, Layer, and MLP Classes

These classes are implemented in an object-oriented manner to create and manage a multi-layer perceptron (MLP) neural network. Each neuron performs a weighted sum of its inputs and applies a non-linear activation function. Layers and MLPs are built by stacking these neurons together.

## Comparison with PyTorch

The project includes a comparison with PyTorch's autograd engine to verify the correctness of the custom implementation. The results from the custom autograd engine and PyTorch's engine are shown to be consistent:

```python
import torch

# Define inputs and weights using PyTorch
x1 = torch.tensor([2.0], requires_grad=True)
x2 = torch.tensor([0.0], requires_grad=True)
w1 = torch.tensor([-3.0], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([6.8813735870195432], requires_grad=True)

n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)

# Backward pass
o.backward()

# Print gradients
print('x1', x1.grad.item())
print('x2', x2.grad.item())
print('w1', w1.grad.item())
print('w2', w2.grad.item())
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
