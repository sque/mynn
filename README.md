## My-NN  ⋮∶• Playground for neural networks

A simple and self-descriptive implementation of neural networks with support
of different topology, activation functions and optimization strategies. 

A useful tool if you are interested to discover the intuition of neural networks through experimentation.

### Usage

For quick demo you can see the notebook with a detailed example on how to train and use a NN:
 * [Feedforward NN on binary classification problem](docs/FNN_on_binary_classification_of_a_flower.ipynb)

Or you can see the following snippet on how to train a binary classifier of 2 hidden layers (50, 20 nodes) with `tanh` 
activation function for the hidden layers and `sigmoid` for the output on a dataset
of `m` examples with 10-sized feature vector:

```python
from mynn import FNN
from mynn.activation import TanhActivation, SigmoidActivation


nn = FNN(
    layers_config=[
        (50, TanhActivation),
        (20, TanhActivation),
        (1, SigmoidActivation)
    ],
    n_x=10
)

# To train the vector we need to assure that input
# vector is in (n, m) shape where n is the size of input
# feature vector and m the number of samples
X, Y = load_dataset()

nn.train(X, Y)

# To predict the class you can use the `predict()` method
Y_pred = nn.predict(X_test)

# If you need to take the prediction probability you can
# just perform a forward propagation
Y_proba = nn.forward(X_test)
```

### Activation functions

Library provides abstract interface for activation functions as long it is possible to provide a forward computation 
and a derivative estimation. The package is shipped with 3 implementations of activation functions:

* `ReLU`
* `Tanh`
* `Sigmoid`

### Optimization strategy
The optimization strategy is a core component for any optimization problem. Depending the strategy
the training can be faster, more effective. For this reason it is very beneficial to understand the mechanics behind
each algorithm before choosing one. Sebastian Ruder has published [a very nice blog post](http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms)
where he presents in detail different optimization algorithms.

MyNN is shipped with different implementation of optimization strategies.

* `GradientDescent`: Typical Gradient Descent algorithm
* `GradientDescentMomentum`: It works as the gradient descent but takes
into account the previous grads in order to keep a momentum
* `AdaptiveGradientDescent`: A gradient descent that adapts learning rate per
optimized parameter. 
* `RMSProp`: Gradient descent that adapts learnig rate based on the average of recent magnitudes of the gradients for 
the weight.
* `Adam` (**Default**): Adaptive moment estimation, an algorithm that encapsulates both the ideas
of momentum and RMSProp.

### Weight Initialization

The library supports different ways to initialize the weights of the linear functions. By
default it ships with the following strategies:
* `ConstantWeightInitializer` : It will initialize weight from a predefined list of constant values.
* `NormalWeightInitializer` : It will initialize weights with random values following a normal distribution.
* `VarianceScalingWeightInitializer` : (Default with scale=2) It will initialize weights with random values following a 
normal distribution scaled to ensure a specific variance per layer.


### Regularization
MyNN supports regularization but only one method per train. As it does not distinguish
the stage of training and usage, you need to disable regularization after training
in order to take reasonable results. Currenlty the following algorithms
are supported:
* `L2Regularization`: Regularization on the cost function using L2 norm on the weights of neurons.
* `DropoutRegularization`: Inverse dropout with support for different keep_probability per layer.
### Installation
MyNN can **only** work on `python >= 3.6`. It is proposed to use `virtualenv` to perform
installation.

```sh
$ virtualenv -ppython3.6 venv
$ source venv/bin/activate
```

You can install the dependencies using `requirements.txt` file.
```sh
$ pip install -r requirements.txt
```

