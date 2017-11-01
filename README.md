## My-NN  ⋮∶• Playground for neural networks

A simple and self-descriptive implementation of neural networks with support
of different topology, activation functions and optimization strategies. 

A useful tool if you are interested to understand the intuition of neural networks through experimentation.

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

MyNN is shipped with 3 different implementation of generic optimization strategies.

* `GradientDescent`: Classic gradient descent that works in batches
* `GradientDescentMomentum`: It works as the classic gradient descent but takes
into account the previous grads in order to keep a momentum
* `AdaptiveGradientDescent` (**Default**): A gradient descent that adapts learning rate per
optimized parameter. 


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

### Usage

In the following example you can see how to train a binary classifier of 2 hidden layers (50, 20 nodes) with `tanh` activation function on a dataset
of `m` examples with 10-sized feature vector:

```python
from mynn import NeuralNetwork
from mynn.activation import TanhActivation, SigmoidActivation


nn = NeuralNetwork(
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