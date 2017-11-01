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
